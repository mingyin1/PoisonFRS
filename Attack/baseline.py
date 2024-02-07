from FedRec.client import FedRecClient
import torch
import torch.nn as nn
import numpy as np
from parse import args


class BaselineAttackClient(FedRecClient):
    def train_(self, items_emb, reg=0,epoch = None):
        a, b, _ = super().train_(items_emb, reg)
        return a, b, None

    def eval_(self, _items_emb):
        return None, None
class FedRecAttackClient(object):
    def __init__(self, center, train_all_size):
        self._center = center
        self._train_all_size = train_all_size
        self.train_all = None

    def eval_(self, _items_emb):
        return None, None

    @staticmethod
    def noise(shape, std):
        noise = np.random.multivariate_normal(
            mean=np.zeros(shape[1]), cov=np.eye(shape[1]) * std, size=shape[0]
        )
        return torch.Tensor(noise).to(args.device)

    def train_(self, items_emb, std=1e-7, epoch = None):
        self._center.train_(items_emb, args.attack_batch_size)

        with torch.no_grad():
            items_emb_grad = self._center.items_emb_grad

            if self.train_all is None:
                target_items = self._center.target_items
                target_items_ = torch.Tensor(target_items).long().to(args.device)

                p = (items_emb_grad + self.noise(items_emb_grad.shape, std)).norm(2, dim=-1)
                p[target_items_] = 0.
                p = (p / p.sum()).cpu().numpy()
                rand_items = np.random.choice(
                    np.arange(len(p)), self._train_all_size - len(target_items), replace=False, p=p
                ).tolist()

                self.train_all = torch.Tensor(target_items + rand_items).long().to(args.device)

            items_emb_grad = items_emb_grad[self.train_all]
            items_emb_grad_norm = items_emb_grad.norm(2, dim=-1, keepdim=True)
            grad_max = args.grad_limit
            too_large = items_emb_grad_norm[:, 0] > grad_max
            items_emb_grad[too_large] /= (items_emb_grad_norm[too_large] / grad_max)
            items_emb_grad += self.noise(items_emb_grad.shape, std)

            self._center.items_emb_grad[self.train_all] -= items_emb_grad
        return self.train_all, -items_emb_grad*args.lr, None

class FedRecAttackCenter(nn.Module):
    def __init__(self, users_train_ind, target_items, m_item, dim):
        super().__init__()
        self.users_train_ind = users_train_ind
        self.target_items = target_items
        self.n_user = len(users_train_ind)
        self.m_item = m_item
        self.dim = dim

        self._opt = None
        self.items_emb = None
        self.users_emb = None
        self.items_emb_grad = None

        self.users_emb_ = nn.Embedding(self.n_user, dim)
        nn.init.normal_(self.users_emb_.weight, std=0.01)

    @property
    def opt(self):
        if self._opt is None:
            self._opt = torch.optim.Adam([self.users_emb_.weight], lr=args.attack_lr)
        return self._opt

    def loss(self, batch_users):
        users, pos_items, neg_items = [], [], []
        for user in batch_users:
            for pos_item in self.users_train_ind[user]:
                users.append(user)
                pos_items.append(pos_item)
                neg_item = np.random.randint(self.m_item)
                while neg_item in self.users_train_ind[user]:
                    neg_item = np.random.randint(self.m_item)
                neg_items.append(neg_item)
        users = torch.Tensor(users).long().to(args.device)
        pos_items = torch.Tensor(pos_items).long().to(args.device)
        neg_items = torch.Tensor(neg_items).long().to(args.device)

        users_emb = self.users_emb_(users)
        pos_items_emb = self.items_emb[pos_items]
        neg_items_emb = self.items_emb[neg_items]

        pos_scores = torch.sum(users_emb * pos_items_emb, dim=-1)
        neg_scores = torch.sum(users_emb * neg_items_emb, dim=-1)
        loss = -(pos_scores - neg_scores).sigmoid().log().sum()
        return loss

    def train_(self, new_items_emb, batch_size, eps=1e-4, epoch = None):
        new_items_emb = new_items_emb.clone().detach()
        if (self.items_emb is not None) and ((new_items_emb - self.items_emb).abs().sum() < eps):
            return
        self.items_emb = new_items_emb

        if len(self.users_train_ind):
            rand_users = np.arange(self.n_user)
            np.random.shuffle(rand_users)
            total_loss = 0.
            for i in range(0, len(rand_users), batch_size):
                loss = self.loss(rand_users[i: i + batch_size])
                total_loss += loss.cpu().item()
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
            total_loss /= len(rand_users)
        else:
            nn.init.normal_(self.users_emb_.weight, std=0.1)
        self.users_emb = self.users_emb_.weight.clone().detach()

        rand_users = np.arange(self.n_user)
        ignore_users, ignore_items = [], []
        for idx, user in enumerate(rand_users):
            for item in self.target_items:
                if item not in self.users_train_ind[user]:
                    ignore_users.append(idx)
                    ignore_items.append(item)
            for item in self.users_train_ind[user]:
                ignore_users.append(idx)
                ignore_items.append(item)
        ignore_users = torch.Tensor(ignore_users).long().to(args.device)
        ignore_items = torch.Tensor(ignore_items).long().to(args.device)

        with torch.no_grad():
            users_emb = self.users_emb[torch.Tensor(rand_users).long().to(args.device)]
            items_emb = self.items_emb
            scores = torch.matmul(users_emb, items_emb.t())
            scores[ignore_users, ignore_items] = - (1 << 10)
            _, top_items = torch.topk(scores, 10)
        top_items = top_items.cpu().tolist()

        users, pos_items, neg_items = [], [], []
        for idx, user in enumerate(rand_users):
            for item in self.target_items:
                if item not in self.users_train_ind[user]:
                    users.append(user)
                    pos_items.append(item)
                    neg_items.append(top_items[idx].pop())
        users = torch.Tensor(users).long().to(args.device)
        pos_items = torch.Tensor(pos_items).long().to(args.device)
        neg_items = torch.Tensor(neg_items).long().to(args.device)
        users_emb = self.users_emb[users]
        items_emb = self.items_emb.clone().detach().requires_grad_(True)
        pos_items_emb = items_emb[pos_items]
        neg_items_emb = items_emb[neg_items]
        pos_scores = torch.sum(users_emb * pos_items_emb, dim=-1)
        neg_scores = torch.sum(users_emb * neg_items_emb, dim=-1)
        loss = neg_scores - pos_scores
        loss[loss < 0] = torch.exp(loss[loss < 0]) - 1
        loss = loss.sum()
        loss.backward()
        self.items_emb_grad = items_emb.grad

# PipAttack
class Estimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 3)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.loss = nn.CrossEntropyLoss()
    def train(self, input_emb, labels):
        self.optimizer.zero_grad()
        output = self.layers(input_emb)
        loss = self.loss(output, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    def forward(self, input_emb):
        return self.layers(input_emb)
    
class PipAttackClient(nn.Module):
    def __init__(self, dim, target_items):
        super().__init__()
        self.items_emb_grad = None
        self.items_emb = None
        self.user_emb = nn.Embedding(1, dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        self.target_items = target_items
        self._opt = torch.optim.Adam([self.user_emb.weight], lr=args.lr)

    @property
    def opt(self):
        if self._opt is None:
            self._opt = torch.optim.Adam([self.user_emb.weight], lr=args.lr)
        return self._opt
    
    def train_(self, items_emb, epoch = None):
        return self.target_items, -self.items_emb_grad[self.target_items] * args.lr, None
    
    def eval_(self, _items_emb):
        return None, None
    
class PipAttackCenter(nn.Module):
    def __init__(self, target_items, item_popularity):
        super().__init__()
        self.estimator = Estimator()
        self.clients = []
        self.target_items = target_items
        self.item_popularity = item_popularity
        # top 10%: popularity_label=0; top 10% to 45%: popularity_label=1; others: popularity_label=2
        self.popularity_label = torch.zeros(len(item_popularity)).long().to(args.device)
        sorted_item_popularity = np.sort(item_popularity)[::-1]
        threshold1 = sorted_item_popularity[int(len(item_popularity) * 0.1)]
        threshold2 = sorted_item_popularity[int(len(item_popularity) * 0.45)]
        self.popularity_label[item_popularity < threshold1] = 1
        self.popularity_label[item_popularity < threshold2] = 2
        self.cross_entropy = nn.CrossEntropyLoss()

    def loss(self):
        loss_exp = 0
        loss_pop = 0
        top_item = np.argmax(self.item_popularity)
        for client in self.clients:
            item_emb = client.items_emb[self.target_items]
            user_emb = client.user_emb
            scores = torch.matmul(user_emb.weight.data, item_emb.t())
            loss_exp += -torch.sum(torch.log(scores))
            loss_pop += self.cross_entropy(self.estimator(client.items_emb[[top_item]]), self.popularity_label[[top_item]])
        return loss_exp + 60 * loss_pop
    
    def train_(self, items_emb, epoch = None):
        self.items_emb = items_emb.clone().detach()
        for client in self.clients:
            client.items_emb = items_emb.clone().detach().requires_grad_(False)
        item_idx_without_target = [i for i in range(len(items_emb)) if i not in self.target_items]
        self.estimator.train(self.items_emb[item_idx_without_target], self.popularity_label[item_idx_without_target])
        for client in self.clients:
            client.items_emb = items_emb.clone().detach().requires_grad_(True)
        loss = self.loss()
        for client in self.clients:
            client._opt.zero_grad()
        loss.backward()
        for client in self.clients:
            client.items_emb_grad = client.items_emb.grad
            client._opt.step()
        return loss.item()
    
class PSMUAttackClient(nn.Module):
    def __init__(self, dim, target_items):
        super().__init__()
        self.items_emb_grad = None
        self.items_emb = None
        self.user_emb = nn.Embedding(1, dim)
        self.interacted_items = None
        nn.init.normal_(self.user_emb.weight, std=0.01)
        self.target_items = target_items
        self._opt = torch.optim.Adam([self.user_emb.weight], lr=2)
    
    def train_(self, items_emb, epoch = None):
        return self.interacted_items, -self.items_emb_grad[self.interacted_items] * args.lr, None
    
    def eval_(self, _items_emb):
        return None, None
    
class PSMUAttackCenter(nn.Module):
    def __init__(self, target_items):
        super().__init__()
        self.clients = []
        self.target_items = target_items

    def loss_rec(self, items_emb):
        loss_rec = 0
        for client in self.clients:
            pos_items = client.items_emb[client.interacted_items]
            neg_items = client.items_emb[[i for i in range(client.items_emb.shape[0]) if i not in client.interacted_items]]
            neg_items = neg_items[np.random.choice(neg_items.shape[0], pos_items.shape[0], replace=False)]
            pos_scores = torch.sum(client.user_emb.weight * pos_items, dim=-1)
            neg_scores = torch.sum(client.user_emb.weight * neg_items, dim=-1)
            loss_rec += -(pos_scores - neg_scores).sigmoid().log().sum()
        return loss_rec
    
    def train_(self, items_emb, epoch = None):
        self.items_emb = items_emb.clone().detach()
        for client in self.clients:
            nn.init.normal_(client.user_emb.weight, std=0.01)
            client.items_emb = items_emb.clone().detach().requires_grad_(False)
            # interacted items are randomly chosen 30 items along with the target items
            client.interacted_items = []
            while len(client.interacted_items) < 30:
                new_interacted = np.random.randint(0,len(items_emb))
                if new_interacted not in client.interacted_items and new_interacted not in self.target_items:
                    client.interacted_items.append(new_interacted)
        for epoch in range(20):
            for client in self.clients:
                client._opt.zero_grad()
            loss_rec = self.loss_rec(items_emb)
            loss_rec.backward()
            for client in self.clients:
                client._opt.step()
        
        loss_attack = 0
        items_emb = items_emb.clone().detach().requires_grad_(True)
        for client in self.clients:
            client.items_emb = items_emb
            for target_item in self.target_items:
                if target_item in client.interacted_items:
                    continue
                score_target = torch.sum(client.user_emb.weight.data * client.items_emb[[target_item]], dim=-1)
                scores = torch.sum(client.user_emb.weight.data * client.items_emb, dim=-1)
                # make sure the target item is not in the top 5
                scores[target_item] = -1e10
                recommend_items = torch.argsort(scores, descending=True)[:5]
                item_similarity = torch.sum(client.items_emb * client.items_emb[target_item], dim=-1)
                item_similarity[target_item] = 1e-10
                item_similarity[recommend_items] = 1e10
                competitive_items = torch.argsort(item_similarity, descending=True)[:10]
                score_competitive = torch.sum(client.user_emb.weight.data * client.items_emb[competitive_items], dim=-1)
                loss_attack += (score_competitive - score_target).sigmoid().sum()
        
        loss_attack.backward()
        for client in self.clients:
            client.items_emb_grad = client.items_emb.grad
