import torch
import torch.nn as nn
from parse import args
from .aggregation import *
from collections import defaultdict
import numpy
class FedRecServer(nn.Module):
    def __init__(self, m_item, dim):
        super().__init__()
        self.m_item = m_item
        self.dim = dim
        self.items_emb = nn.Embedding(m_item, dim)
        nn.init.normal_(self.items_emb.weight, std=0.01)
        self.update_recorded = False
        if args.aggregation == 'hics':
            self.gradient_bank = torch.zeros_like(self.items_emb.weight).to(args.device)

    def aggregate(self, item_emb_updates, item_emb):
        if item_emb_updates == []:
            return torch.zeros_like(self.items_emb.weight[0])
        if args.aggregation == 'median':
            return median_aggregation(item_emb_updates)
        elif args.aggregation == 'mean':
            return mean_aggregation(item_emb_updates)
        elif args.aggregation == 'trim':
            return trimmed_mean_aggregation(item_emb_updates, args.clients_limit * item_emb_updates.shape[0])
        elif args.aggregation == 'clip':
            return l2_clip_aggregation(item_emb_updates, args.grad_limit)
        elif args.aggregation == 'krum':
            return krum_aggregation(item_emb_updates, math.ceil(args.clients_limit * len(item_emb_updates)))
        elif args.aggregation == 'flame':
            return flame_aggregation(item_emb_updates, item_emb)
        elif args.aggregation == 'hics':
            return mean_aggregation(item_emb_updates)
    def train_(self, clients, batch_clients_idx, epoch):
        if args.aggregation == 'hics':
            batch_loss = []
          
            items_emb_updates = defaultdict(list)
            updated_item_set = set()

            for idx in batch_clients_idx:
                client = clients[idx]
                items, items_emb_update, loss = client.train_(self.items_emb.weight, epoch = epoch)
                if items is None or items_emb_update is None:
                    continue
                for item, item_emb_update in zip(items, items_emb_update):
                  
                    if type(item) == int:
                        items_emb_updates[item].append(item_emb_update)
                        updated_item_set.add(item)
                    else:
                        items_emb_updates[item.item()].append(item_emb_update)
                        updated_item_set.add(item.item())
                

                if loss is not None:
                    batch_loss.append(loss)
            updated_item_set = list(updated_item_set)
            items_emb_updates = [torch.stack(items_emb_updates[item]) for item in updated_item_set]
            with torch.no_grad():
                items_emb_updates = torch.stack([self.aggregate(items_emb_updates[i], self.items_emb.weight[item]) for i, item in enumerate(updated_item_set)])
        
            items_emb_updates_norm = torch.linalg.norm(items_emb_updates, dim=1)   
            ratio = items_emb_updates_norm / 3
            ratio = torch.where(ratio < 1, torch.ones_like(ratio), ratio)
            ratio = ratio.unsqueeze(1)
            items_emb_updates = items_emb_updates / ratio
            self.gradient_bank[updated_item_set] += items_emb_updates
            gradient_magnitudes = torch.linalg.norm(self.gradient_bank, dim=1)
            top_magnitude_items = torch.topk(gradient_magnitudes, int(1 * self.m_item), dim=0)[1]
            top_magnitude_gradients = self.gradient_bank[top_magnitude_items]
            self.gradient_bank[top_magnitude_items] -= top_magnitude_gradients
            with torch.no_grad():
                self.items_emb.weight[top_magnitude_items]+=top_magnitude_gradients
            return batch_loss
        else:
            batch_loss = []
            items_emb_updates = defaultdict(list)
            updated_item_set = set()
            fake_item_emb_updates = []
            genuine_item_emb_updates = []
            for idx in batch_clients_idx:
                client = clients[idx]
                items, items_emb_update, loss = client.train_(self.items_emb.weight, epoch = epoch)
                if items is None or items_emb_update is None:
                    continue
                for item, item_emb_update in zip(items, items_emb_update):
                    if item == 25601:
                        if idx>=14575:
                            fake_item_emb_updates.append(item_emb_update.cpu())
                        else:
                            genuine_item_emb_updates.append(item_emb_update.cpu())
                    if type(item) == int:
                        items_emb_updates[item].append(item_emb_update)
                        updated_item_set.add(item)
                    else:
                        items_emb_updates[item.item()].append(item_emb_update)
                        updated_item_set.add(item.item())

                if loss is not None:
                    batch_loss.append(loss)  
            fake_item_emb_updates = np.array(fake_item_emb_updates)
            genuine_item_emb_updates = np.array(genuine_item_emb_updates)

            np.save(f"fake/fake_item_emb_updates_{epoch}.npy", fake_item_emb_updates)
            np.save(f"genuine/genuine_item_emb_updates_{epoch}.npy", genuine_item_emb_updates)

            updated_item_set = list(updated_item_set)
            items_emb_updates = [torch.stack(items_emb_updates[item]) for item in updated_item_set]
            with torch.no_grad():
                batch_items_emb_update = torch.stack([self.aggregate(items_emb_updates[i], self.items_emb.weight[item]) for i, item in enumerate(updated_item_set)])
                self.items_emb.weight[updated_item_set]+=batch_items_emb_update
            return batch_loss

    def eval_(self, clients, batch_clients_idx):
        test_cnt = 0
        test_results = 0.
        target_cnt = 0
        target_results = 0.

        for idx in batch_clients_idx:
            client = clients[idx]
            test_result, target_result = client.eval_(self.items_emb.weight)
            if test_result is not None:
                test_cnt += 1
                test_results += test_result
            if target_result is not None:
                target_cnt += 1
                target_results += target_result
        return test_results / test_cnt, target_results / target_cnt
