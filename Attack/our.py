import torch
import numpy as np
from parse import args
import torch.nn as nn
import math

class OurAttackClient(nn.Module):
    Lambda = args.Lambda
    stop_adjust_lambda = False
    start_supervise = False
    PreDist = [-1]
    first_occupied = 0
    def __init__(self, target_items):
        super().__init__()
        self.model_updates = torch.tensor([]).to(args.device)
        self.target_model = None
        self.target_items = target_items
        self.pre_dist = -1 # related to the adapt parameter
        self.step = args.alpha
        self.alpha = args.alpha
        self.is_first = 0

    def eval_(self, _items_emb):
        return None, None
    
    def compute_k_popularities(self, k, items_emb):
        norms = torch.norm(items_emb, dim=1)
        self.k_popularities = torch.argsort(norms)[:k]
        average = torch.mean(items_emb, axis=0)
        inner_product = torch.sum(items_emb * average, axis=1)
        self.k_popularities = torch.argsort(inner_product)[:args.k]

    def train_(self, items_emb, epoch = None):
        with torch.no_grad():
            if epoch >= args.attack_round + 1 and self.target_model is None:
                self.compute_k_popularities(args.k, items_emb)
                top_k_embedding = items_emb[self.k_popularities]
                self.average_top_k_embedding = torch.mean(top_k_embedding, axis=0)
                self.target_model = items_emb.clone()
            if epoch < args.attack_round + 1:
                return None, None, None
            if self.__class__.first_occupied == 0:
                self.is_first = 1
                self.__class__.first_occupied = 1
            self.target_model[self.target_items] = self.average_top_k_embedding * self.__class__.Lambda
            if args.adapt and not self.__class__.stop_adjust_lambda:
                if len(self.__class__.PreDist) < epoch - args.attack_round + 1:
                    self.__class__.PreDist.append(torch.linalg.norm(items_emb[self.target_items] - self.target_model[self.target_items]))
                    self.__class__.Lambda = min(200, self.__class__.Lambda * 2)
                    if self.__class__.start_supervise and self.__class__.PreDist[-2]/self.__class__.PreDist[-1] > 1:
                        self.__class__.stop_adjust_lambda = True
                    print(self.__class__.Lambda, self.__class__.PreDist[-2], self.__class__.PreDist[-1])
                    if self.__class__.PreDist[-2]/self.__class__.PreDist[-1] > 1:
                        self.__class__.start_supervise = True
                
            self.target_model[self.target_items] = self.average_top_k_embedding * self.__class__.Lambda
            print(np.linalg.norm(items_emb[self.target_items].cpu().detach()), np.sort(np.linalg.norm(items_emb.cpu().detach(),axis=1))[-2], np.min(np.linalg.norm(items_emb.cpu().detach(),axis=1)), np.mean(np.linalg.norm(items_emb.cpu().detach(),axis=1)), np.std(np.linalg.norm(items_emb.cpu().detach(),axis=1)))
            items_emb_model_update = self.target_model - items_emb
            items_emb_model_update[self.target_items] *= self.alpha
            if args.items_limit == len(self.target_items):
                return self.target_items, items_emb_model_update[self.target_items], None
            chosen_items = torch.argsort(torch.norm(items_emb_model_update, dim=1), descending=True)[:args.items_limit - len(self.target_items)]
            # erase target items from chosen_items
            chosen_items = torch.tensor(list(set(chosen_items.tolist()) - set(self.target_items))).to(args.device)
            log_chosen_items = chosen_items.cpu().detach().tolist()
            log_chosen_items = [str(item) for item in log_chosen_items]
            if self.is_first == 1:
                with open('filler_items.txt', 'a') as fp:
                    fp.write(','.join(log_chosen_items)+'\n')
            # add target items to chosen_items
            chosen_items = torch.cat((chosen_items, torch.tensor(self.target_items).to(args.device)), dim=0)
            noise = torch.randn(items_emb_model_update[chosen_items].size())
            # Calculate the mean and standard deviation of each row
            row_means = noise.mean(dim=1, keepdim=True)
            row_stddevs = noise.std(dim=1, keepdim=True)

            # Make sure the mean is 0 and standard deviation is 1 for each row
            normalized_noise = (noise - row_means) / row_stddevs

            # Add the normalized noise to each row of the matrix
            items_emb_model_update[chosen_items] = items_emb_model_update[chosen_items] + normalized_noise.to(args.device)
        return chosen_items, items_emb_model_update[chosen_items], None
