import torch
import math
import numpy as np
from hdbscan import HDBSCAN


def median_aggregation(item_emb_updates):
    if len(item_emb_updates) % 2 == 0 and len(item_emb_updates) > 0:
        sorted = torch.sort(item_emb_updates, dim=0)[0]
        return (sorted[int(len(item_emb_updates) / 2 - 1)] + sorted[int(len(item_emb_updates) / 2)]) / 2
    return torch.median(item_emb_updates, dim=0)[0]

def mean_aggregation(item_emb_updates):
    return torch.mean(item_emb_updates, dim=0)

def trimmed_mean_aggregation(item_emb_updates, filter_out_num):
    item_emb_updates = torch.sort(item_emb_updates, dim=0)[0]
    if int(filter_out_num / 2) > 0:
        return torch.mean(item_emb_updates[int(filter_out_num / 2):-int(filter_out_num / 2)], dim=0)
    else:
        return torch.mean(item_emb_updates, dim=0)

def l2_clip_aggregation(item_emb_updates, clip_value):
    items_emb_updates_norm = torch.linalg.norm(item_emb_updates, dim=1)
    ratio = items_emb_updates_norm / clip_value
    ratio = torch.where(ratio < 1, torch.ones_like(ratio), ratio)
    ratio = ratio.unsqueeze(1)
    item_emb_updates = item_emb_updates / ratio
    return torch.mean(item_emb_updates, dim=0)


def similarity(model_update1, model_update2):
    return torch.norm(model_update1 - model_update2)

def krum_aggregation(item_emb_updates, f):
    if len(item_emb_updates) <= 2:
        return mean_aggregation(item_emb_updates)
    q = f
    if len(item_emb_updates) - f - 2 <= 0:
        q = len(item_emb_updates) - 3
    differences = item_emb_updates.unsqueeze(1) - item_emb_updates.unsqueeze(0)
    distances = torch.norm(differences, dim=-1)
    distances = torch.sort(distances, dim=1)[0]
    num_neighbours = item_emb_updates.shape[0] - 2 - q
    scores = torch.sum(distances[:, 1:num_neighbours + 1], dim=1)
    aggregated_update = item_emb_updates[torch.argmin(scores)]
    return aggregated_update

def flame_aggregation(item_emb_updates, item_emb):
    if len(item_emb_updates) <= 2:
        return mean_aggregation(item_emb_updates)
    
    epsilon = 1e-10
    
    item_emb = item_emb_updates + item_emb.unsqueeze(0)
    norm_item_emb = torch.linalg.norm(item_emb, dim=1, keepdim=True)
    normalized_item_emb = item_emb / (norm_item_emb + epsilon)
    
    cosine_dist_mat = 1 - torch.matmul(normalized_item_emb, normalized_item_emb.t())
    
    clusterer = HDBSCAN(min_cluster_size=int(len(item_emb_updates) / 2) + 1, min_samples=1, metric='precomputed', allow_single_cluster=True)
    cluster_labels = clusterer.fit_predict(cosine_dist_mat.detach().cpu().numpy().astype(np.float64))
    indices = np.where(cluster_labels == 0)[0]
    
    norm_mat = torch.linalg.norm(item_emb_updates, dim=1)
    St = torch.median(norm_mat)
    
    aggregated_update = item_emb_updates[indices] * torch.min(torch.ones_like(norm_mat[indices]), St / (norm_mat[indices] + epsilon)).unsqueeze(1)
    aggregated_update = torch.mean(aggregated_update, dim=0)
    
    sigma = 0.01 * St
    aggregated_update += torch.normal(0, sigma, size=aggregated_update.shape).to(aggregated_update.device)
    
    return aggregated_update






