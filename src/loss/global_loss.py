import torch
import torch.nn.functional as F

def triplet_margin_loss(query, positive, negative, margin=0.1):
    distance_positive = torch.norm(query - positive, dim=1, p=2)**2
    distance_negative = torch.norm(query - negative, dim=1, p=2)**2
    losses = torch.relu(distance_positive - distance_negative + margin)
    loss = torch.mean(losses)
    return loss

# def triplet_margin_loss(query, positive, negative, margin=0.1):
#     distance_positive = F.cosine_similarity(query, positive, dim=1)
#     distance_negative = F.cosine_similarity(query, negative, dim=1)
#     losses = torch.relu(distance_negative - distance_positive + margin)
#     loss = torch.mean(losses)
#     return loss
