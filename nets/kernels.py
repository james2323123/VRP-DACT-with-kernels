import torch
import math
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# calculates squared euclidean distance between points
def euclidean_matrix(Q, K):
    return (
    torch.sum(Q ** 2, dim=-1).unsqueeze(-1)
    +torch.sum(K ** 2, dim=-1).unsqueeze(-2)
    -2*torch.matmul(Q, K.transpose(-2, -1))
    )

def normalize(G, a=-9, b=9):
    max_vals = torch.max(torch.max(G, dim=-1)[0], dim=-1)[0]
    min_vals = torch.min(torch.min(G, dim=-1)[0], dim=-1)[0]
    return a + ((G - min_vals[..., None, None]) * (b - a) / (max_vals - min_vals)[..., None, None])

def vanilla_attention(Q, K, a=9):
    return torch.matmul(Q, K.transpose(-2, -1))

def cauchy(Q, K, a=9):
    dist = euclidean_matrix(Q,K)
    compatibility = 1 / (1 + dist)
    return compatibility

def normalized_cauchy(Q, K, a=9):
    dist = euclidean_matrix(Q,K)
    compatibility = 1 / (1 + dist)
    return normalize(compatibility, -a, a)

def clipped_cauchy(Q, K, a=9):
    dist = euclidean_matrix(Q,K)
    compatibility = 1 / (1 + dist)
    return torch.clamp(compatibility,-a,a)

def exponential(Q, K, a=9):
    dist = euclidean_matrix(Q,K)
    compatibility = torch.exp(-torch.sqrt(torch.abs(dist)))
    return compatibility

def gaussian(Q, K, a=9):
    dist = euclidean_matrix(Q,K)
    compatibility = torch.exp(-dist)
    return compatibility

kernels = {
    'cauchy' : cauchy,
    'norm_cauchy' : normalized_cauchy,
    'clip_cauchy' : clipped_cauchy,
    'no_kernel' : vanilla_attention
}
