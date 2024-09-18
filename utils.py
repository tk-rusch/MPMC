import torch
import math
from itertools import combinations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def L2discrepancy(x):
    N = x.size(1)
    dim = x.size(2)
    prod1 = 1. - x ** 2.
    prod1 = torch.prod(prod1, dim=2)
    sum1 = torch.sum(prod1, dim=1)
    pairwise_max = torch.maximum(x[:, :, None, :], x[:, None, :, :])
    product = torch.prod(1 - pairwise_max, dim=3)
    sum2 = torch.sum(product, dim=(1, 2))
    one_dive_N = 1. / N
    out = torch.sqrt(math.pow(3., -dim) - one_dive_N * math.pow(2., 1. - dim) * sum1 + 1. / math.pow(N, 2.) * sum2)
    return out

def hickernell_all_emphasized(x,dim_emphasize):
    nbatch, nsamples, dim = x.size(0), x.size(1), x.size(2)
    mean_disc_projections = torch.zeros(nbatch).to(device)
    for d in dim_emphasize:
        subsets_of_d = list(combinations(range(dim), d))
        for i in range(len(subsets_of_d)):
            set_inds = subsets_of_d[i]
            mean_disc_projections += L2discrepancy(x[:, :, set_inds])

    return mean_disc_projections
