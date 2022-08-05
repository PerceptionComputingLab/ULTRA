import torch
import numpy as np
from scipy import stats


def compute_score(probs, num_discretizing):
    '''
    Get the real tumor cellularity score based on the predicted distribution.
    probs : predicted probability distribution
    num_discretizing: number of samplings on the distribution.
    '''
    pred = probs.argmax(dim=-1) / (num_discretizing-1)
    return pred


def compute_loss(criterion, probs, target):
    '''
    calculate the Kl-div loss between the predicted distribution and target distribution.
    criterion: the loss function class, e.g.: criterion = nn.KLDivLoss(reduction='batchmean')
    probs: predicted distribution after softmax layer
    target: label distribution generated from the tumor cellularity score. 
    '''
    eps = 1e-7
    loss = criterion(torch.log(probs+eps), target)
    return loss


def compute_target_dist(mean, std=0.01, num_discretizing=100):
    '''
    calculate the target distribution based on the tumor cellularity score. 
    Mean: tumor cellularity score
    std: hyper-parameter of the std for gaussian distribution
    '''
    tmp = stats.norm.pdf(np.linspace(0, 1, num_discretizing), loc=mean, scale=std).astype(np.float32)
    dist= tmp / tmp.sum()
    return dist


if __name__ == '__main__':
    # main()
    mean_val = 0.98
    std = 0.01
    num_discretizing = 100
    tmp = stats.norm.pdf(np.linspace(0, 1, num_discretizing), loc=mean_val, scale=std).astype(np.float32)
    data= tmp / tmp.sum()
    s = 2




