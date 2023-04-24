#!coding:utf-8
import torch
from torch.nn import functional as F

def kl_div_with_logit(input_logits, target_logits):
    assert input_logits.size()==target_logits.size()
    targets = F.softmax(targets_logits, dim=1)
    return F.kl_div(F.log_softmax(input_logits,1), targets)

def entropy_y_x(logit):
    soft_logit = F.softmax(logit, dim=1)
    return -torch.mean(torch.sum(soft_logit* F.log_softmax(logit,dim=1), dim=1))

def softmax_loss_no_reduce(input_logits, target_logits, eps=1e-10):
    assert input_logits.size()==target_logits.size()
    target_soft = F.softmax(target_logits, dim=1)
    return -torch.sum(target_soft* F.log_softmax(input_logits+eps,dim=1), dim=1)

def softmax_loss_mean(input_logits, target_logits, eps=1e-10):
    assert input_logits.size()==target_logits.size()
    target_soft = F.softmax(target_logits, dim=1)
    return -torch.mean(torch.sum(target_soft* F.log_softmax(input_logits+eps,dim=1), dim=1))

def sym_mse(logit1, logit2):
    assert logit1.size()==logit2.size()
    return torch.mean((logit1 - logit2)**2)

def sym_mse_with_softmax(logit1, logit2):
    assert logit1.size()==logit2.size()
    return torch.mean((F.softmax(logit1,1) - F.softmax(logit2,1))**2)

def mse_with_softmax(logit1, logit2):
    assert logit1.size()==logit2.size()
    return F.mse_loss(F.softmax(logit1,1), F.softmax(logit2,1))

def one_hot(targets, nClass):
    logits = torch.zeros(targets.size(0), nClass).to(targets.device)
    return logits.scatter_(1,targets.unsqueeze(1),1)

def label_smooth(one_hot_labels, epsilon=0.1):
    nClass = labels.size(1)
    return ((1.-epsilon)*one_hot_labels + (epsilon/nClass))

def uniform_prior_loss(logits):
    logit_avg = torch.mean(F.softmax(logits,dim=1), dim=0)
    num_classes, device = logits.size(1), logits.device
    p = torch.ones(num_classes).to(device) / num_classes
    return -torch.sum(torch.log(logit_avg) * p)

def SNP_TNPLoss(x_recon, idxs, total_X, knn_index, M):
    loss = 0.
    for i in range(x_recon.size()[0]):
        nn_idxs = knn_index[idxs[i], :].long()
        # nn_idxs = knn_index[idxs[i], :].astype('int64')
        weight = M[idxs[i], :]

        nn_array = total_X[nn_idxs, :]
        error = torch.linalg.norm(nn_array - x_recon[i, :], axis=1)
        loss += torch.sum(error ** 2 * weight)
    loss /= x_recon.size()[0] * knn_index.shape[1]
    return loss


def SNP_TNPLoss_v2(x_recon, knn_output, w,mean=True):
    bs = x_recon.size()[0]
    k = knn_output.size()[1]
    x_recon = x_recon.reshape(bs, 1, -1)

    res = torch.square(knn_output - x_recon)
    res = torch.sum(res, dim=2).reshape(-1)
    loss = torch.sum(res * w)
    if mean:
        loss /= bs * k
    else:
        loss /= k
    return loss
