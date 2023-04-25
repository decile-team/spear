import torch
import numpy as np
from torch import optim
from sklearn.metrics import f1_score
import torch.nn as nn
from torch.nn import MultiLabelMarginLoss as mlml
from random import sample

def svm_supervised(w, b, y, x):
    y = torch.tensor(y)#2 * y - 1)
    #print('y.shape ',y[0])
    #y = torch.reshape(y, (list(y.size())[0],1))
#    print('y.shape ',y.shape)
    #y = nn.functional.pad(y,pad=(0,41), value=-1)
    #print('y.shape afte rpadd',y.shape)
    #y = y.long()
    # print('x.shape ',x.shape)
    # print('w.shape ',w.shape)
    # print('b.shape ',b.shape)
    wx = x.matmul(w) + b
 #   print('wx.shape ',wx.shape)
    #print('y.shape ',y[0])
    #wx = wx.float()
    #lo =  mlml()
    #loss = lo(wx,y)
    #print(loss)
    #return loss
    # print('wxx shape', y)
    ywx = (wx*y[:,None]).sum(1)
  #  print('ywx shap', ywx.shape)
    sy = torch.tensor([0] * y.shape[0]).double()
    #stacked_tensor = torch.stack([1 - ywx, sy])
    sty = 1-(ywx - wx)
    mx = torch.max(sty, sy)
    #print('stacked ka shape', mx.shape)#max(0)[0].sum())
    return torch.sum(mx, 0)/y.shape[0] + torch.norm(w)*torch.norm(w)
    #return 0.5 * torch.norm(w) * torch.norm(w) + stacked_tensor.max(0)[0].sum() / y.shape[0]


def entropy(probabilities):
	entropy = - (probabilities * torch.log(probabilities)).sum(1)
	return entropy.sum() / probabilities.shape[0]

def entropy_pre(probabilities):
	entropy = - (probabilities * torch.log(probabilities)).sum(1)
	return entropy/ probabilities.shape[0]


def cross_entropy(probabilities, y):
    return - torch.log(probabilities[:, y].sum() / y.shape[0])


def kl_divergence(probs_p, probs_q):
    return (probs_p * torch.log(probs_p / probs_q)).sum() / probs_p.shape[0]

def vat_loss(model, x, y, xi=1e-6, eps=2.5, n_iters=1):
    d = torch.Tensor(x.shape).double().normal_()
    for i in range(n_iters):
        d = xi * _l2_normalize(d)
        d = Variable(d, requires_grad=True)
        y_hat = model(x + d)
        delta_kl = kl_div_with_logit(y.detach(), y_hat)
        delta_kl.backward()

        d = d.grad.data.clone().cpu()
        model.zero_grad()

    d = _l2_normalize(d)
    d = Variable(d)
    r_adv = eps * d
    # compute lds
    y_hat = model(x + r_adv.detach())
    delta_kl = kl_div_with_logit(y.detach(), y_hat)
    return delta_kl


def kl_div_with_logit(q_logit, p_logit):

    q = F.softmax(q_logit, dim=1)
    logq = F.log_softmax(q_logit, dim=1)
    logp = F.log_softmax(p_logit, dim=1)

    qlogq = ( q *logq).sum(dim=1).mean(dim=0)
    qlogp = ( q *logp).sum(dim=1).mean(dim=0)

    return qlogq - qlogp

def getDiverseInstances(ent, budget,n_lfs, count): #ent is dict of {lfs, [indices]}
	if count < budget :
		print("budget cannot exceed total instances")
		return 
	each = int(budget/n_lfs)
	print('each is ', each)
	bud = each * n_lfs
	indic = []
	for j in ent.keys():
		if each > len(ent[j]):
			indic.extend(ent[j])
		else:
			#print(type(ent[j]))
			x = sample(list(ent[j]), each)
			#print(x, each)
			indic.extend(x)

	return indic
