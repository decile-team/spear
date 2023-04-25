import torch
from torch.distributions.beta import Beta
import math

def probability_y(pi_y):
    pi = torch.exp(pi_y)
    return pi / pi.sum()


def phi(theta, l, device):
    return theta * torch.abs(l).double()


def calculate_normalizer(theta, k, n_classes, device):
    z = 0
    for y in range(n_classes):
        m_y = torch.exp(phi(theta[y], torch.ones(k.shape, device=device), device))
        z += (1 + m_y).prod()
    return z


def probability_l_y(theta, l, k, n_classes, weights, device):
    probability = torch.zeros(l.shape[0], n_classes, device=device)
    z = calculate_normalizer(weights.view(1, -1)*theta, k, n_classes, device)
    for y in range(n_classes):
        probability[:, y] = torch.exp(phi(weights.view(1, -1)*theta[y], l, device).sum(1)) / z
    return probability.double()


def probability_s_given_y_l(pi, s, y, l, k, continuous_mask, weights, ratio_agreement=0.85, model=1, theta_process=2):
    eq = torch.eq(k.view(-1, 1), y).double().t()
    r = ratio_agreement * eq.squeeze() + (1 - ratio_agreement) * (1 - eq.squeeze())
    params = torch.exp(pi)
    probability = 1
    for i in range(k.shape[0]):
        m = Beta(weights[i]* (r[i] * params[i] - 1) +1, weights[i]*((params[i] * (1 - r[i]))-1) + 1)
        probability *= (torch.exp(m.log_prob(s[:, i].double())) * l[:, i].double() + (1 - l[:, i]).double()) * continuous_mask[i] + (1 - continuous_mask[i])
    return probability


def probability(theta, pi_y, pi, l, s, k, n_classes, continuous_mask, weights, device):
    p_l_y = probability_l_y(theta, l, k, n_classes, weights, device)
    p_s = torch.ones(s.shape[0], n_classes, device=device).double()
    for y in range(n_classes):
        p_s[:, y] = probability_s_given_y_l(pi[y], s, y, l, k, continuous_mask, weights)
    return p_l_y * p_s
    # print((prob.T/prob.sum(1)).T)
    # input()
    # return prob
    # return (prob.T/prob.sum(1)).T


def log_likelihood_loss(theta, pi_y, pi, l, s, k, n_classes, continuous_mask, weights, device):
    eps = 1e-8
    return - torch.log(probability(theta, pi_y, pi, l, s, k, n_classes, continuous_mask, weights, device).sum(1)).sum() / s.shape[0]


def log_likelihood_loss_supervised(theta, pi_y, pi, y, l, s, k, n_classes, continuous_mask, weights, device):
    eps = 1e-8
    prob = probability(theta, pi_y, pi, l, s, k, n_classes, continuous_mask, weights, device)
    prob = (prob.t() / prob.sum(1)).t()
    return torch.nn.NLLLoss()(torch.log(prob), y)

    
def precision_loss(theta, k, n_classes, a, weights, device):
    n_lfs = k.shape[0]
    prob = torch.ones(n_lfs, n_classes, device=device).double()
    z_per_lf = 0
    for y in range(n_classes):
        m_y = torch.exp(phi(theta[y] * weights, torch.ones(n_lfs, device=device), device))
        #m_y = torch.exp(phi(theta[y], torch.ones(n_lfs, device=device), device))
        per_lf_matrix = torch.tensordot((1 + m_y).view(-1, 1), torch.ones(m_y.shape, device=device).double().view(1, -1), 1) - torch.eye(n_lfs, device=device).double()
        prob[:, y] = per_lf_matrix.prod(0).double()
        z_per_lf += prob[:, y].double()
    prob /= z_per_lf.view(-1, 1)
    correct_prob = torch.zeros(n_lfs, device=device)
    for i in range(n_lfs):
        correct_prob[i] = prob[i, k[i]]
    loss = (1 / math.exp(1)) * (a * torch.log(correct_prob).double() + (1 - a) * torch.log(1 - correct_prob).double())
    #loss = (1/math.exp(1)) * (a * torch.exp(weights) * torch.log(correct_prob).double() + (1 - a) *  torch.exp(weights) * torch.log(1 - correct_prob).double())
    return -loss.sum()

# def precision_loss(theta, k, n_classes, a, weights, device):
#     n_lfs = k.shape[0]
#     prob = torch.ones(n_lfs, n_classes, device=device).double()
#     z_per_lf = 0
#     for y in range(n_classes):
#         #m_y = torch.exp(phi(theta[y] * weights, torch.ones(n_lfs)))
#         m_y = torch.exp(phi(theta[y], torch.ones(n_lfs, device=device), device))
#         per_lf_matrix = torch.tensordot((1 + m_y).view(-1, 1), torch.ones(m_y.shape, device=device).double().view(1, -1), 1) - torch.eye(n_lfs, device=device).double()
#         prob[:, y] = per_lf_matrix.prod(0).double()
#         z_per_lf += prob[:, y].double()
#     prob /= z_per_lf.view(-1, 1)
#     correct_prob = torch.zeros(n_lfs, device=device)
#     for i in range(n_lfs):
#         correct_prob[i] = prob[i, k[i]]
#     #loss = (1 / math.exp(1)) * (a * torch.log(correct_prob).double() + (1 - a) * torch.log(1 - correct_prob).double())
#     loss = (1/math.exp(1)) * (a * torch.exp(weights) * torch.log(correct_prob).double() + (1 - a) *  torch.exp(weights) * torch.log(1 - correct_prob).double())
#     return -loss.sum()

    # n_lfs = k.shape[0]
    # prob = torch.ones(n_lfs, n_classes).double()
    # z_per_lf = 0
    # for y in range(n_classes):
    #     m_y = torch.exp(phi(theta[y] * weights, torch.ones(n_lfs)))
    #     per_lf_matrix = torch.tensordot((1 + m_y).view(-1, 1), torch.ones(m_y.shape).double().view(1, -1), 1) - torch.eye(n_lfs).double()
    #     prob[:, y] = per_lf_matrix.prod(0).double()
    #     z_per_lf += prob[:, y].double()
    # prob /= z_per_lf.view(-1, 1)
    # correct_prob = torch.zeros(n_lfs)
    # for i in range(n_lfs):
    #     correct_prob[i] = prob[i, k[i]]
    # loss = a * torch.log(correct_prob).double() + (1 - a) * torch.log(1 - correct_prob).double()
    # return -loss.sum()
