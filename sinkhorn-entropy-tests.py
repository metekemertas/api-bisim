
import argparse
import math
import numpy as np
import os
import ot
import torch as th
import torch.nn.functional as F

from matplotlib import colors
from matplotlib import pyplot as plt
from utils import set_seed_everywhere


def sample_distance_matrix(m, n, d=32, max_dist=1.):
    """

    @param m: Number of source vectors
    @param n: Number of sink vectors
    @param d: Dimension of vectors
    @param max_dist: Diameter of the space
    @return: mxn distance matrix
    """
    max_norm = max_dist / 2
    unit_normal = th.distributions.MultivariateNormal(
        loc=th.zeros(d), covariance_matrix=th.eye(d))
    vecs_a = unit_normal.sample((m,))
    vecs_a = F.normalize(vecs_a, dim=-1)
    vecs_a *= max_norm

    vecs_b = unit_normal.sample((n,))
    vecs_b = F.normalize(vecs_b, dim=-1)
    vecs_b *= max_norm

    dist = (vecs_a.unsqueeze(1) - vecs_b).norm(p=2, dim=-1)

    return dist


def entropy_of_joint(w):
    h = -w * (th.log(w + 1e-7))
    if len(w.size()) == 3:
        return h.sum(-1).sum(-1)
    else:
        return h.sum()


def compute_pairwise_wasserstein(mu_1, mu_2s, d, p=1., lambda_=1.):
    d = d.pow(p)

    if math.isinf(lambda_):
        wass = mu_1 @ d @ mu_2s.transpose(1, 0)
    else:
        wass, log = ot.sinkhorn2(mu_1.squeeze(0), mu_2s.transpose(1, 0), d,
                                 reg=lambda_, warn=True, log=True, numItermax=10000, stopThr=1e-5)

    wass = wass.pow(1/p)
    assert not wass.isnan().any()

    return wass


def sample_uniform_from_simplex(N, n_samples, min_entropy=None, max_entropy=None):
    if min_entropy is None or max_entropy is None:
        return th.distributions.Dirichlet(th.ones(N) / N).sample((n_samples,))
    else:
        assert min_entropy is not None and max_entropy is not None
        dists = th.distributions.Dirichlet(th.ones(N) / N).sample((n_samples,))
        h_dists = -(dists * th.log2(dists)).sum(-1)
        dists = dists[th.logical_and(min_entropy < h_dists, h_dists < max_entropy)]
        while len(dists) < n_samples:
            new_dists = th.distributions.Dirichlet(th.ones(N) / N).sample((5000,))
            h_dists = -(new_dists * th.log2(new_dists)).sum(-1)
            while h_dists.max() < min_entropy:
                new_dists = 0.75 * new_dists + 0.25 * (th.ones_like(new_dists) / N)
                h_dists = -(new_dists * th.log2(new_dists)).sum(-1)
            new_dists = new_dists[th.logical_and(min_entropy < h_dists, h_dists < max_entropy)]
            dists = th.cat([dists, new_dists], dim=0)

        return dists[:n_samples]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', default=32, type=int, help='Probability vector size (number of points in ambient space).')
    parser.add_argument('--dim_dist', default=32, type=int, help='Dimensionality of the ambient space.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lambda_1', default=float('inf'), type=float)
    parser.add_argument('--lambda_2', default=0.02, type=float, help='Sinkhorn entropic-regularization weight.')
    parser.add_argument('--p', default=1., type=float, help='p-Wasserstein distance.')
    parser.add_argument('--work_dir', default='./api_bisim_debug', type=str)
    args = parser.parse_args()

    try:
        os.makedirs(args.work_dir)
    except FileExistsError:
        pass

    set_seed_everywhere(args.seed)
    max_entropy = math.floor(math.log2(args.dim))
    min_ent_l = np.linspace(0, max_entropy - 0.5, 10)
    i_range_min = 0.01
    min_ent_u = min_ent_l + i_range_min
    i_range_max = 0.2
    Xs, Ys, Zs = [], [], []
    for i in range(len(min_ent_l)):
        for _ in range(20):
            mu1 = sample_uniform_from_simplex(args.dim, 1, min_entropy=min_ent_l[i], max_entropy=min_ent_u[i])
            h_mu1 = -(mu1 * th.log2(mu1)).sum(-1)
            max_ent_l = np.linspace(h_mu1.item(), max_entropy - i_range_max, 11)
            max_ent_u = max_ent_l + i_range_max
            mu2s = []
            for j in range(len(max_ent_l)):
                D = sample_distance_matrix(m=args.dim, n=args.dim, d=args.dim_dist)
                mu2 = sample_uniform_from_simplex(
                    args.dim, 50, min_entropy=max_ent_l[j], max_entropy=max_ent_u[j])
                mu2s.append(mu2)
            mu2s = th.cat(mu2s, dim=0)

            h_mu2s = -(mu2s * th.log2(mu2s)).sum(-1)

            sinkhorn_small = compute_pairwise_wasserstein(mu1, mu2s, D, lambda_=args.lambda_2)
            sinkhorn_large = compute_pairwise_wasserstein(mu1, mu2s, D, lambda_=args.lambda_1)
            error = (sinkhorn_large - sinkhorn_small)/sinkhorn_small
            Xs.append(h_mu1.repeat(len(mu2s)))
            Ys.append(h_mu2s)
            Zs.append(error)

    fig, ax = plt.subplots()
    flierprops = dict(marker='o',
                      markerfacecolor='none',
                      markeredgecolor=colors.to_rgba('b', 0.005))
    ax.boxplot(th.cat(Zs).flatten().view(len(min_ent_l), -1), flierprops=flierprops)
    ax.set_xticks(np.arange(1, len(min_ent_l) + 1),
                  labels=[_ if _.is_integer() else '' for _ in min_ent_l])
    ax.set_xlabel('$\min(\mathcal{H}(\mu_1), \mathcal{H}(\mu_2))$', fontsize='x-large')
    ax.set_ylabel('$(W^{\lambda}_1 - W^{\lambda^\prime}_1)/W^{\lambda^\prime}_1$', fontsize='x-large')
    l1_text = args.lambda_1 if not math.isinf(args.lambda_1) else '\infty'
    ax.set_title(
        'Marginal Entropy vs. Relative Error ($\lambda={}, \lambda^\prime={}$ )'.format(
            l1_text, args.lambda_2),
        fontsize='x-large')
    plt.savefig(os.path.join(args.work_dir, 'wasserstein_entropy_{}_l{}.pdf'.format(args.dim_dist, args.lambda_1)))


if __name__ == '__main__':
    main()
