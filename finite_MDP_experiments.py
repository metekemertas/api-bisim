
import argparse
import math
import numpy as np
import os
import ot
import time
import torch
import torch as th
import warnings

from ot.utils import list_to_array
from ot.backend import get_backend
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import normalized_mutual_info_score
from utils import set_seed_everywhere
from utils import Logger


GI_error_scale = 1e-1


def measure_NMI(d, m):
    S, _ = d.size()
    N_eq = int(S/m)
    assert N_eq == S/m
    d_baseline = th.ones_like(d)
    for i in range(m):
        d_baseline[i*N_eq:(i+1)*N_eq, i*N_eq:(i+1)*N_eq] = 0.
    d.masked_scatter_(th.eye(S).to(d.device).bool(), torch.zeros(S).to(d.device))
    pred_labels = KMedoids(n_clusters=m, metric='precomputed', method='pam').fit(d.cpu().numpy()).labels_
    true_labels = KMedoids(n_clusters=m, metric='precomputed', method='pam').fit(d_baseline.cpu().numpy()).labels_

    nmi = normalized_mutual_info_score(true_labels, pred_labels)

    return nmi


def sinkhorn_knopp_warmstart(
        a, b, M, reg, u_init=None, v_init=None, numItermax=1000, stopThr=1e-9,
        verbose=False, log=False, warn=True):
    a, b, M = list_to_array(a, b, M)
    """
    The same as ot.sinkhorn2 except initializes u and v with u_init and v_init if provided.
    """

    nx = get_backend(M, a, b)

    if len(a) == 0:
        a = nx.full((M.shape[0],), 1.0 / M.shape[0], type_as=M)
    if len(b) == 0:
        b = nx.full((M.shape[1],), 1.0 / M.shape[1], type_as=M)

    # init data
    dim_a = len(a)
    dim_b = b.shape[0]

    if len(b.shape) > 1:
        n_hists = b.shape[1]
    else:
        n_hists = 0

    if log:
        log = {'err': []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    if u_init is not None and v_init is not None:
        u = u_init
        v = v_init
    else:
        if n_hists:
            u = nx.ones((dim_a, n_hists), type_as=M) / dim_a
            v = nx.ones((dim_b, n_hists), type_as=M) / dim_b
        else:
            u = nx.ones(dim_a, type_as=M) / dim_a
            v = nx.ones(dim_b, type_as=M) / dim_b

    K = nx.exp(M / (-reg))

    Kp = (1 / a).reshape(-1, 1) * K

    numerical_errors = False
    for ii in range(numItermax):
        uprev = u
        vprev = v
        KtransposeU = nx.dot(K.T, u)
        v = b / KtransposeU
        u = 1. / nx.dot(Kp, v)

        if (nx.any(KtransposeU == 0)
                or nx.any(nx.isnan(u)) or nx.any(nx.isnan(v))
                or nx.any(nx.isinf(u)) or nx.any(nx.isinf(v))):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            warnings.warn('Warning: numerical errors at iteration %d' % ii)
            numerical_errors = True
            u = uprev
            v = vprev
            break
        if ii % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            if n_hists:
                tmp2 = nx.einsum('ik,ij,jk->jk', u, K, v)
            else:
                # compute right marginal tmp2= (diag(u)Kdiag(v))^T1
                tmp2 = nx.einsum('i,ij,j->j', u, K, v)
            err = nx.norm(tmp2 - b)  # violation of marginal
            if log:
                log['err'].append(err)

            if err < stopThr:
                break
            if verbose:
                if ii % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(ii, err))
    else:
        if warn:
            warnings.warn("Sinkhorn did not converge. You might want to "
                          "increase the number of iterations `numItermax` "
                          "or the regularization parameter `reg`.")
    if log:
        log['niter'] = ii
        if not numerical_errors:
            log['u'] = u
            log['v'] = v
        else:
            # Since these are used to warm-start optimization in the next iteration,
            # do not save (u, v) values that precede numerical errors.
            log['u'] = None
            log['v'] = None

    if n_hists:  # return only loss
        res = nx.einsum('ik,ij,jk,ij->k', u, K, v, M)
        if log:
            return res, log
        else:
            return res

    else:  # return OT matrix
        if log:
            return u.reshape((-1, 1)) * K * v.reshape((1, -1)), log
        else:
            return u.reshape((-1, 1)) * K * v.reshape((1, -1))


def compute_pairwise_wasserstein(P, d, logs=None, p=1., lambda_=1.):
    S, _ = P.size()
    wass = th.zeros((S, S)).to(P.device)
    if d.sum() == 0:
        return d
    d = d.pow(p)
    # Smooth distributions slightly to improve numerical stability.
    P += 1e-7
    P /= P.sum(dim=-1, keepdim=True)

    if math.isinf(lambda_):
        wass = P @ d @ P.transpose(1, 0)
    elif lambda_ == 0:
        P = P.double()
        P /= P.sum(-1, keepdim=True)
        wass = wass.double()
        for i in range(S-1):
            ret = ot.emd2(
                P[i], P[i + 1:].transpose(1, 0), d, log=True)
            wass[i, i + 1:] = th.stack([_[0] for _ in ret])
            log = [_[1]['warning'] for _ in ret]
        wass = wass.float()
    else:
        for i in range(S-1):
            wass[i, i+1:], log = sinkhorn_knopp_warmstart(
                P[i], P[i+1:].transpose(1, 0), d,
                u_init=logs[i]['u'] if logs else None,
                v_init=logs[i]['v'] if logs else None,
                reg=lambda_, numItermax=1000, stopThr=1e-2, log=True)
            if logs is not None:
                logs[i] = {'u': log['u'], 'v': log['v']}

        wass = wass + wass.T

    wass = wass.pow(1/p)
    assert not wass.isnan().any()

    return wass


def sample_uniform_from_simplex(N, n_samples):
    return th.distributions.Dirichlet(th.ones(N) / N).sample((n_samples,))


def sample_prob_with_cuts(N, n_samples):
    cut_points = torch.sort(torch.rand(N-1, n_samples), dim=0)[0]
    a1 = torch.cat([torch.zeros(1, n_samples), cut_points], dim=0)
    a2 = torch.cat([cut_points, torch.ones(1, n_samples)], dim=0)
    probs = a2 - a1
    assert torch.allclose(probs.sum(0), torch.ones_like(probs.sum(0)))
    assert torch.all(probs >= 0)

    return probs.transpose(1, 0)


def R_pi_P_pi(pi, R, P):
    R_pi = (R * pi).sum(-1, keepdim=True)
    P_pi = (P * pi.unsqueeze(1)).sum(-1)

    assert P_pi.sum(dim=-1).allclose(torch.ones(1, device=P_pi.device))

    return R_pi, P_pi


def bisim_fp_iteration(R_pi, P_pi, n, logs, c_T, c_R=1., d_0=None, tol=1e-5, p=1., lambda_=1.):
    S, _ = P_pi.size()
    R_diff = (R_pi - R_pi.squeeze()).abs()
    d = d_0 if d_0 is not None else c_R * R_diff
    for i in range(n):
        T_diff = compute_pairwise_wasserstein(P_pi, d, logs, p=p, lambda_=lambda_)
        d_old = d
        d = c_R * R_diff + c_T * T_diff
        if i > 0 and (d - d_old).abs().max() < tol:
            break
    return d


def value_iteration(R_pi, P_pi, gamma, V_0=None, tol=1e-5):
    S = P_pi.size(0)
    if V_0 is None:
        V_0 = th.zeros(S, 1, device=R_pi.device)
    else:
        assert V_0.size(1) == 1

    error = float('Inf')
    V = V_0
    while error > tol:
        V_old = V
        V = R_pi + gamma * P_pi @ V
        error = (V - V_old).abs().max()

    return V


def T_pi_V(pi, V, R, P, gamma):
    R_pi, P_pi = R_pi_P_pi(pi, R, P)

    return R_pi + gamma * P_pi @ V


def T_V(V, R, P, gamma):
    S, A = R.size()
    assert V.size(1) == 1

    V_next = th.bmm(P.permute(2, 0, 1), V.unsqueeze(0).expand(A, -1, -1)).squeeze(-1).permute(1, 0)
    Q = R + gamma * V_next
    t_v, a_star = Q.max(-1)

    pi_g = th.zeros(S, A, device=R.device)
    pi_g.scatter_(1, a_star.unsqueeze(-1), 1)

    return t_v, pi_g


def policy_evaluation(pi, R, P, gamma, delta_PE=0.):
    assert (pi >= 0).all()
    R_pi, P_pi = R_pi_P_pi(pi, R, P)
    V_pi = value_iteration(R_pi, P_pi, gamma)

    error = 0
    if delta_PE > 0:
        delta_PE = th.distributions.Normal(loc=delta_PE, scale=0.1 * delta_PE).sample()
        noise = th.distributions.Normal(loc=delta_PE, scale=0.1 * delta_PE).sample(V_pi.size())
        V_pi += noise.to(V_pi.device)
        error = noise.abs().max()

    return V_pi, error


def greedy_update(V, R, P, gamma, delta_GI=0):
    global GI_error_scale
    S, A = R.size()

    T_V_, true_pi_g = T_V(V, R, P, gamma)

    if delta_GI == 0:
        pi_g = true_pi_g
    else:
        while True:
            noise = th.distributions.Normal(loc=0, scale=GI_error_scale).sample((S, A)).to(true_pi_g.device).abs()
            pi_g = true_pi_g + noise
            pi_g /= pi_g.sum(dim=-1, keepdim=True)
            T_pi_g_V = T_pi_V(pi_g, V, R, P, gamma).squeeze()
            GI_error = (T_V_ - T_pi_g_V).abs().max()
            if GI_error > delta_GI:
                GI_error_scale /= 1.25
            elif GI_error < delta_GI * 0.5:
                GI_error_scale *= 1.25
            else:
                break

    assert (pi_g >= 0).all()

    return pi_g


def api_no_aggregation(L: Logger, pi_0, R, P, gamma, max_k, delta_PE_init, delta_PE_decay, delta_GI, alpha=1.,
                       tol=1e-5, pi_star=None, V_star=None, log=True):
    pi = pi_0
    V_pi_old = None
    for k in range(max_k):
        delta_PE = delta_PE_init * ((k+1) ** -delta_PE_decay)
        delta_PE = max(delta_PE, 5e-4 * delta_PE_init)
        start_time = time.time()
        V_pi, PE_error = policy_evaluation(pi, R, P, gamma, delta_PE=delta_PE)

        if k > 0:
            delta_V = (V_pi - V_pi_old).abs().max()
            if delta_V < tol:
                break

        pi_g = greedy_update(V_pi, R, P, gamma, delta_GI=delta_GI)
        pi = alpha * pi_g + (1 - alpha) * pi
        V_pi_old = V_pi

        elapsed = time.time() - start_time
        if log:
            if V_star is not None:
                L.log('train/Max_delta_V', (V_pi - V_star).abs().max(), k)
            if pi_star is not None:
                L.log('train/Max_D_TV', 0.5 * (pi_star - pi).abs().sum(-1).max(), k)
            L.log('train/Max_delta_V_pi', PE_error, k)
            L.log('train/duration', elapsed, k)
            L.dump(k)

    print("API (without aggregation) terminated in {} update(s)".format(k-1))
    V_pi, _ = policy_evaluation(pi, R, P, gamma)

    return pi, V_pi


def api_bisim(L: Logger, pi_0, R, P, gamma, n, args, max_k, pi_star=None, V_star=None):
    assert 0 < args.alpha <= 1
    if args.alpha < 1:
        assert args.warm_start
    c_R = args.c_R
    S_e = args.S_e
    pi = pi_0

    # Uncomment below to generate plots like Fig. 8

    # V_pi, _ = policy_evaluation(pi, R, P, gamma)
    # fig, ax = plt.subplots()
    # V_star_sorted, V_star_argsort = th.sort(V_star, dim=0)

    # ax.plot(np.arange(len(V_star)), V_star_sorted.cpu().numpy())
    # ax.plot(np.arange(len(V_pi)), V_pi[V_star_argsort.squeeze()].cpu().numpy())
    # ax.set_ylabel('$V^*(\mathbf{s}_i)$', fontsize='large')
    # ax.set_xlabel('Index $i$', fontsize='large')
    # # ax.set_ylim(bottom=0.0, top=math.ceil(max(V_star).item() * 1.1))
    # plt.savefig('api_bisim_debug/V*.png')
    # plt.close()

    S, A = pi.size()
    d = None
    logs = [{'u': None, 'v': None} for _ in range(S - 1)]
    nmi_baseline = measure_NMI((V_star - V_star.squeeze()).abs(), S_e)
    Phi = th.eye(S).to(P.device)

    for k in range(max_k):
        start_time = time.time()
        R_pi, P_pi = R_pi_P_pi(pi, R, P)

        if args.warm_start:
            d = bisim_fp_iteration(R_pi, P_pi, n, logs, c_R=c_R, c_T=gamma, d_0=d,
                                   tol=args.epsilon/10, p=args.p, lambda_=args.lambda_)
        else:
            d = bisim_fp_iteration(R_pi, P_pi, n, logs, c_R=c_R, c_T=gamma,
                                   tol=args.epsilon/10, p=args.p, lambda_=args.lambda_)

        Phi_new = hard_aggregation(d/c_R, args.epsilon, cap=args.agg_cap)
        if args.agg_cap:
            # Permute rows to find a close configuration to the previous partitioning.
            Phi = reorder_aggregation(Phi_new, Phi)
        else:
            Phi = Phi_new
        R_pi_tilde = Phi @ R_pi / Phi.sum(-1, keepdim=True)
        P_pi_tilde = (Phi @ P_pi / Phi.sum(-1, keepdim=True)) @ Phi.transpose(1, 0)
        V_pi_tilde = value_iteration(R_pi_tilde, P_pi_tilde, gamma)
        V_pi_tilde_Phi = Phi.transpose(1, 0) @ V_pi_tilde

        pi_g = greedy_update(V_pi_tilde_Phi, R, P, gamma, delta_GI=args.delta_GI)
        if args.decay_alpha:
            alpha_e = max(args.alpha, (k+1) ** -0.8)
        else:
            alpha_e = args.alpha
        pi = alpha_e * pi_g + (1 - alpha_e) * pi

        elapsed = time.time() - start_time
        V_pi, _ = policy_evaluation(pi, R, P, gamma)
        _, P_pi = R_pi_P_pi(pi, R, P)

        L.log('train/Max_delta_V', (V_pi - V_star).abs().max(), k)
        L.log('train/Avg_delta_V', (V_pi - V_star).abs().mean(), k)
        L.log('train/Max_D_TV', 0.5 * (pi_star - pi).abs().sum(-1).max(), k)
        L.log('train/Avg_D_TV', 0.5 * (pi_star - pi).abs().sum(-1, keepdim=True).mean(), k)
        L.log('train/Max_delta_V_pi', (V_pi_tilde_Phi - V_pi).abs().max(), k)
        L.log('train/Max_metric_error', ((V_pi - V_pi.squeeze()).abs() - d/c_R).abs().max(), k)
        L.log('train/n_partitions', len(Phi), k)
        L.log('train/NMI', measure_NMI(d, S_e), k)
        L.log('train/NMI_baseline', nmi_baseline, k)
        L.log('train/duration', elapsed, k)
        L.dump(k)

        # Uncomment below to generate plots like Fig. 7

        # if (k + 1) % 10 == 0 and k < 210:
        #     fig, ax = plt.subplots()
        #     flierprops = dict(marker='o',
        #                       markerfacecolor='none',
        #                       # markersize=4,
        #                       markeredgecolor=colors.to_rgba('b', 0.01))
        #     ax.boxplot(d_approx_data,  flierprops=flierprops)
        #     ax.axhline(y=0, color='black', linestyle='--')
        #     ax.set_xticks([y + 1 for y in range(len(d_approx_data))],
        #                   labels=[interval * y if (y % (100 / interval)) == 0 else ''
        #                           for y in range(len(d_approx_data))])
        #     ax.set_xlabel('API steps (k)')
        #     ax.set_ylabel('$\\widehat{d}_\\pi(s, s^\\prime) - \Delta V^\\pi(s, s^\\prime)$')
        #     ax.set_title('$\\widehat{d}_\\pi(s, s^\\prime) - \Delta V^\\pi(s, s^\\prime)$ over API steps')
        #     plt.savefig('api_bisim_debug/boxplot_n{}_env{}_lambda{}.png'.format(n, args.env_id, args.lambda_))
        #     plt.close()
        #
        #     fig, ax = plt.subplots()
        #     V_star_sorted, V_star_argsort = th.sort(V_star, dim=0)
        #     ax.plot(np.arange(len(V_star)), V_star_sorted.cpu().numpy())
        #     ax.plot(np.arange(len(V_pi)), V_pi[V_star_argsort.squeeze()].cpu().numpy())
        #     ax.plot(np.arange(len(V_pi)), V_pi_tilde_Phi[V_star_argsort.squeeze()].cpu().numpy())
        #     ax.set_ylabel('$V(\mathbf{s}_i)$', fontsize='x-large')
        #     ax.set_xlabel('State index $i$', fontsize='large')
        #     ax.legend(['$V^*$', '$V^\pi$', '$\widetilde{V}^\pi_\Phi$'], fontsize='x-large')
        #     # ax.set_ylim(bottom=0.0, top=math.ceil(max(V_star).item() * 1.1))
        #     plt.savefig('api_bisim_debug/V_env{}_{}.png'.format(args.env_id, k+1))
        #     plt.close()

    if args.warm_start:
        print("API({}) bisim terminated in {} update(s)".format(args.alpha, k-1))
    else:
        print("API bisim terminated in {} update(s)".format(k-1))

    print("Last Phi had {} partitions.".format(len(Phi)))
    V_pi, _ = policy_evaluation(pi, R, P, gamma)

    return pi, V_pi, d


def epsilon_aggregate(d, eps):
    device = d.device
    S, _ = d.size()
    n_leq_eps = (d <= 2 * eps).sum(-1)
    assigned = torch.zeros(S, device=device).bool()
    membership = []
    while assigned.sum() < S:
        centroid = n_leq_eps.argmax()
        candidates = th.argwhere((d[centroid] <= 2 * eps) & ~assigned)
        candidates = candidates[th.argsort(n_leq_eps[candidates].squeeze(), descending=True)]
        members = [centroid.item()]
        for candidate in candidates:
            if candidate == centroid or assigned[candidate]:
                continue

            if (d[candidate, members] <= 2 * eps).all():
                members.append(candidate.item())

        members = th.tensor(members, device=device)
        membership.append(members)
        assigned.scatter_(-1, members, 1)
        n_leq_eps -= assigned.long() * S

    return membership


def reorder_aggregation(Phi, Phi_old):
    S_tilde = Phi.size(0)
    device = Phi.device
    if S_tilde != Phi_old.size(0):
        return Phi
    intersection = th.logical_and(Phi.bool().unsqueeze(1), Phi_old.bool()).sum(-1)
    union = th.logical_or(Phi.bool().unsqueeze(1), Phi_old.bool()).sum(-1)
    iou = intersection / union
    sorted_idx = th.argsort(iou, dim=-1, descending=True)
    assigned = th.zeros(S_tilde).bool().to(device)
    Phi_new = th.zeros_like(Phi)
    iterates = th.argsort(intersection.max(-1)[0], descending=True)
    for i in iterates:
        sorted_i = sorted_idx[i]
        for j in sorted_i:
            if not assigned[j]:
                Phi_new[j] = Phi[i]
                assigned[j] = True
                break

    return Phi_new


def hard_aggregation(d, epsilon, cap=None, seed=None):
    S, _ = d.size()

    cap = cap if cap else float('Inf')
    if math.isinf(cap):
        membership = epsilon_aggregate(d, epsilon)
        Phi = torch.zeros(len(membership), S, device=d.device)
        for i, members in enumerate(membership):
            Phi[i].scatter_(0, members, 1)
    else:
        d.masked_scatter_(th.eye(S).to(d.device).bool(), torch.zeros(S).to(d.device))
        membership = KMedoids(n_clusters=cap, metric='precomputed',
                              method='pam', random_state=seed).fit(d.cpu().numpy()).labels_
        Phi = torch.zeros(cap, S, device=d.device)
        Phi.scatter_(0, th.from_numpy(membership).to(d.device).unsqueeze(0), 1)
        Phi = Phi[Phi.sum(1) != 0]

    assert (Phi.sum() == S)
    assert len(Phi) <= cap

    return Phi


def construct_MDP1(S, S_e, A, device, env_noise):
    assert 0 <= env_noise < 1
    N_eq = int(S / S_e)  # number of states per equivalence class.

    P = th.zeros(S, S, A, device=device)
    optimal_a = 0
    for i in range(S_e-1):
        for a in range(A):
            if a == optimal_a:
                # taking the right action transitions you to a state on the correct trajectory
                P[i*N_eq:(i+1)*N_eq, (i+1)*N_eq:(i+2)*N_eq, a] = sample_uniform_from_simplex(N_eq, N_eq)
            else:
                # taking the wrong action boots you to the first partition.
                P[i*N_eq:(i+1)*N_eq, :N_eq, a] = sample_uniform_from_simplex(N_eq, N_eq)

    # once you reach the final (rewarding) state
    for i in range(A):
        if i == optimal_a:
            # Stay there if you take the optimal action
            P[-N_eq:, -N_eq:, i] = sample_uniform_from_simplex(N_eq, N_eq)
        else:
            # Back to the inital state otherwise
            P[-N_eq:, :N_eq, i] = sample_uniform_from_simplex(N_eq, N_eq)

    for i in range(A):
        P[:, :, i] = env_noise * sample_uniform_from_simplex(S, S).to(P.device) + (1-env_noise) * P[:, :, i]

    assert P.sum(dim=1).allclose(torch.ones(1, device=P.device))

    R = th.zeros(S, A, device=device)
    # collect a reward only if you take the optimal action in the last partition.
    R[-N_eq:, optimal_a] = 1.

    return R, P


def construct_MDP2(S, S_e, A, gamma, device, env_noise):
    e = (1 - gamma) / (1 - gamma ** S_e)
    assert 0 <= env_noise < 1
    N_eq = int(S / S_e)  # number of states per equivalence class.

    R = th.zeros(S, A, device=device)
    for i in range(S_e):
        optimal_a = i % A
        R[i*N_eq:(i+1)*N_eq, optimal_a] = e * (1 - gamma ** (i+1)) / (1 - gamma)

    P = th.zeros(S, S, A, device=device)
    for i in range(S_e-1):
        optimal_a = i % A
        for a in range(A):
            if a == optimal_a:
                # taking the right action transitions you to a state on the correct trajectory
                P[i*N_eq:(i+1)*N_eq, (i+1)*N_eq:(i+2)*N_eq, a] = sample_uniform_from_simplex(N_eq, N_eq)
            else:
                # taking the wrong action transitions you to a state in the same partition
                P[i*N_eq:(i+1)*N_eq, i*N_eq:(i+1)*N_eq, a] = sample_uniform_from_simplex(N_eq, N_eq)

    # once you reach the final state, you stay there forever no matter which action you take
    for a in range(A):
        P[-N_eq:, -N_eq:, a] = sample_uniform_from_simplex(N_eq, N_eq)

    for i in range(A):
        P[:, :, i] = env_noise * sample_uniform_from_simplex(S, S).to(P.device) + (1-env_noise) * P[:, :, i]

    assert P.sum(dim=1).allclose(torch.ones(1, device=P.device))

    return R, P


def construct_MDP3(S, S_e, A, device, env_noise):
    assert 0 <= env_noise < 1
    N_eq = int(S / S_e)  # number of states per equivalence class.

    P = th.zeros(S, S, A, device=device)
    for i in range(S_e-1):
        optimal_a = i % A
        for a in range(A):
            if a == optimal_a:
                prob = torch.rand(1) * 0.25
                # taking the right action transitions you to a state on the correct trajectory whp
                P[i*N_eq:(i+1)*N_eq, (i+1)*N_eq:(i+2)*N_eq, a] = (1-prob) * sample_uniform_from_simplex(N_eq, N_eq)
                # or transition to another random state with low probability
                ne = np.random.choice([_ for _ in range(S_e) if _ not in (i, i+1)], size=1, replace=False)[0]
                P[i*N_eq:(i+1)*N_eq, ne*N_eq:(ne+1)*N_eq, a] = prob * sample_uniform_from_simplex(N_eq, N_eq)
            else:
                # taking the wrong action boots you to the first partition.
                P[i*N_eq:(i+1)*N_eq, :N_eq, a] = sample_uniform_from_simplex(N_eq, N_eq)

    # once you reach the final (rewarding) state
    optimal_a = (S_e-1) % A
    for a in range(A):
        if a == optimal_a:
            prob = torch.rand(1) * 0.25
            # Stay there if you take the optimal action whp
            P[-N_eq:, -N_eq:, a] = (1-prob) * sample_uniform_from_simplex(N_eq, N_eq)
            # or transition to a random state with low probability
            ne = np.random.choice([_ for _ in range(S_e-1)], size=1, replace=False)[0]
            P[-N_eq:, ne*N_eq:(ne+1)*N_eq, a] = prob * sample_uniform_from_simplex(N_eq, N_eq)
        else:
            # Back to the inital state otherwise
            P[-N_eq:, :N_eq, a] = sample_uniform_from_simplex(N_eq, N_eq)

    for a in range(A):
        P[:, :, a] = env_noise * sample_uniform_from_simplex(S, S).to(P.device) + (1-env_noise) * P[:, :, a]

    assert P.sum(dim=1).allclose(torch.ones(1, device=P.device))

    R = th.zeros(S, A, device=device)
    # collect a reward only if you take the optimal action in the last partition.
    R[-N_eq:, optimal_a] = 1.

    return R, P


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=0, help='Integer index of CUDA device to use or the str "cpu".')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--env_id', type=int, default=1,
                        help='choose 1, 2 or 3 for three finite MDPs discussed in the paper.')
    parser.add_argument('--k', default=100, type=int, help='Number of steps.')
    parser.add_argument('--n', default=0, type=int, help='Number of fixed-point updates for bisimulation.')
    parser.add_argument('--S', default=200, type=int, help='Size of state space.')
    parser.add_argument('--A', default=2, type=int, help='Size of action space.')
    parser.add_argument('--S_e', default=20, type=int, help='Number of equivalence classes.')
    parser.add_argument('--delta_GI', default=0.1, type=float, help='Approximate greedy improvement error')
    parser.add_argument('--epsilon', default=0.1, type=float, help='Aggregation threshold')
    parser.add_argument('--alpha', default=1., type=float, help='Policy update size. Used only if warm_start is set.')
    parser.add_argument('--lambda_', default=1., type=float, help='Sinkhorn entropic-regularization weight.')
    parser.add_argument('--p', default=1., type=float, help='p-Wasserstein distance.')
    parser.add_argument('--warm_start', action='store_true', help='Warm-start metric learning.')
    parser.add_argument('--use_tb', action='store_true', help='Save TensorBoard')
    parser.add_argument('--decay_alpha', action='store_true', help='Decay alpha at a rate of k^-0.5')
    parser.add_argument('--c_R', default=1.0, type=float)
    parser.add_argument('--agg_cap', nargs='?', type=int)
    parser.add_argument('--gamma', default=0.9, type=float, help='c_T is set to gamma by default.')
    parser.add_argument('--work_dir', default='./api_bisim_debug', type=str)
    parser.add_argument('--env_noise', default=0.0, type=float,
                        help='Mixture weight of random probability vectors into the transition matrix rows.')
    args = parser.parse_args()

    try:
        os.makedirs(args.work_dir)
    except FileExistsError:
        pass
    set_seed_everywhere(args.seed)
    L = Logger(args.work_dir, config='api', use_tb=args.use_tb)

    gamma = args.gamma
    S = args.S
    S_e = args.S_e  # number of equivalence classes.
    assert S % S_e == 0
    A = args.A
    device = args.device if args.device == 'cpu' else 'cuda:{}'.format(args.device)

    if args.env_id == 1:
        R, P = construct_MDP1(S, S_e, A, device, args.env_noise)
    elif args.env_id == 2:
        R, P = construct_MDP2(S, S_e, A, gamma, device, args.env_noise)
    elif args.env_id == 3:
        R, P = construct_MDP3(S, S_e, A, device, args.env_noise)
    else:
        raise ValueError

    if args.n < 1:
        # Choose theoretical default
        n = math.ceil(math.log((1-gamma)/(1+gamma))/math.log(gamma))
    else:
        n = args.n

    pi = th.ones(S, A, device=device) / A
    pi_star, V_star = api_no_aggregation(
        L, pi,  R, P, gamma, args.k, delta_PE_init=0, delta_PE_decay=1, delta_GI=0, log=False)
    V_star, _ = policy_evaluation(pi_star, R, P, gamma)

    # Initialize with MaxEnt policy
    pi = th.ones(S, A, device=device) / A
    pi, V_pi, _ = api_bisim(L, pi, R, P, gamma, n, args, args.k, V_star=V_star, pi_star=pi_star)

    print("||V* - V_pi||: {:.4f}".format((V_pi - V_star).abs().max().item()))
    print("Avg D_TV(pi*, pi): {:.4f}".format(0.5 * (pi_star - pi).abs().sum(-1).mean()))
    print("Max D_TV(pi*, pi): {:.4f}".format(0.5 * (pi_star - pi).abs().sum(-1).max()))


if __name__ == '__main__':
    main()
