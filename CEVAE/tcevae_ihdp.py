#!/usr/bin/env python
"""TCEVAE (Thermodynamic Causal Effect VAE) on IHDP — TVO over the confounder latent z."""

import torch
import torch.nn as nn
from torch.distributions import Normal, Bernoulli

from datasets import IHDP
from evaluation import Evaluator
from utils import FCNet, get_y0_y1

import numpy as np
import time
from scipy.stats import sem
from argparse import ArgumentParser

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TCEVAE(nn.Module):
    """CEVAE with Thermodynamic Variational Objective (TVO) for the latent z instead of KL."""

    def __init__(self, n_bin, n_cont, d=20, nh=3, h=200, weight_decay=1e-4,
                 S=50, K=10, alpha=2.0):
        super().__init__()
        dimx = n_bin + n_cont
        self.n_bin = n_bin
        self.n_cont = n_cont
        self.d = d

        # Thermodynamic parameters
        self.S = S  # number of importance samples
        self.K = K  # number of integration partitions
        # Power-law lambda schedule: lambda_k = (k/K)^alpha
        # alpha > 1 clusters points near lambda=1 (the data-driven posterior),
        # spending more integration energy where confounding bias separates from signal.
        ks = torch.arange(self.K, dtype=torch.float32)
        self.register_buffer('lambdas', (ks / max(self.K - 1, 1)) ** alpha)

        # ---- Decoder / Generative model (unchanged from CEVAE) ----
        self.px_z_shared = FCNet(d, (nh - 1) * [h], activation=nn.ELU, weight_decay=weight_decay)
        self.px_z_bin = FCNet(h, [h], out_heads=[(n_bin, None)], activation=nn.ELU, weight_decay=weight_decay)
        self.px_z_cont = FCNet(h, [h], out_heads=[(n_cont, None), (n_cont, nn.Softplus())],
                               activation=nn.ELU, weight_decay=weight_decay)
        self.pt_z = FCNet(d, [h], out_heads=[(1, None)], activation=nn.ELU, weight_decay=weight_decay)
        self.py_t0z = FCNet(d, nh * [h], out_heads=[(1, None)], activation=nn.ELU, weight_decay=weight_decay)
        self.py_t1z = FCNet(d, nh * [h], out_heads=[(1, None)], activation=nn.ELU, weight_decay=weight_decay)

        # ---- Encoder / Inference model (unchanged from CEVAE) ----
        self.qt_x = FCNet(dimx, [d], out_heads=[(1, None)], activation=nn.ELU, weight_decay=weight_decay)
        self.qy_xt_shared = FCNet(dimx, (nh - 1) * [h], activation=nn.ELU, weight_decay=weight_decay)
        self.qy_xt0 = FCNet(h, [h], out_heads=[(1, None)], activation=nn.ELU, weight_decay=weight_decay)
        self.qy_xt1 = FCNet(h, [h], out_heads=[(1, None)], activation=nn.ELU, weight_decay=weight_decay)
        self.qz_shared = FCNet(dimx + 1, (nh - 1) * [h], activation=nn.ELU, weight_decay=weight_decay)
        self.qz_t0 = FCNet(h, [h], out_heads=[(d, None), (d, nn.Softplus())],
                           activation=nn.ELU, weight_decay=weight_decay)
        self.qz_t1 = FCNet(h, [h], out_heads=[(d, None), (d, nn.Softplus())],
                           activation=nn.ELU, weight_decay=weight_decay)

    def reparameterize_multi_sample(self, mu, logvar):
        """
        mu, logvar shape: [batch_size, z_dim]
        Output shape: [batch_size, S, z_dim]
        """
        batch_size, z_dim = mu.size()
        mu_expanded = mu.unsqueeze(1).expand(batch_size, self.S, z_dim)
        logvar_expanded = logvar.unsqueeze(1).expand(batch_size, self.S, z_dim)
        std = torch.exp(0.5 * logvar_expanded)
        eps = torch.randn_like(std, device=mu.device)
        return mu_expanded + eps * std

    def compute_tvo_loss(self, log_p_x_given_z, log_p_z, log_q_z_given_x):
        """
        Thermodynamic Variational Objective for the confounder z.
        All inputs shape: [batch_size, S]
        Returns the TVO lower bound on log p(x,t,y) (to be maximized).
        """
        log_weights = log_p_x_given_z + log_p_z - log_q_z_given_x
        log_weights = log_weights.clamp(-100, 100)
        batch_size = log_weights.size(0)
        tvo_integral = torch.zeros(batch_size, device=log_weights.device, dtype=log_weights.dtype)
        for k in range(1, self.K):
            lambda_k = self.lambdas[k]
            lambda_prev = self.lambdas[k - 1]
            delta_lambda = lambda_k - lambda_prev
            scaled_log_weights = lambda_k * log_weights
            log_Z_k = torch.logsumexp(scaled_log_weights, dim=1, keepdim=True)
            normalized_log_weights = scaled_log_weights - log_Z_k
            normalized_weights = torch.exp(normalized_log_weights).detach()
            expected_energy = torch.sum(normalized_weights * log_weights, dim=1)
            tvo_integral = tvo_integral + delta_lambda * expected_energy
        return torch.mean(tvo_integral)

    def _encode(self, x, t, y):
        """Run the inference network; returns distributions qt, qy, qz and (mu_z, logvar_z)."""
        logits_t = self.qt_x(x)
        qt = Bernoulli(logits=logits_t)
        hqy = self.qy_xt_shared(x)
        mu_qy_t0 = self.qy_xt0(hqy)
        mu_qy_t1 = self.qy_xt1(hqy)
        mu_qy = t * mu_qy_t1 + (1.0 - t) * mu_qy_t0
        qy = Normal(mu_qy, torch.ones_like(mu_qy))
        inpt = torch.cat([x, y], dim=1)
        hqz = self.qz_shared(inpt)
        muq_t0, sigmaq_t0 = self.qz_t0(hqz)
        muq_t1, sigmaq_t1 = self.qz_t1(hqz)
        mu_qz = t * muq_t1 + (1.0 - t) * muq_t0
        sigma_qz = (t * sigmaq_t1 + (1.0 - t) * sigmaq_t0).clamp(min=1e-6, max=100.0)
        logvar_qz = 2.0 * torch.log(sigma_qz)
        qz = Normal(mu_qz, sigma_qz)
        return qt, qy, qz, mu_qz, logvar_qz

    def _decode(self, z, t):
        """Run the generative model conditioned on z and t."""
        hx = self.px_z_shared(z)
        x_bin_logits = self.px_z_bin(hx)
        x_cont_mu, x_cont_sigma = self.px_z_cont(hx)
        x_cont_sigma = x_cont_sigma.clamp(min=1e-6, max=100.0)
        t_logits = self.pt_z(z)
        mu_y_t0 = self.py_t0z(z)
        mu_y_t1 = self.py_t1z(z)
        mu_y = t * mu_y_t1 + (1.0 - t) * mu_y_t0
        return x_bin_logits, x_cont_mu, x_cont_sigma, t_logits, mu_y

    def forward(self, x_bin, x_cont, t, y, beta=1.0):
        """
        Compute loss = -(TVO + auxiliary).
        TVO replaces the entire ELBO (reconstruction + prior - posterior).
        Auxiliary terms (q(t|x), q(y|x,t)) are added separately.
        """
        x = torch.cat([x_bin, x_cont], dim=1)
        B = x.shape[0]
        S = self.S

        qt, qy, qz, mu_qz, logvar_qz = self._encode(x, t, y)
        z_multi = self.reparameterize_multi_sample(mu_qz, logvar_qz)  # [B, S, d]

        # Flatten to [B*S, d] for decoding; repeat inputs S times
        z_flat = z_multi.view(B * S, self.d)
        t_flat = t.unsqueeze(1).expand(B, S, 1).reshape(B * S, 1)
        x_bin_flat = x_bin.unsqueeze(1).expand(B, S, self.n_bin).reshape(B * S, self.n_bin)
        x_cont_flat = x_cont.unsqueeze(1).expand(B, S, self.n_cont).reshape(B * S, self.n_cont)
        y_flat = y.unsqueeze(1).expand(B, S, 1).reshape(B * S, 1)

        x_bin_logits, x_cont_mu, x_cont_sigma, t_logits, mu_y = self._decode(z_flat, t_flat)

        # Log probs per sample [B*S] -> reshape to [B, S]
        log_px_bin = Bernoulli(logits=x_bin_logits).log_prob(x_bin_flat).sum(dim=1).view(B, S)
        log_px_cont = Normal(x_cont_mu, x_cont_sigma).log_prob(x_cont_flat).sum(dim=1).view(B, S)
        log_pt = Bernoulli(logits=t_logits).log_prob(t_flat).sum(dim=1).view(B, S)
        log_py = Normal(mu_y, torch.ones_like(mu_y)).log_prob(y_flat).sum(dim=1).view(B, S)
        log_p_x_given_z = log_px_bin + log_px_cont + log_pt + log_py  # [B, S]

        pz = Normal(torch.zeros_like(z_flat), torch.ones_like(z_flat))
        log_p_z = pz.log_prob(z_flat).sum(dim=1).view(B, S)

        mu_qz_exp = qz.loc.unsqueeze(1).expand(B, S, self.d)
        sigma_qz_exp = qz.scale.unsqueeze(1).expand(B, S, self.d)
        qz_exp = Normal(mu_qz_exp, sigma_qz_exp)
        log_q_z_given_x = qz_exp.log_prob(z_multi).sum(dim=2)  # [B, S]

        # TVO bounds log p(x,t,y) from below — we want to maximize it
        tvo_bound = self.compute_tvo_loss(log_p_x_given_z, log_p_z, log_q_z_given_x)

        # Auxiliary terms: encourage q(t|x) and q(y|x,t) to match observations
        log_qt = qt.log_prob(t).sum(dim=1)
        log_qy = qy.log_prob(y).sum(dim=1)
        aux = (log_qt + log_qy).mean()

        # Minimize negative (TVO + aux)
        total_loss = -(beta * tvo_bound + aux)
        return total_loss

    def compute_logp_valid(self, x_bin, x_cont, t, y):
        """Deterministic validation bound (same as CEVAE, use mean z)."""
        x = torch.cat([x_bin, x_cont], dim=1)
        qt, qy, qz, _, _ = self._encode(x, t, y)
        z_mean = qz.mean
        x_bin_logits, x_cont_mu, x_cont_sigma, t_logits, mu_y = self._decode(z_mean, t)
        log_px_bin = Bernoulli(logits=x_bin_logits).log_prob(x_bin).sum(dim=1)
        log_px_cont = Normal(x_cont_mu, x_cont_sigma).log_prob(x_cont).sum(dim=1)
        log_pt = Bernoulli(logits=t_logits).log_prob(t).sum(dim=1)
        log_py = Normal(mu_y, torch.ones_like(mu_y)).log_prob(y).sum(dim=1)
        pz = Normal(torch.zeros_like(qz.loc), torch.ones_like(qz.scale))
        log_pz = pz.log_prob(z_mean).sum(dim=1)
        log_qz = qz.log_prob(z_mean).sum(dim=1)
        logp = (log_px_bin + log_px_cont + log_pt + log_py + log_pz - log_qz).mean().item()
        return logp

    def predict_y0_y1(self, x_bin, x_cont, y_obs):
        """Predict E[y|x,t=0] and E[y|x,t=1] (single sample from q(z|x,t,y))."""
        x = torch.cat([x_bin, x_cont], dim=1)
        n = x.shape[0]
        t0 = torch.zeros(n, 1, device=x.device)
        t1 = torch.ones(n, 1, device=x.device)
        _, _, qz0, _, _ = self._encode(x, t0, y_obs)
        _, _, qz1, _, _ = self._encode(x, t1, y_obs)
        z0 = qz0.rsample()
        z1 = qz1.rsample()
        _, _, _, _, mu_y0 = self._decode(z0, t0)
        _, _, _, _, mu_y1 = self._decode(z1, t1)
        return mu_y0, mu_y1


def l2_penalty(model, weight_decay):
    penalty = 0.0
    for name, param in model.named_parameters():
        if 'weight' in name:
            penalty += (param ** 2).sum()
    return weight_decay * penalty


def main():
    parser = ArgumentParser()
    parser.add_argument('-reps', type=int, default=10)
    parser.add_argument('-earl', type=int, default=10)
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-opt', choices=['adam', 'adamw'], default='adam')
    parser.add_argument('-epochs', type=int, default=100)
    parser.add_argument('-print_every', type=int, default=10)
    parser.add_argument('-S', type=int, default=50, help='Number of importance samples for TVO')
    parser.add_argument('-K', type=int, default=10, help='Number of lambda partitions for TVO')
    parser.add_argument('-beta', type=float, default=1.0, help='Weight on TVO term')
    parser.add_argument('-alpha', type=float, default=2.0, help='Power-law exponent for lambda schedule (>1 clusters near lambda=1)')
    args = parser.parse_args()

    dataset = IHDP(replications=args.reps)
    scores = np.zeros((args.reps, 3))
    scores_test = np.zeros((args.reps, 3))

    d = 20
    lamba = 1e-4
    nh, h = 3, 200

    for i, (train, valid, test, contfeats, binfeats) in enumerate(dataset.get_train_valid_test()):
        print(f'\nReplication {i + 1}/{args.reps}')
        (xtr, ttr, ytr), (y_cftr, mu0tr, mu1tr) = train
        (xva, tva, yva), (y_cfva, mu0va, mu1va) = valid
        (xte, tte, yte), (y_cfte, mu0te, mu1te) = test
        evaluator_test = Evaluator(yte, tte, y_cf=y_cfte, mu0=mu0te, mu1=mu1te)

        perm = binfeats + contfeats
        xtr, xva, xte = xtr[:, perm], xva[:, perm], xte[:, perm]

        xalltr = np.concatenate([xtr, xva], axis=0)
        talltr = np.concatenate([ttr, tva], axis=0)
        yalltr = np.concatenate([ytr, yva], axis=0)
        evaluator_train = Evaluator(
            yalltr, talltr,
            y_cf=np.concatenate([y_cftr, y_cfva], axis=0),
            mu0=np.concatenate([mu0tr, mu0va], axis=0),
            mu1=np.concatenate([mu1tr, mu1va], axis=0),
        )

        ym, ys = np.mean(ytr), np.std(ytr)
        ytr, yva = (ytr - ym) / ys, (yva - ym) / ys
        best_logpvalid = -np.inf

        torch.manual_seed(1)
        np.random.seed(1)

        n_bin, n_cont = len(binfeats), len(contfeats)
        model = TCEVAE(
            n_bin, n_cont, d=d, nh=nh, h=h, weight_decay=lamba,
            S=args.S, K=args.K, alpha=args.alpha,
        ).to(device)

        if args.opt == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=lamba)

        best_state = None
        n_epoch = args.epochs
        n_iter_per_epoch = 10 * int(xtr.shape[0] / 100)
        idx = np.arange(xtr.shape[0])

        def to_tensor(arr):
            return torch.tensor(arr, dtype=torch.float32, device=device)

        xalltr_bin_t = to_tensor(xalltr[:, :n_bin])
        xalltr_cont_t = to_tensor(xalltr[:, n_bin:])
        yalltr_t = to_tensor((yalltr - ym) / ys)
        xte_bin_t = to_tensor(xte[:, :n_bin])
        xte_cont_t = to_tensor(xte[:, n_bin:])
        yte_norm_t = to_tensor((yte - ym) / ys)
        xva_bin_t = to_tensor(xva[:, :n_bin])
        xva_cont_t = to_tensor(xva[:, n_bin:])
        tva_t = to_tensor(tva)
        yva_t = to_tensor(yva)

        for epoch in range(n_epoch):
            model.train()
            avg_loss = 0.0
            t0 = time.time()
            np.random.shuffle(idx)

            for j in range(n_iter_per_epoch):
                batch = np.random.choice(idx, 100)
                x_batch_bin = to_tensor(xtr[batch, :n_bin])
                x_batch_cont = to_tensor(xtr[batch, n_bin:])
                t_batch = to_tensor(ttr[batch])
                y_batch = to_tensor(ytr[batch])

                loss = model(x_batch_bin, x_batch_cont, t_batch, y_batch, beta=args.beta)
                loss = loss + l2_penalty(model, lamba)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                avg_loss += loss.item()

            avg_loss = avg_loss / n_iter_per_epoch

            if epoch % args.earl == 0 or epoch == (n_epoch - 1):
                model.eval()
                with torch.no_grad():
                    logpvalid = model.compute_logp_valid(xva_bin_t, xva_cont_t, tva_t, yva_t)
                if logpvalid >= best_logpvalid:
                    print(f'Improved validation bound, old: {best_logpvalid:.3f}, new: {logpvalid:.3f}')
                    best_logpvalid = logpvalid
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if epoch % args.print_every == 0:
                y0, y1 = get_y0_y1(model, xalltr_bin_t, xalltr_cont_t, yalltr_t, L=1)
                y0, y1 = y0 * ys + ym, y1 * ys + ym
                score_train = evaluator_train.calc_stats(y1, y0)
                rmses_train = evaluator_train.y_errors(y0, y1)
                y0, y1 = get_y0_y1(model, xte_bin_t, xte_cont_t, yte_norm_t, L=1)
                y0, y1 = y0 * ys + ym, y1 * ys + ym
                score_test = evaluator_test.calc_stats(y1, y0)
                print(
                    f"Epoch: {epoch + 1}/{n_epoch}, loss: {avg_loss:.3f}, "
                    f"ite_tr: {score_train[0]:.3f}, ate_tr: {score_train[1]:.3f}, pehe_tr: {score_train[2]:.3f}, "
                    f"rmse_f_tr: {rmses_train[0]:.3f}, rmse_cf_tr: {rmses_train[1]:.3f}, "
                    f"ite_te: {score_test[0]:.3f}, ate_te: {score_test[1]:.3f}, pehe_te: {score_test[2]:.3f}, "
                    f"dt: {time.time() - t0:.3f}"
                )

        if best_state is not None:
            model.load_state_dict(best_state)
        model.to(device)

        y0, y1 = get_y0_y1(model, xalltr_bin_t, xalltr_cont_t, yalltr_t, L=100)
        y0, y1 = y0 * ys + ym, y1 * ys + ym
        score = evaluator_train.calc_stats(y1, y0)
        scores[i, :] = score

        y0t, y1t = get_y0_y1(model, xte_bin_t, xte_cont_t, yte_norm_t, L=100)
        y0t, y1t = y0t * ys + ym, y1t * ys + ym
        score_test = evaluator_test.calc_stats(y1t, y0t)
        scores_test[i, :] = score_test

        print(
            f'Replication: {i + 1}/{args.reps}, '
            f'tr_ite: {score[0]:.3f}, tr_ate: {score[1]:.3f}, tr_pehe: {score[2]:.3f}, '
            f'te_ite: {score_test[0]:.3f}, te_ate: {score_test[1]:.3f}, te_pehe: {score_test[2]:.3f}'
        )

    print('\nTCEVAE model total scores')
    means, stds = np.mean(scores, axis=0), sem(scores, axis=0)
    print(f'train ITE: {means[0]:.3f}+-{stds[0]:.3f}, train ATE: {means[1]:.3f}+-{stds[1]:.3f}, '
          f'train PEHE: {means[2]:.3f}+-{stds[2]:.3f}')
    means, stds = np.mean(scores_test, axis=0), sem(scores_test, axis=0)
    print(f'test ITE: {means[0]:.3f}+-{stds[0]:.3f}, test ATE: {means[1]:.3f}+-{stds[1]:.3f}, '
          f'test PEHE: {means[2]:.3f}+-{stds[2]:.3f}')


if __name__ == '__main__':
    main()
