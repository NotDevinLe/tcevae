#!/usr/bin/env python
"""CEVAE model on IHDP — PyTorch implementation"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Bernoulli, kl_divergence

from datasets import IHDP
from evaluation import Evaluator
from utils import FCNet, get_y0_y1

import numpy as np
import time
from scipy.stats import sem
from argparse import ArgumentParser

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CEVAE(nn.Module):
    def __init__(self, n_bin, n_cont, d=20, nh=3, h=200, weight_decay=1e-4):
        super().__init__()
        dimx = n_bin + n_cont
        self.n_bin = n_bin
        self.n_cont = n_cont
        self.d = d

        # ---- Decoder / Generative model ----
        # p(x|z): shared trunk then binary + continuous heads
        self.px_z_shared = FCNet(d, (nh - 1) * [h], activation=nn.ELU, weight_decay=weight_decay)
        self.px_z_bin = FCNet(h, [h], out_heads=[(n_bin, None)], activation=nn.ELU, weight_decay=weight_decay)
        self.px_z_cont = FCNet(h, [h], out_heads=[(n_cont, None), (n_cont, nn.Softplus())],
                               activation=nn.ELU, weight_decay=weight_decay)

        # p(t|z)
        self.pt_z = FCNet(d, [h], out_heads=[(1, None)], activation=nn.ELU, weight_decay=weight_decay)

        # p(y|t,z): separate networks for t=0 and t=1
        self.py_t0z = FCNet(d, nh * [h], out_heads=[(1, None)], activation=nn.ELU, weight_decay=weight_decay)
        self.py_t1z = FCNet(d, nh * [h], out_heads=[(1, None)], activation=nn.ELU, weight_decay=weight_decay)

        # ---- Encoder / Inference model ----
        # q(t|x)
        self.qt_x = FCNet(dimx, [d], out_heads=[(1, None)], activation=nn.ELU, weight_decay=weight_decay)

        # q(y|x,t): shared trunk then t=0/t=1 heads
        self.qy_xt_shared = FCNet(dimx, (nh - 1) * [h], activation=nn.ELU, weight_decay=weight_decay)
        self.qy_xt0 = FCNet(h, [h], out_heads=[(1, None)], activation=nn.ELU, weight_decay=weight_decay)
        self.qy_xt1 = FCNet(h, [h], out_heads=[(1, None)], activation=nn.ELU, weight_decay=weight_decay)

        # q(z|x,t,y): input is [x, qy_mean], shared trunk then t=0/t=1 heads for mu and sigma
        self.qz_shared = FCNet(dimx + 1, (nh - 1) * [h], activation=nn.ELU, weight_decay=weight_decay)
        self.qz_t0 = FCNet(h, [h], out_heads=[(d, None), (d, nn.Softplus())],
                           activation=nn.ELU, weight_decay=weight_decay)
        self.qz_t1 = FCNet(h, [h], out_heads=[(d, None), (d, nn.Softplus())],
                           activation=nn.ELU, weight_decay=weight_decay)

    def _encode(self, x, t, y):
        """Run the inference network; returns distributions qt, qy, qz."""
        # q(t|x)
        logits_t = self.qt_x(x)
        qt = Bernoulli(logits=logits_t)

        # q(y|x,t) — use observed t to select the correct head
        hqy = self.qy_xt_shared(x)
        mu_qy_t0 = self.qy_xt0(hqy)
        mu_qy_t1 = self.qy_xt1(hqy)
        mu_qy = t * mu_qy_t1 + (1. - t) * mu_qy_t0
        qy = Normal(mu_qy, torch.ones_like(mu_qy))

        # q(z|x,t,y) — use observed y as input
        inpt = torch.cat([x, y], dim=1)
        hqz = self.qz_shared(inpt)
        muq_t0, sigmaq_t0 = self.qz_t0(hqz)
        muq_t1, sigmaq_t1 = self.qz_t1(hqz)
        mu_qz = t * muq_t1 + (1. - t) * muq_t0
        sigma_qz = t * sigmaq_t1 + (1. - t) * sigmaq_t0
        qz = Normal(mu_qz, sigma_qz)

        return qt, qy, qz

    def _decode(self, z, t):
        """Run the generative model conditioned on z and t; returns distribution parameters."""
        # p(x|z)
        hx = self.px_z_shared(z)
        x_bin_logits = self.px_z_bin(hx)
        x_cont_mu, x_cont_sigma = self.px_z_cont(hx)

        # p(t|z)
        t_logits = self.pt_z(z)

        # p(y|t,z)
        mu_y_t0 = self.py_t0z(z)
        mu_y_t1 = self.py_t1z(z)
        mu_y = t * mu_y_t1 + (1. - t) * mu_y_t0

        return x_bin_logits, x_cont_mu, x_cont_sigma, t_logits, mu_y

    def forward(self, x_bin, x_cont, t, y):
        """Compute the negative ELBO loss."""
        x = torch.cat([x_bin, x_cont], dim=1)

        # Encode
        qt, qy, qz = self._encode(x, t, y)

        # Sample z from approximate posterior
        z = qz.rsample()

        # Decode
        x_bin_logits, x_cont_mu, x_cont_sigma, t_logits, mu_y = self._decode(z, t)

        # Reconstruction log-likelihoods
        log_px_bin = Bernoulli(logits=x_bin_logits).log_prob(x_bin).sum(dim=1)
        log_px_cont = Normal(x_cont_mu, x_cont_sigma).log_prob(x_cont).sum(dim=1)
        log_pt = Bernoulli(logits=t_logits).log_prob(t).sum(dim=1)
        log_py = Normal(mu_y, torch.ones_like(mu_y)).log_prob(y).sum(dim=1)

        # Auxiliary log-likelihoods for the inference network
        log_qt = qt.log_prob(t).sum(dim=1)
        log_qy = qy.log_prob(y).sum(dim=1)

        # KL divergence: KL(q(z|x,t,y) || p(z))
        pz = Normal(torch.zeros_like(qz.loc), torch.ones_like(qz.scale))
        kl_z = kl_divergence(qz, pz).sum(dim=1)

        # ELBO = E_q[log p(x,t,y|z)] - KL(q(z)||p(z)) + log q(t|x) + log q(y|x,t)
        # The auxiliary terms encourage qt and qy to match observations
        elbo = log_px_bin + log_px_cont + log_pt + log_py - kl_z + log_qt + log_qy
        return -elbo.mean()

    def compute_logp_valid(self, x_bin, x_cont, t, y):
        """Deterministic approximation of log p(x,t,y) for validation early stopping."""
        x = torch.cat([x_bin, x_cont], dim=1)
        qt, qy, qz = self._encode(x, t, y)

        z_mean = qz.mean

        x_bin_logits, x_cont_mu, x_cont_sigma, t_logits, mu_y = self._decode(z_mean, t)

        log_px_bin = Bernoulli(logits=x_bin_logits).log_prob(x_bin).sum(dim=1)
        log_px_cont = Normal(x_cont_mu, x_cont_sigma).log_prob(x_cont).sum(dim=1)
        log_pt = Bernoulli(logits=t_logits).log_prob(t).sum(dim=1)
        log_py = Normal(mu_y, torch.ones_like(mu_y)).log_prob(y).sum(dim=1)

        pz = Normal(torch.zeros_like(qz.loc), torch.ones_like(qz.scale))
        log_pz = pz.log_prob(z_mean).sum(dim=1)
        log_qz = qz.log_prob(z_mean).sum(dim=1)

        logp = log_px_bin + log_px_cont + log_pt + log_py + log_pz - log_qz
        return logp.mean().item()

    def predict_y0_y1(self, x_bin, x_cont, y_obs):
        """Predict E[y|x,t=0] and E[y|x,t=1] using the posterior predictive."""
        x = torch.cat([x_bin, x_cont], dim=1)
        n = x.shape[0]

        t0 = torch.zeros(n, 1, device=x.device)
        t1 = torch.ones(n, 1, device=x.device)

        # q(z|x,t=0,y) and q(z|x,t=1,y) — use observed y
        _, _, qz0 = self._encode(x, t0, y_obs)
        _, _, qz1 = self._encode(x, t1, y_obs)

        z0 = qz0.rsample()
        z1 = qz1.rsample()

        # p(y|t=0,z0) and p(y|t=1,z1)
        _, _, _, _, mu_y0 = self._decode(z0, t0)
        _, _, _, _, mu_y1 = self._decode(z1, t1)

        return mu_y0, mu_y1


def l2_penalty(model, weight_decay):
    """Compute L2 regularization penalty over all weight parameters."""
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
        model = CEVAE(n_bin, n_cont, d=d, nh=nh, h=h, weight_decay=lamba).to(device)

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

        # Pre-convert evaluation data to tensors
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

                loss = model(x_batch_bin, x_batch_cont, t_batch, y_batch)
                loss = loss + l2_penalty(model, lamba)

                optimizer.zero_grad()
                loss.backward()
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
                    f"Epoch: {epoch + 1}/{n_epoch}, log p(x) >= {avg_loss:.3f}, "
                    f"ite_tr: {score_train[0]:.3f}, ate_tr: {score_train[1]:.3f}, pehe_tr: {score_train[2]:.3f}, "
                    f"rmse_f_tr: {rmses_train[0]:.3f}, rmse_cf_tr: {rmses_train[1]:.3f}, "
                    f"ite_te: {score_test[0]:.3f}, ate_te: {score_test[1]:.3f}, pehe_te: {score_test[2]:.3f}, "
                    f"dt: {time.time() - t0:.3f}"
                )

        # Restore best model and do final evaluation
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

    print('\nCEVAE model total scores')
    means, stds = np.mean(scores, axis=0), sem(scores, axis=0)
    print(f'train ITE: {means[0]:.3f}+-{stds[0]:.3f}, train ATE: {means[1]:.3f}+-{stds[1]:.3f}, '
          f'train PEHE: {means[2]:.3f}+-{stds[2]:.3f}')

    means, stds = np.mean(scores_test, axis=0), sem(scores_test, axis=0)
    print(f'test ITE: {means[0]:.3f}+-{stds[0]:.3f}, test ATE: {means[1]:.3f}+-{stds[1]:.3f}, '
          f'test PEHE: {means[2]:.3f}+-{stds[2]:.3f}')


if __name__ == '__main__':
    main()
