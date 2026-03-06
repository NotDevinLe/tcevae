import torch
import torch.nn as nn
import numpy as np
import sys


class FCNet(nn.Module):
    """Fully connected network with optional multiple output heads."""

    def __init__(self, in_dim, hidden_layers, out_heads=None, activation=nn.ELU,
                 weight_decay=1e-4):
        super().__init__()
        self.shared = nn.Sequential()
        prev = in_dim
        for i, h in enumerate(hidden_layers):
            layer = nn.Linear(prev, h)
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            self.shared.add_module(f'fc_{i}', layer)
            self.shared.add_module(f'act_{i}', activation())
            prev = h

        self.heads = nn.ModuleList()
        self.head_activations = []
        if out_heads:
            for j, (out_dim, act_fn) in enumerate(out_heads):
                head = nn.Linear(prev, out_dim)
                nn.init.xavier_normal_(head.weight)
                nn.init.zeros_(head.bias)
                self.heads.append(head)
                self.head_activations.append(act_fn)

    def forward(self, x):
        h = self.shared(x)
        if not self.heads:
            return h
        outputs = []
        for head, act in zip(self.heads, self.head_activations):
            o = head(h)
            if act is not None:
                o = act(o)
            outputs.append(o)
        return outputs if len(outputs) > 1 else outputs[0]


@torch.no_grad()
def get_y0_y1(model, x_bin, x_cont, y_obs, L=1, verbose=True):
    """Compute E[y|x,t=0] and E[y|x,t=1] via posterior predictive sampling."""
    model.eval()
    n = x_bin.shape[0]
    y0 = torch.zeros(n, 1, device=x_bin.device)
    y1 = torch.zeros(n, 1, device=x_bin.device)
    for l in range(L):
        if L > 1 and verbose:
            sys.stdout.write(f'\r Sample {l + 1}/{L}')
            sys.stdout.flush()
        y0_l, y1_l = model.predict_y0_y1(x_bin, x_cont, y_obs)
        y0 += y0_l / L
        y1 += y1_l / L
    if L > 1 and verbose:
        print()
    model.train()
    return y0.cpu().numpy(), y1.cpu().numpy()
