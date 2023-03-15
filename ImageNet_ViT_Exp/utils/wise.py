import copy

import torch
import torch.nn as nn


class WISE(nn.Module):
    def __init__(self, new, anchor, mu) -> None:
        super().__init__()
        self.mu = mu
        self.new = new
        self.apply_constraints(anchor)

    def apply_constraints(
        self,
        anchor,
    ):
        for (_, new_para), anchor_para in zip(
            self.new.named_parameters(), anchor.parameters()
        ):
            temp = self.mu * new_para + (1 - self.mu) * anchor_para.detach()
            with torch.no_grad():
                new_para.copy_(temp)

    def forward(self, x):
        out = self.new(x)
        return out
