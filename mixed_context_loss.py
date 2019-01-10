import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class MixedContextLoss(nn.Module):
    def __init__(self, theta_glo=1.15, delta=5, gamma=0.5, scale_aware=True):
        super(MixedContextLoss, self).__init__()
        self.theta_glo = theta_glo
        self.delta = delta
        self.gamma = gamma
        self.scale_aware = scale_aware

    def forward(self, y_a, y_p, targets):
        y_n = []
        for i in range(len(y_a)):
            cand_yn = y_p[targets!=targets[i]]
            if self.scale_aware:
                d_n_ik = F.pairwise_distance(y_a[i].unsqueeze(dim=0).repeat(len(cand_yn), 1), cand_yn)
                y_n.append(cand_yn[d_n_ik.argmin()].unsqueeze(dim=0))
            else:
                y_n.append(cand_yn[random.randrange(len(cand_yn))].unsqueeze(dim=0))
        y_n = torch.cat(y_n, dim=0)

        d_p = F.pairwise_distance(y_a, y_p)
        d_n = F.pairwise_distance(y_a, y_n)

        theta = self.gamma*(d_p + d_n)/2 + (1 - self.gamma)*self.theta_glo
        loss = -(1/(2*self.delta)*F.logsigmoid(2*self.delta*(theta - d_p))
                 + 1/(2*self.delta)*F.logsigmoid(2*self.delta*(d_n - theta)))
        loss = torch.mean(loss)

        return loss
