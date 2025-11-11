import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class OrdinalReweightingLoss:
    def __init__(self, args, per_cls_weights):
        self.args = args

        self.ordinal_regression_loss = OrdinalRegressionLoss(args)
        self.ordinal_ib_focal_loss = OrdinalIBFocalLoss(weight=per_cls_weights, alpha=args.alpha, gamma=args.gamma)

    def __call__(self, ord_p, encoded_gt, z, ib_mode=False):
        if ib_mode:
            return self.ordinal_ib_focal_loss(ord_p, encoded_gt, z)
        else:
            return self.ordinal_regression_loss(ord_p, encoded_gt)


class OrdinalRegressionLoss:
    def __init__(self, args):
        self.args = args
        # self.bce_loss = nn.BCELoss(weight=None)
        self.bce_loss = nn.CrossEntropyLoss()

    def __call__(self, ord_p, encoded_gt):
        ord_log_nk, ord_log_pk = torch.chunk(ord_p, 2, -1)
        ord_log_pk = ord_log_pk.squeeze(-1)

        valid_mask = torch.arange(ord_log_pk.size(1), device=encoded_gt.device).unsqueeze(0) <= (encoded_gt)

        ord_p = rearrange(ord_p, "b k p -> (b k) p")
        valid_mask = rearrange(valid_mask, "b k -> (b k)")

        return self.bce_loss(ord_p, valid_mask.long())


class OrdinalIBFocalLoss(nn.Module):
    def __init__(self, weight=None, alpha=10000.0, gamma=0.0):
        super().__init__()

        assert alpha > 0

        self.alpha = alpha
        self.epsilon = 0.001
        self.weight = weight
        self.gamma = gamma

    def ib_focal_loss(self, input_values, ib, gamma):
        p = torch.exp(-input_values)
        loss = (1 - p) ** gamma * input_values * ib.reshape(-1, 1)
        return loss.mean()

    def forward(self, ord_p, encoded_gt, features):
        features = torch.sum(torch.abs(features), 1).reshape(-1, 1)

        ord_log_nk, ord_log_pk = torch.chunk(ord_p.softmax(-1), 2, -1)
        ord_log_pk = ord_log_pk.squeeze(-1)

        valid_mask = torch.arange(ord_log_pk.size(1), device=encoded_gt.device).unsqueeze(0) <= (encoded_gt)

        grads = torch.sum(torch.abs(ord_log_pk - valid_mask.float()), 1)  # N * 1
        ib = grads * (features.reshape(-1))
        ib = self.alpha / (ib + self.epsilon)

        return self.ib_focal_loss(
            F.binary_cross_entropy(ord_log_pk, valid_mask.float(), reduction="none", weight=self.weight),
            ib,
            self.gamma,
        )
