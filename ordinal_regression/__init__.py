import numpy as np
import torch

from .loss import OrdinalReweightingLoss


def create_ordinal_lookup(args):
    minima = args.Hb_minima + args.disc_shift
    maxima = args.Hb_maxima + args.disc_shift
    bins = args.ordinal_bins

    lookup_vals = np.exp(np.linspace(np.log(minima), np.log(maxima), bins))
    lookup_table = {k: float(v) for k, v in enumerate(lookup_vals)}
    return lookup_table


def encode_ordinal(args, decoded_labels: torch.Tensor):
    decoded_labels = decoded_labels + args.disc_shift

    encoded_labels = torch.zeros_like(decoded_labels)

    for k in range(args.ordinal_bins - 1):
        mask = (decoded_labels >= args.lookup[k]) & (decoded_labels < args.lookup[k + 1])
        encoded_labels[mask] = k

    mask = decoded_labels >= args.Hb_maxima + args.disc_shift
    encoded_labels[mask] = args.ordinal_bins - 1

    return encoded_labels


def decode_ordinal(args, pred: torch.Tensor):
    pred = pred.cpu().detach()
    B, K, _ = pred.shape
    assert K == args.ordinal_bins - 1, "Mismatch in number of bins"

    p_geq = torch.softmax(pred, -1)[..., -1]

    p_class = []
    for k in range(K):
        if k < K - 1:
            p_k = p_geq[:, k] - p_geq[:, k + 1]
        else:
            p_k = p_geq[:, k]
        p_class.append(p_k)

    p_class = torch.stack(p_class, dim=1)

    midpoints = torch.tensor(
        [(args.lookup[k] + args.lookup[k + 1]) / 2.0 for k in range(K)],
        device=pred.device,
        dtype=pred.dtype,
    ).unsqueeze(0)

    expected_hb = torch.sum(p_class * midpoints, dim=1, keepdim=True)
    return expected_hb
