import torch
from timm.scheduler.cosine_lr import CosineLRScheduler

from .hbnet import HbNet


def create_optimizer(args, model, n_iter_per_epoch):
    if args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=float(args.weight_decay))
    elif args.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=float(args.weight_decay))
    elif args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            weight_decay=float(args.weight_decay),
            nesterov=True,
            momentum=0.9,
        )
    else:
        raise NotImplementedError()

    if args.scheduler.lower() == "lrp":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)
    elif args.scheduler.lower() == "cosine":
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=args.epochs * n_iter_per_epoch,
            lr_min=float(args.min_lr),
            warmup_lr_init=float(args.warm_lr),
            warmup_t=args.warm_epochs * n_iter_per_epoch,
            cycle_limit=1,
            t_in_epochs=False,
        )
    else:
        scheduler = None

    return optimizer, scheduler
