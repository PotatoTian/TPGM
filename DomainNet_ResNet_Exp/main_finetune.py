import argparse
import copy
import torch
from torch.utils import data
from datasets import get_loader
from engine_finetune import train_one_epoch
from models import get_model
import os
from datetime import datetime
from utils import (
    accuracy,
    AverageMeter,
    dump_logs,
)
from utils.clip_utils import clip_config
from utils.tpgm import tpgm_trainer


def train(logdir, args):

    dump_logs(logdir, "Let the games begin")
    device = torch.device("cuda")

    # Setup dataloader
    t_loader = get_loader(
        "train", name=args.dataset, root=args.root, site=args.site, data_dir=args.data_dir, percent=args.percent
    )
    v_loader = get_loader(
        "val", name=args.dataset, root=args.root, site=args.site, data_dir=args.data_dir, percent=args.percent
    )
    pgm_loader = get_loader(
        "val", name=args.dataset, root=args.root, site=args.site, data_dir=args.data_dir, percent=args.percent
    )

    trainloader = data.DataLoader(
        t_loader,
        shuffle=True,
        batch_size=args.batch_size*args.gpu_per_node,
        num_workers=args.n_workers,
    )

    valloader = data.DataLoader(
        v_loader,
        batch_size=args.batch_size*args.gpu_per_node,
        num_workers=args.n_workers,
    )        

    n_classes = args.n_classes

    # Setup model
    model_cfg = {"arch": args.arch}

    # Setup Model and load pre-train
    if args.load_pretrained is not None:
        if os.path.isfile(args.load_pretrained):
            info = "Loading model and optimizer from checkpoint '{}'".format(
                args.load_pretrained
            )
            dump_logs(logdir, info + "\n")

            with open(args.load_pretrained, "rb") as fp:
                checkpoint = torch.load(fp)

            if "clip" in args.load_pretrained:
                checkpoint = checkpoint.state_dict()
                clip_config(model_cfg, checkpoint, pretrained=True)
                checkpoint = {
                    k.replace("visual.", ""): v
                    for k, v in checkpoint.items()
                    if "transformer" not in k
                }

            elif "moco" in args.load_pretrained:
                checkpoint = checkpoint["state_dict"]
                checkpoint = {
                    k.replace("base_encoder.", "").replace("module.", ""): v
                    for k, v in checkpoint.items()
                }

            model = get_model(**model_cfg, num_classes=n_classes).to(device)

            model_dict = model.state_dict()
            filtered_checkpoint = {
                k: v
                for k, v in checkpoint.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }

            model.load_state_dict(filtered_checkpoint, strict=False)
            info = "Loaded pretrained model '{}' and {}/{} layers".format(
                args.load_pretrained, len(filtered_checkpoint), len(model_dict)
            )
            dump_logs(logdir, info + "\n")
            print(info)
        else:
            info = "No pretrained model found at '{}'".format(args.load_pretrained)
            print(info)
            dump_logs(logdir, info + "\n")
            model = get_model(**model_cfg, num_classes=n_classes).to(device)
    else:
        info = "Use random initialization"
        dump_logs(logdir, info + "\n")
        print(info)
        model = get_model(**model_cfg, num_classes=n_classes).to(device)


    
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    # Setup automatic PGM parameters, optimizer, and scheduler
    if bool(args.proj_freq):
        sampler_tpgm = torch.utils.data.RandomSampler(pgm_loader)
        pgmloader = torch.utils.data.DataLoader(
            v_loader,
            sampler=sampler_tpgm,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=False,
            persistent_workers=True,
        )
        tpgm = tpgm_trainer(
            model,
            pgmloader,
            args.norm_mode,
            args.proj_lr,
            args.max_iters,
            exclude_list=["head.weight","head.bias"]
        )
    else:
        tpgm = None

    # Setup optimizer, lr_scheduler and loss function
    optimizer_params = {
        "lr": args.lr,
        "weight_decay": 5.0e-4,
        "momentum": 0.9,
        "nesterov": True,
    }
    optimizer = torch.optim.SGD(model.parameters(), **optimizer_params)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)

    start_epoch = 0
    best_acc1 = -100.0
    best_model = model

    # ================================ Training ==========================================
    for epoch in range(start_epoch, args.epoch):
        best_acc1, best_model = train_one_epoch(
            args,
            model,
            loss_fn,
            optimizer,
            scheduler,
            tpgm,
            trainloader,
            valloader,
            device,
            logdir,
            epoch,
            best_acc1,
            best_model,
        )
    # ================================ Testing ==========================================
    print("start testing")
    sites = ["real", "sketch", "painting", "infograph", "clipart"]
    datasets = [
        get_loader("test", name=args.dataset, root=args.root, data_dir=args.data_dir, site=site)
        for site in sites
    ]
    loaders = [
        data.DataLoader(
            dataset,
            batch_size=args.batch_size * args.gpu_per_node* 2,
            num_workers=args.n_workers,
        )
        for dataset in datasets
    ]

    best_model.eval()
    with torch.no_grad():
        for site, loader in zip(sites, loaders):
            test_top1 = AverageMeter("Acc@1", ":6.2f")
            test_top5 = AverageMeter("Acc@5", ":6.2f")
            for i, (image, target) in enumerate(loader):
                image = image.to(device)
                target = target.to(device)
                logit = best_model(image)

                acc1, acc5 = accuracy(logit, target, topk=(1, 5))
                test_top1.update(acc1[0], image.size(0))
                test_top5.update(acc5[0], image.size(0))
                if i % args.print_interval == 0:
                    output = "{} test: [{}/{}]".format(
                        site,
                        i,
                        len(loader),
                    )
                    print(output)

            output = "{site} test results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}\t".format(
                site=site,
                top1=test_top1,
                top5=test_top5,
            )

            print(output)
            dump_logs(logdir, output + "\n")

def main(args):
    main_worker(0,args)


def main_worker(local_rank, args):
    args.local_rank = local_rank
    now = datetime.now()
    logdir = "./log/{}_{}".format(args.id,now.strftime("%d_%m_%Y_%H_%M_%S"))
    args.output_dir = logdir
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    
    print("RUNDIR: {}".format(logdir))
    train(logdir, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--arch",
        nargs="?",
        type=str,
        default="clip_resnet50",
        help="Backbone Architecture",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Specify save directory",
    )
    parser.add_argument(
        "--load_pretrained",
        type=str,
        default=None,
        help="pretrained model direcotry",
    )
    parser.add_argument(
        "--id",
        nargs="?",
        type=str,
        default=None,
        help="Additional run information",
    )
    parser.add_argument(
        "--epoch",
        default=200,
        type=int,
        help="training epoch",
    )

    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * gpus",
    )
    parser.add_argument(
        "--gpu_per_node", default=1, type=int, help="Number of gpus per node"
    )
    parser.add_argument(
        "--print_interval",
        default=100,
        type=int,
        help="print interval",
    )
    parser.add_argument(
        "--val_freq",
        default=1,
        type=int,
        help="Validation interval",
    )
    parser.add_argument(
        "--n_workers",
        default=4,
        type=int,
        help="number of workers",
    )
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )

    # dataset parameters
    parser.add_argument(
        "--dataset",
        nargs="?",
        type=str,
        default="domainnet",
        help="Dataset name",
    )
    parser.add_argument(
        "--root",
        nargs="?",
        type=str,
        default= "./datasets/",
        help="Data meta data root",
    )
    parser.add_argument(
        "--data_dir",
        nargs="?",
        type=str,
        default="/datasets/domainnet",
        help="Image data directory",
    )
    parser.add_argument(
        "--site",
        nargs="?",
        type=str,
        default="real",
        help="DomainNet site",
    )
    parser.add_argument(
        "--percent",
        nargs="?",
        type=str,
        default="5",
        help="DomainNet percentage",
    )
    parser.add_argument(
        "--n_classes",
        nargs="?",
        type=int,
        default=345,
        help="Number of classes",
    )

    # optimizer
    parser.add_argument(
        "--optimizer",
        nargs="?",
        type=str,
        default="sgd",
        help="optimizer type",
    )
    parser.add_argument(
        "--lr",
        default=None,
        type=float,
        help="Custom Learning Rate",
    )

    parser.add_argument(
        "--load_ft",
        nargs="?",
        type=str,
        default=None,
        help="load fine-tuned checkpoint",
    )

    # projection hyperparameters
    parser.add_argument("--proj_freq", default=0, type=int)
    parser.add_argument("--max_iters", default=1, type=int)
    parser.add_argument(
        "--proj_lr",
        default=1e-2,
        type=float,
        help="Custom Project Learning Rate",
    )
    parser.add_argument("--norm_mode", default="mars", type=str)
    parser.add_argument(
        "--mu", default=0.0, type=float, help="Hyperparamter for TV smoothing"
    )

    args = parser.parse_args()
    main(args)
