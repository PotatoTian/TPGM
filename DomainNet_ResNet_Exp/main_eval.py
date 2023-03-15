import argparse
import copy
import torch
from datasets import get_loader
from models import get_model
import os
from datetime import datetime
from torch.utils import data
from utils import (
    accuracy,
    AverageMeter,
)
from utils.clip_utils import clip_config




def eval(args):

    # Setup device
    device = torch.device("cuda")

    n_classes = args.n_classes

    # Setup Model
    model_cfg = {"arch": args.arch}

    # Setup Model and Load pretrain
    if args.load_pretrained is not None:
        if os.path.isfile(args.load_pretrained):
            info = "Loading model and optimizer from checkpoint '{}'".format(
                args.load_pretrained
            )

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
            print(info)
        else:
            info = "No pretrained model found at '{}'".format(args.load_pretrained)
            print(info)
            model = get_model(**model_cfg, num_classes=n_classes).to(device)
    else:
        info = "Use random initialization"
        print(info)
        model = get_model(**model_cfg, num_classes=n_classes).to(device)

   
    
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    with open(args.resume, "rb") as fp:
        checkpoint = torch.load(fp)
    checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint, strict=True)

    ##================== Testing ============================
    print("start testing")
    sites = ["real", "sketch", "painting", "infograph", "clipart"]
    datasets = [
        get_loader("test", name=args.dataset, root=args.root, data_dir=args.data_dir, site=site)
        for site in sites
    ]
    loaders = [
        data.DataLoader(
            dataset,
            batch_size=args.batch_size * args.gpu_per_node,
            num_workers=args.n_workers,
        )
        for dataset in datasets
    ]

    model.eval()
    with torch.no_grad():
        for site, loader in zip(sites, loaders):
            test_top1 = AverageMeter("Acc@1", ":6.2f")
            test_top5 = AverageMeter("Acc@5", ":6.2f")
            for i, (image, target) in enumerate(loader):
                image = image.to(device)
                target = target.to(device)
                logit = model(image)

                acc1, acc5 = accuracy(logit, target, topk=(1, 5))
                test_top1.update(acc1[0], image.size(0))
                test_top5.update(acc5[0], image.size(0))
                if i % 100 == 0:
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
        "--load_pretrained",
        type=str,
        default=None,
        help="Pretrained model direcotry",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resuming checkpoing",
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
    args = parser.parse_args()
    eval(args)