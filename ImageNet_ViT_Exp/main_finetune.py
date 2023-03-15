import argparse
import copy
import logging
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

from datasets import build_dataset, build_ood_dataset
from engine_finetune import evaluate

from models.clip_models import VisionTransformer as clip_transformer
from utils.misc import dump_logs,log_constraints
from utils.tpgm import tpgm_trainer
from utils.wise import WISE

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def get_args_parser():
    parser = argparse.ArgumentParser(
        "MAE fine-tuning for image classification", add_help=False
    )

    # Model parameters
    parser.add_argument("--input_size", default=224, type=int, help="images input size")
    parser.add_argument(
        "--output_dir",
        default="./output_dir",
        help="path where to save, empty for no saving",
    )
    # Finetuning params
    parser.add_argument("--load_ft", default=None, help="load fine-tuned checkpoint")
    parser.add_argument("--load_head", default=None, help="load CLIP zero-shot head")
    parser.add_argument("--load_pretrained", default="", help="finetune from checkpoint")
    
    # Dataset parameters
    parser.add_argument(
        "--data_path",
        default="/datasets/ImageNet",
        type=str,
        help="dataset path",
    )
    parser.add_argument(
        "--meta_path",
        default="./datasets/imagenet",
        type=str,
        help="dataset meta info path",
    )
    parser.add_argument(
        "--nb_classes",
        default=1000,
        type=int,
        help="number of the classification types",
    )
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size",
    )

    # Projection hyperparameters
    parser.add_argument("--proj_freq", default=0, type=int)
    parser.add_argument("--max_iters", default=200, type=int)
    parser.add_argument(
        "--proj_lr",
        default=1e-2,
        type=float,
        help="Custom Project Learning Rate",
    )
    parser.add_argument("--norm_mode", default="l2", type=str)
    parser.add_argument(
        "--mu",
        default=0.0,
        type=float,
        help="Hyperparameter for WISE/mu",
    )

    return parser




def main_worker( args):

    device = torch.device("cuda")
    cudnn.benchmark = True

    imagenet_val = build_dataset(phase="val", args=args)
    imagenet_val_loader = torch.utils.data.DataLoader(
        imagenet_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
    )


    ood_splits =  ["imagenet-2", "imagenet-a", "imagenet-r", "imagenet-s"]
    ood_datasets = [build_ood_dataset(split, args=args) for split in ood_splits]
    
    ood_data_loaders = [
        torch.utils.data.DataLoader(
            ood_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=False,
        )
        for ood_dataset in ood_datasets
    ]

    all_datasets = ["imagenet"] + ood_splits
    all_loaders = [imagenet_val_loader] + ood_data_loaders


    # Setup TPGM loader
    pgmloader = torch.utils.data.DataLoader(
        imagenet_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=False,
    )
    
    
    # Load CLIP visual enocder
    with open(args.load_pretrained, "rb") as fp:
        checkpiont = torch.jit.load(fp, map_location=device)
    checkpoint_model = checkpiont.state_dict()
    vision_width = checkpoint_model["visual.conv1.weight"].shape[0]
    vision_layers = len(
        [
            k
            for k in checkpoint_model.keys()
            if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")
        ]
    )
    vision_patch_size = checkpoint_model["visual.conv1.weight"].shape[-1]
    embed_dim = checkpoint_model["text_projection"].shape[1]
    grid_size = round(
        (checkpoint_model["visual.positional_embedding"].shape[0] - 1) ** 0.5
    )
    image_resolution = vision_patch_size * grid_size
    vision_heads = vision_width // 64
    model = clip_transformer(
        input_resolution=image_resolution,
        patch_size=vision_patch_size,
        width=vision_width,
        layers=vision_layers,
        heads=vision_heads,
        output_dim=embed_dim,
        n_class=args.nb_classes,
    )
    checkpoint_model = {
        k.replace("visual.", ""): v
        for k, v in checkpoint_model.items()
        if k.startswith("visual.")
    }
    model_dict = model.state_dict()
    filtered_checkpoint = {
        k: v
        for k, v in checkpoint_model.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }
    model.load_state_dict(filtered_checkpoint, strict=False)
    info = "Loaded pre-trained checkpoint '{}' and {}/{} layers".format(
        args.load_pretrained, len(filtered_checkpoint), len(model_dict)
    )
    print(info)

    # Load CLIP zero-shot head
    if args.load_head is not None:
        with open(args.load_head, "rb") as fp:
            checkpoint_head = torch.load(fp, map_location=device)
        with torch.no_grad():
            model.head.weight.copy_(checkpoint_head)
        print("Loaded zero-shot head from '{}'".format(args.load_head))
    anchor = copy.deepcopy(model)  # Keep a copy of the pre-trained model as anchor
    anchor.to(device)
    anchor = torch.nn.DataParallel(anchor, device_ids=range(torch.cuda.device_count()))

    # Load fine-tuned model
    if args.load_ft is not None:
        with open(args.load_ft, "rb") as fp:
            checkpiont = torch.load(fp, map_location=device)
        checkpoint_model = checkpiont["model"]
        model_dict = model.state_dict()
        filtered_checkpoint = {
            k: v
            for k, v in checkpoint_model.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        model.load_state_dict(filtered_checkpoint, strict=False)
        info = "Loaded fine-tuned checkpoint '{}' and {}/{} layers".format(
            args.load_ft, len(filtered_checkpoint), len(model_dict)
        )
        print(info)

    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
   
        
    if bool(args.proj_freq):
        print("Training TPGM projection parameters")
        tpgm = tpgm_trainer(
            anchor,
            pgmloader,
            args.norm_mode,
            args.proj_lr,
            args.max_iters,
            exclude_list = []
        )
        tpgm.tpgm_iters(model)
        log_constraints(args.output_dir,tpgm.tpgm, init=True)
        log_constraints(args.output_dir,tpgm.tpgm, save=True)
    else:
        model = WISE(model, anchor, args.mu)

    for i, loader in enumerate(all_loaders):
        print("Testing {}".format(all_datasets[i]))
        test_stats = evaluate(loader, all_datasets[i], model, device)
        log_stats = {
            **{f"{all_datasets[i]}_test_{k}": v for k, v in test_stats.items()},
        }
        print(log_stats)
        dump_logs(args.output_dir, log_stats)


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main_worker(args)
