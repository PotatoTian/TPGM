import PIL
from datasets import imagenet_dataset
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms


def build_ood_dataset(split, args):
    transform = build_transform(args, False)
    return imagenet_dataset.ImageNetOODManifoldDataset(
        split=split,
        root=args.data_path,
        meta_root=args.meta_path,
        transform=transform,
    )


def build_dataset(phase, args):
    transform = build_transform(args,is_train=(phase=="train"))
    return imagenet_dataset.ImageNetManifoldDataset(
        root=args.data_path,
        meta_root=args.meta_path,
        mode=phase,
        transform=transform,
    )


def build_transform(args,is_train):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation="bicubic",
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(
            size, interpolation=PIL.Image.BICUBIC
        ),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
