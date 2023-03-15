from datasets.domainnet_loader import DomainNetDataset
from torchvision import transforms


def get_loader(phase, **kwargs):
    meta_dir = kwargs["meta_dir"]
    data_dir = kwargs["data_dir"]
    dataset = kwargs["name"]
    if dataset == "domainnet":

        # mean and std for the Real domain
        mean = [0.5964, 0.5712, 0.5336]
        std = [0.3228, 0.3165, 0.3374]
        
        normalize = transforms.Normalize(mean=mean, std=std)
        if phase == "train":
            return DomainNetDataset(
                data_dir=data_dir,
                meta_dir=meta_dir,
                sites=[kwargs["site"]],
                train="train",
                percent=kwargs["percent"],
                transform=transforms.Compose(
                    [
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize,
                    ]
                ),
            )
        elif phase == "val":
            return DomainNetDataset(
                data_dir=data_dir,
                meta_dir=meta_dir,
                sites=[kwargs["site"]],
                train="val",
                percent=kwargs["percent"],
                transform=transforms.Compose(
                    [
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ]
                ),
            )
        else:
            return DomainNetDataset(
                    data_dir=data_dir,
                    meta_dir=meta_dir,
                    sites=[kwargs["site"]],
                    train="test",
                    transform=transforms.Compose(
                        [
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize,
                        ]
                    ),
                )
        