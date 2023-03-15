import os
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

class DomainNetDataset(Dataset):
    def __init__(
        self, data_dir, meta_dir, sites, train="train", percent=100, transform=None
    ):
        self.transform = transform
        self.train = train
        base_dir = os.path.join(meta_dir, "domainnet")
        self.labels = []
        self.img_path = []
        for site in sites:
            if self.train == "train":
                label_file = os.path.join(
                    base_dir, "{}_train_{}_percent_split.txt".format(site, str(percent))
                )
            elif self.train == "val":
                label_file = os.path.join(
                    base_dir, "{}_val_{}_percent_split.txt".format(site, str(percent))
                )
            else:
                label_file = os.path.join(base_dir, "{}_test.txt".format(site))
            with open(label_file, "r") as fp:
                file_list = fp.readlines()
                for line in file_list:
                    temp = line.split(" ")
                    full_path = os.path.join(data_dir, temp[0])
                    label = int(temp[1].split("\n'")[0])
                    self.img_path.append(full_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.img_path[idx]
        label = self.labels[idx]
        with open(img_path, "rb") as fp:
            image = Image.open(fp)
            if len(image.split()) != 3:
                image = transforms.Grayscale(num_output_channels=3)(image)

            if self.transform is not None:
                image = self.transform(image)

        return image, label


