import json
import os

from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

import PIL


class ImageNetManifoldDataset(Dataset):
    def __init__(self, root, meta_root, mode, transform=None):
        self.root = root
        self.meta_root = meta_root
        assert mode in ("train", "val", "test")
        self.mode = mode
        self.transform = transform
        self._load_imdb()

    def _load_imdb(self):
        split_path = os.path.join(self.meta_root, f"{self.mode}.json")
        with open(split_path, "r") as f:
            data = f.read()
        self._imdb = json.loads(data)
        for idx in range(len(self._imdb)):
            im_path = self._imdb[idx]["img_path"]
            im_path = os.path.join(self.root,im_path)
            self._imdb[idx]["img_path"] = im_path

    def __getitem__(self, idx):
        im_path = self._imdb[idx]["img_path"]
        cls_idx = self._imdb[idx]["class"]
        with open(im_path, "rb") as f:
            im = Image.open(f)
            im = im.convert("RGB")
        if self.transform is not None:
            im = self.transform(im)
        return im, cls_idx

    def __len__(self):
        return len(self._imdb)


class ImageNetOODManifoldDataset(Dataset):
    def __init__(self, root, meta_root, split, transform=None):
        assert split in ("imagenet-a", "imagenet-r", "imagenet-s", "imagenet-2")
        self.root = root
        self.meta_root = meta_root
        self.split = split
        self.transform = transform
        self._load_imdb()

    def _load_imdb(self):
        split_path = os.path.join(self.meta_root, self.split + ".json")
        with open(split_path, "r") as f:
            data = f.read()
        self._imdb = json.loads(data)
        for idx in range(len(self._imdb)):
            im_path = self._imdb[idx]["img_path"]
            im_path = os.path.join(self.root,im_path)
            self._imdb[idx]["img_path"] = im_path

    def __getitem__(self, idx):
        im_path = self._imdb[idx]["img_path"]
        cls_idx = self._imdb[idx]["class"]
        with open(im_path, "rb") as f:
            im = Image.open(f)
            im = im.convert("RGB")
            im = self.transform(im)
            
            
        return im, cls_idx

    def __len__(self):
        return len(self._imdb)
