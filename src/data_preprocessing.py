"""
DataLoader for Streaming Image Training
enhancement
"""

import torch
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import IterableDataset


def collate_fn(batch, img_size: int = 224):
    """Preprocess the images by resizing and trasforming into Tensor

    :param batch: A training batch consisting of n images
    :param img_size: Input image size
    :type img_size: int
    """
    transform = transforms.Compose(
        [transforms.Resize((img_size, img_size)), transforms.ToTensor()]
    )

    images = torch.stack([transform(x["image"]) for x in batch])
    labels = [x["label"] for x in batch]

    return images, labels


class DataStream(IterableDataset):
    def __init__(self, path: str, split: str, streaming: bool = True, transform=None):
        super().__init__()
        self.dataset = load_dataset(path=path, split=split, streaming=streaming)
        self.transform = transform

    def __iter__(self):
        for item in self.dataset:
            if self.transform:
                yield self.transform(item)
            else:
                yield item
