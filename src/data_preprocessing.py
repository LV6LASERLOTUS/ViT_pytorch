"""
DataLoader for Streaming Image Training
enhancement
"""

import torch
from torchvision import transforms


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
