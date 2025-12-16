import torch
import torch.nn as nn


class RoPE(nn.Module):
    def __init__(self, d_head: int, base: int = 10000):
        super().__init__()

        self.base = base
        self.d_head = d_head
        self.blocks = d_head // 4
        self.sin_cache = None
        self.cos_cache = None

    def compute_rotation(self):
        if self.sin_cache is not None:
            return

        # position(t) = [0,1,2...number of blocks]

        positions = torch.arange(self.blocks)
        # THETA = 100^t-/(d_head//4) or 1/10,000^(2i/d)
        # t: block index

        theta = 1.0 / (100 ** (positions / self.blocks)).float()

        angles = (positions * theta).float()

        self.sin_cache = torch.sin(angles).float()
        self.cos_cache = torch.cos(angles).float()

    def forward(self, x):
        """
        Docstring for forward

        :param self: Description
        :param x: A tensor of shape (Batch, number of patches, embedding_size)
        """
        # split W: (x1,x2) , H: (y1,y2)
        self.compute_rotation()

        x1, x2, y1, y2 = x.split(self.blocks, dim=-1)

        # Rotation
        rx1 = x1 * self.cos_cache - x2 * self.sin_cache
        rx2 = x1 * self.sin_cache + x2 * self.cos_cache
        ry1 = y1 * self.cos_cache - y2 * self.sin_cache
        ry2 = y1 * self.sin_cache + y2 * self.cos_cache

        # Rotated q
        return torch.cat([rx1, rx2, ry1, ry2], dim=-1)
