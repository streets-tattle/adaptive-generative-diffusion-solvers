import pathlib
from typing import Callable
import pickle
import importlib.util

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.io import decode_image

from pi_solvers import dnnlib
from pi_solvers.torch_utils.dataset import ImageFolderDataset


# From EDM2
class Encoder:
    def __init__(self):
        pass

    def init(self, device): # force lazy init to happen now
        pass

    def __getstate__(self):
        return self.__dict__

    def encode(self, x): # raw pixels => final latents
        return self.encode_latents(self.encode_pixels(x))

    def encode_pixels(self, x): # raw pixels => raw latents
        raise NotImplementedError # to be overridden by subclass

    def encode_latents(self, x): # raw latents => final latents
        raise NotImplementedError # to be overridden by subclass

    def decode(self, x): # final latents => raw pixels
        raise NotImplementedError # to be overridden by subclass

#----------------------------------------------------------------------------
# Standard RGB encoder that scales the pixel data into [-1, +1].
class StandardRGBEncoder(Encoder):
    def __init__(self):
        super().__init__()

    def encode_pixels(self, x): # raw pixels => raw latents
        return x

    def encode_latents(self, x): # raw latents => final latents
        return x.to(torch.float32) / 127.5 - 1

    def decode(self, x): # final latents => raw pixels
        return (x.to(torch.float32) * 127.5 + 128).clip(0, 255).to(torch.uint8)


def write_general_info(path: str, **kwargs):
    with open(path, "w") as f:
        for key, value in kwargs.items():
            f.write(f"{key}: {value}\n")


def broadcast_vector(vector: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
    return vector.view(tensor.shape[0], *([1] * (tensor.dim() - 1)))


def load_edm_checkpoint(url: str) -> tuple[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]:
    # Load model
    with dnnlib.util.open_url(url) as f:
        data = pickle.load(f)
    model = data["ema"]

    # Load encoder
    encoder = data.get('encoder', None)
    if encoder is None:
        encoder = StandardRGBEncoder()

    return model, encoder


def plot_images(images: list[str], n_cols: int, col_labels: tuple[int,...] = None) -> plt.Figure:
    # Plot images
    fig, axes = plt.subplots(len(images) // n_cols, n_cols)
    fig.set_size_inches(3 * n_cols + 1, 3 * len(images) // n_cols + 1)

    for index, image_path in enumerate(images):
        j = index // n_cols
        k = index % n_cols

        image_arr = plt.imread(image_path)
        try:
            axes[j][k].imshow(image_arr)
            axes[j][k].axis("off")
        except TypeError:
            axes[k].imshow(image_arr)
            axes[k].axis("off")
        if j == 0 and col_labels is not None:
            axes[j][k].set_title(f"NFE = {col_labels[k]}", fontsize=20)

    fig.tight_layout()
    return fig


def load_config(path: str):
    spec = importlib.util.spec_from_file_location("config", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.get_config()


def compute_discretisation_interpolation(
        paths: np.ndarray,
        num_points: int = 200,
        t_min: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """
    :param paths: Matrix of paths, where paths that have finished are 0 from that point on
    :param num_points: Number of points to compute the interpolation at
    :returns: time grid, the interpolated means, and interpolated variances
    """
    t_grid = np.linspace(0, 1, num_points)
    interpolated = np.empty((paths.shape[0], num_points))

    for i, path in enumerate(paths):
        # Remove all 0s from the path, as 0s indicate the path is finished
        end_index = np.where(path <= t_min)[0][0]
        path = path[:(end_index + 1)]

        if end_index == 1:
            continue

        # Compute interpolation
        path_grid = np.linspace(0, 1, path.shape[0])
        interpolated[i] = np.interp(t_grid, path_grid, path)

    return t_grid, interpolated


class ImageSampleDataset(torch.utils.data.Dataset):

    def __init__(self, image_dir: str, n_images: int = 0, transform = None):
        self._images = self.parse_dirs(pathlib.Path(image_dir))

        if n_images:
            self._images = self._images[:n_images]

        self._n_images = n_images
        self._transform = transform

    @staticmethod
    def parse_dirs(file_iterable):
        flattened = []
        for file in file_iterable.iterdir():
            if file.is_dir():
                flattened.extend(ImageSampleDataset.parse_dirs(file))
            else:
                flattened.append(file)
        return flattened

    def __len__(self):
        return len(self._images)

    def __getitem__(self, item):
        image = decode_image(self._images[item])
        if self._transform:
            image = self._transform(image)
        return image


###
# From https://github.com/NVlabs/edm2/blob/main/generate_images.py
class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])
###
