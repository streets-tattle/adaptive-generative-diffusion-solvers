"""Generates feature vectors of data"""
import os
from abc import abstractmethod, ABC
import pickle

import numpy as np
import torch
import tqdm

from pi_solvers import dnnlib, utils


class FeatureDetector(ABC):

    def __init__(self, n_features: int):
        self._n_features = n_features

    @abstractmethod
    def __call__(self, x: torch.Tensor):
        pass

    @abstractmethod
    def to(self, device: str | torch.device):
        pass

    @property
    def n_features(self):
        return self._n_features


# From EDM2 repository: https://github.com/NVlabs/edm2/tree/main
class InceptionV3Detector(FeatureDetector):

    def __init__(self, n_features: int = 2048):
        super().__init__(n_features)

        url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
        with dnnlib.util.open_url(url, verbose=False) as f:
            self.model = pickle.load(f)

    def __call__(self, x: torch.Tensor):
        return self.model(x, return_features=True)

    def to(self, device: str | torch.device):
        self.model.to(device)


def detect_image_features(
        image_dir: str,
        batch_size: int,
        device: str | torch.device = "cpu",
        n_images: int = 0,
        save_path: str = None,
        detector: FeatureDetector = InceptionV3Detector(),
):
    """
    Detects the feature vectors of provided images.

    :param image_dir: Directory where the images can be found
    :param batch_size: Batch size
    :param detector: The detector used to detect features.
    :param device: Torch device on which calculations are done
    :param n_images: Limits the number of images on which evaluation is done
    :param save_path: Optional save path where feature vectors are stored.
    :return: The feature vectors of shape (n x 2048)
    """
    print("Loading images from disk...")

    images = utils.ImageSampleDataset(image_dir, n_images=n_images)
    dataloader = torch.utils.data.DataLoader(images, batch_size=batch_size, pin_memory=True, num_workers=4)

    features = np.zeros((len(images), detector.n_features), dtype=np.float32)
    detector.to(device)

    bar = tqdm.tqdm(total=len(images), unit="img")

    for i, image_batch in enumerate(dataloader):
        with torch.no_grad():
            feature_vectors = detector(image_batch.to(device)).to(torch.float64).to("cpu")

        features[i * batch_size : i * batch_size + image_batch.shape[0]] = feature_vectors.numpy()

        bar.update(image_batch.shape[0])

    bar.close()

    features = torch.from_numpy(features)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(features, save_path)

    return features
