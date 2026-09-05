"""Computes metrics of the data"""
import enum
from abc import abstractmethod, ABC

import torch
from scipy.linalg import sqrtm
import numpy as np


class Metric(ABC):

    def pretty_print(self, x: torch.Tensor, x_hat: torch.Tensor) -> str:
        return str(self) + " = " + str(round(self(x, x_hat), 4))

    def uses_stats(self) -> bool:
        return False

    @abstractmethod
    def __call__(self, x: torch.Tensor, x_hat: torch.Tensor):
        pass


class MIND(Metric):
    """
    Computes the MIND metric from the paper MIND: Monge Inception Distance for Generative Models
    Evaluation: https://arxiv.org/html/2605.06797v1#A3.F10.sf2
    """

    def __init__(self, seed: int = 0, n_projections: int = 100):
        self._seed = seed
        self._n_projections = n_projections

    def __call__(self, x, x_hat):
        x = reduce_ref_dimensionality(x, x_hat, self._seed)
        n_samples, d = x_hat.shape

        generator = torch.Generator(device=x.device).manual_seed(self._seed)

        alpha = 3 * d

        u_proj = torch.randn(
            (self._n_projections, d),
            generator=generator,
            dtype=x.dtype,
            device=x.device
        )

        u_proj /= torch.linalg.norm(u_proj, dim=-1, keepdim=True)

        x_proj = u_proj @ x.T
        x_hat_proj = u_proj @ x_hat.T


        dists = torch.mean(
            (
                    torch.topk(x_hat_proj, n_samples, dim=-1).values -
                    torch.topk(x_proj, n_samples, dim=-1).values
            ) ** 2,
            dim=-1
        )

        return float(alpha * torch.mean(dists))

    def __str__(self):
        return "MIND"


class FID(Metric):
    """
    Frechet Inception Distance metric.
    """

    def __init__(self, dtype = torch.float64):
        self._dtype = dtype

    def uses_stats(self) -> bool:
        return True

    def __call__(self, x, x_hat):
        """
        Computes the frechet inception distance.

        :param x: Feature vectors of the reference dataset.
        :param x_hat: Feature vectors of the generated samples.
        :param dtype: Torch dtype
        :return: The value of the Frechet Inception Distance
        """
        x_hat = x_hat.to(self._dtype)

        if isinstance(x, dict):
            mu, sigma = x["mu"], x["sigma"]
        else:
            x = x.to(self._dtype)
            mu, sigma = self.calc_mean_covariance(x)

        mu_hat, sigma_hat = self.calc_mean_covariance(x_hat)

        mu_diff = np.sum((mu - mu_hat) ** 2)
        s, _ = sqrtm(sigma @ sigma_hat, disp=False)
        covs = sigma + sigma_hat - 2 * s
        return float(np.real(mu_diff + np.trace(covs)))

    def __str__(self):
        return "FID"

    @staticmethod
    def calc_mean_covariance(x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        return x.mean(dim=0).cpu().numpy(), x.T.cov().cpu().numpy()


class PrecisionRecall(Metric):
    """
    Precision and Recall Metric. From https://arxiv.org/pdf/1904.06991
    """

    def __init__(self, k: int = 3, dtype = torch.float32, seed: int = 0):
        self._k = k
        self._dtype = dtype
        self._seed = seed

    def pretty_print(self, x: torch.Tensor, x_hat: torch.Tensor) -> str:
        precision, recall = self(x, x_hat)
        return f"Precision = {precision} \nRecall = {recall}"

    def __call__(self, x: torch.Tensor, x_hat: torch.Tensor) -> tuple[float, float]:
        x = reduce_ref_dimensionality(x, x_hat, self._seed)

        x, x_hat = x.to(self._dtype), x_hat.to(self._dtype)

        dist_matrix = self.pairwise_distances(x, x_hat)

        precision = self.precision(x, dist_matrix)
        recall = self.recall(x_hat, dist_matrix)

        return precision, recall

    def precision(self, x: torch.Tensor, dist_matrix: torch.Tensor) -> float:
        x_manifold = self.kth_nearest_neighbour(x)

        f = torch.sum((dist_matrix.T <= x_manifold), dim=1) >= 1
        return float(1 / x.shape[0] * torch.sum(f))

    def recall(self, x_hat: torch.Tensor, dist_matrix: torch.tensor) -> float:
        x_hat_manifold = self.kth_nearest_neighbour(x_hat)

        f = torch.sum((dist_matrix <= x_hat_manifold), dim=0) >= 1
        return float(1 / x_hat.shape[0] * torch.sum(f))

    def pairwise_distances(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """
        Computes the pairwise euclidian distance matrix between each vector in x to each vector
        in x_hat.

        :param x: Matrix of shape (N, d)
        :param x_hat: Matrix of shape (N, d)
        :return: Pairwise distance matrix of shape (N, N). Rows are x, columns are x_hat
        """
        # (x - y)^2 = x^2 + y^2 - 2xy
        x, x_hat = x.to(self._dtype), x_hat.to(self._dtype)
        return (torch.sum(x**2, dim=1, keepdim=True) + torch.sum(x_hat**2, dim=1) - 2 * x @ x_hat.T).clamp(min=0)

    def kth_nearest_neighbour(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the distance to the kth nearest neighbour for each vector in x of shape
        (N, d)

        :param x: A matrix of feature vectors of shape (N, d)
        :return: A vector of shape (N,) where each value is the Euclidean distance to the kth
                nearest neighbour of that vector.
        """
        x = x.to(self._dtype)

        dist_matrix = self.pairwise_distances(x, x)
        topk, _ = torch.topk(dist_matrix, self._k + 1, dim=1, largest=False)
        return topk[:, -1]

    def __str__(self):
        return "PrecisionRecall"


def reduce_ref_dimensionality(x: torch.Tensor, x_hat: torch.Tensor, seed: int) -> torch.Tensor:
    """
    Equalises the dimensionality between reference data and samples.

    :param x: Reference data of shape (N, d).
    :param x_hat: Sample data (M <= N, d).
    :param seed: Random seed.
    :return: x with randomly selected vectors of shape (M, d)
    """
    generator = torch.Generator(device=x.device).manual_seed(seed)
    assert x_hat.shape[0] <= x.shape[0], "Ground truth needs to have at least as many samples as predicted"
    return x[torch.randint(x.shape[0], size=(x_hat.shape[0],), generator=generator)]


class Metrics(enum.Enum):
    MIND = MIND()
    FID = FID()
    PrecisionRecall = PrecisionRecall()

    def uses_stats(self):
        return self.value.uses_stats()

    def pretty_print(self, x: torch.Tensor, x_hat: torch.Tensor):
        return self.value.pretty_print(x, x_hat)

    def __call__(self, x: torch.Tensor, x_hat: torch.Tensor):
        return self.value(x, x_hat)

    def __str__(self):
        return str(self.value)
