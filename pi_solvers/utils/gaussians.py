"""Gaussians, and their score."""
import math
from typing import Callable

import torch

from pi_solvers.sde_lib import LinearDriftSDE


class Gaussian:
    r"""
    A Gaussian, a function that follows:
    :math:`f(x) = \frac{1}{\sigma * \sqrt{2\pi}}e^{-\frac{(x - \mu)^2}{2\sigma^2}}`
    With score :math:`-\frac{x - \mu}{\sigma}`
    """

    def __init__(self,
                 mu: torch.Tensor | float,
                 sigma: torch.Tensor | float,
                 weight: float = 1
                 ):
        """
        Constructs the Gaussian.

        :param mu: The mean of the Gaussian.
        :param sigma: The standard deviation of the Gaussian.
        :param norm: Normalisation constant of the Gaussian.
        :param weight: Weight, useful for multimodal Gaussians.
        """
        self._mu = mu
        self._sigma = sigma

        # Set up weight
        self._weight = weight

        # Set normalisation constant
        self._norm = 1 / (sigma * math.sqrt(2 * math.pi)) * weight

    @property
    def mu(self) -> torch.Tensor:
        return self._mu

    @property
    def sigma(self) -> torch.Tensor:
        return self._sigma

    @property
    def norm(self) -> float:
        return self._norm

    @property
    def weight(self) -> float:
        return self._weight

    def score(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Computes the score at point x.

        :param x: Points for which to compute the score, a tensor of shape (batch_size, 1)
        :return: :math:`-\frac{x - \mu}{\sigma}`, a tensor of shape (batch_size, 1)
        """
        return - (x - self.mu) / self.sigma ** 2

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Computes the probability density of the Gaussian at points x.

        :param x: Points for which to compute the score, a tensor of shape (batch_size, 1)
        :return: :math:`ce^{-\frac{(x - \mu)^2}{2\sigma^2}}`
        """
        return self.norm * torch.exp(-0.5 * torch.square(x - self.mu) / self.sigma ** 2)


class MultiGaussian:
    r"""
    A multi-modal Gaussian, consisting of a sum of Gaussians:
    :math:`f(x) = \sum_{i=1}^n \pi_i f_i(x)`
    Where each :math:`f(i)` is a Gaussian. Implemented to allow for easy use with an SDE.
    """

    def __init__(self, gaussians: tuple[Gaussian, ...], sde: LinearDriftSDE):
        """
        Constructs the MultiModalGaussian

        :param gaussians: A tuple of n gaussians for which the total weights should add up to 1.
        :param sde: The stochastic differential equation that transforms the multi-modal Gaussian.
        """
        self._gaussians = gaussians
        self._sde = sde
        self._nfe = 0

    @property
    def nfe(self) -> int:
        return self._nfe

    def reset_nfe(self):
        self._nfe = 0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Computes the probability density at each point x.

        :param x: A tensor of shape (batch_size, 1) for which the probability density should be computed.
        :return: :math:`f(x) = \sum_{i=1}^n \pi_i f_i(x)`
        """
        total = torch.zeros(x.shape)
        for gaussian in self._gaussians:
            total += gaussian(x)
        return total

    def sample(self, n: int) -> torch.Tensor:
        """
        Samples elements from the MultiGaussian, using ancestral sampling

        :param n: The amount of samples to be obtained.
        :return: A tensor of shape (n,), following the distribution described by this MultiGaussian.
        """
        # Sample which gaussian each point belongs to from a multinomial.
        weights = torch.Tensor([gaussian.weight for gaussian in self._gaussians])
        indices = torch.multinomial(weights, n, replacement=True)

        # For each point which belongs to a certain Gaussian, sample from that Gaussian
        mus = torch.Tensor([gaussian.mu for gaussian in self._gaussians])
        sigmas = torch.Tensor([gaussian.sigma for gaussian in self._gaussians])
        return torch.normal(mus[indices], sigmas[indices])

    def get_score_function(self) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        r"""
        Obtains the score function from the Gaussian, so it can be used by a reverse SDE.

        :return: The score function :math:`\nabla_x \log p_t(x)`
        """
        return lambda x, t, _: self.score(x, t)

    def score(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        r"""
        Computes the score of the multi-modal Gaussian at time t for a datapoint x. This is computed by summing
        the scores of each individual Gaussian multiplied by their value at that point, normalised over the sum of all
        Gaussians. Based on the SDE, the mean and standard deviation are computed for each Gaussian at time t.

        :param x: The data for which to compute the score, a tensor of shape (batch_size, 1).
        :param t: The time at which to compute the score, a tensor of shape (batch_size, 1).
        :return: The score at x, t, a tensor of shape (batch_size, 1).
        """
        self._nfe += x.shape[0]

        # Apply kernel defined by SDE to gaussians, to determine their new mean and std.
        convolved_gaussians = self.convolve_gaussians(t)

        # Compute the sum of scores for each gaussian, computed by multiplying the score of the Gaussian by its value
        # at point x.
        non_normalised_score = sum(map(lambda gaussian: gaussian.score(x) * gaussian(x), convolved_gaussians))
        # Normalise by the sum of all Gaussians at point x, at small factor for when this is near 0
        normalisation = sum(map(lambda gaussian: gaussian(x), convolved_gaussians)) + 1e-13

        return torch.Tensor(non_normalised_score / normalisation)

    def convolve_gaussians(self, t: torch.Tensor) -> tuple[Gaussian, ...]:
        r"""
        Compute the probability distribution at time t, obtained by convolving the original distribution by the kernel
        defined by the SDE. Specifically, compute :math:`p_t(x_t) = \int_{-\infty}^\infty p(x_t|x_0)p(x_0)dx_0`. As
        this is an integral, the solution of this for a mult-modal gaussian is the sum of this integral for normal
        Gaussians. For these, :math:`\mu_{new} = \mu_0 * \alpha_t` and
        :math:`\sigma_{new}^2 = \alpha_t^2 * \sigma_0^2 + \sigma_t^2`. Here, :math:`\alpha_t, \sigma_t` are defined by
        the marginal distribution of the SDE.

        :param t: The time t at which the new PDF needs to be computed.
        :return: A tuple of Gaussians, defining a multimodal gaussian convolved with the gaussian defined by the SDEs
                 marginal.
        """
        convolved_gaussians = []

        # Obtain marginal distribution for convolution
        alpha_t, sigma_t = self._sde.marginal(torch.zeros(t.shape), t)

        # Compute new gaussian for each gaussian in the distribution.
        for gaussian in self._gaussians:
            mu_new = gaussian.mu * alpha_t
            sigma_new = torch.sqrt(alpha_t**2 * gaussian.sigma**2 + sigma_t**2)
            convolved_gaussians.append(Gaussian(mu_new, sigma_new, gaussian.weight))

        return tuple(convolved_gaussians)

    def multigaussian_at_t(self, t: torch.Tensor) -> 'MultiGaussian':
        return MultiGaussian(self.convolve_gaussians(t), self._sde)
