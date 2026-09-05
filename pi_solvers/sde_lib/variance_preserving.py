"""Variance preserving SDE definitions."""
from abc import abstractmethod, ABC

import torch

from pi_solvers.sde_lib import LinearDriftSDE


class VariancePreservingSDE(LinearDriftSDE, ABC):
    r"""
    Implements the Variance Preserving SDE by Song et al. (2021). This SDE is given by
    :math:`dx = -\frac{1}{2}\beta(t)xdt + \sqrt{\beta(t)}dw`, where :math:`\beta(t)` is any differentiable function of t.
    """

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return 1 - torch.exp(-self._B(t))

    def mu(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.exp(-0.5 * self._B(t))

    def drift(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        return -0.5 * self._beta(t) * x

    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self._beta(t))

    @abstractmethod
    def _beta(self, t: torch.Tensor) -> torch.Tensor:
        r"""
        The beta function, a differentiable function of t.

        :param t: The times t, a tensor of shape (batch_size, 1)
        :return: :math:`\beta(t)`
        """
        pass

    @abstractmethod
    def _B(self, t: torch.Tensor) -> torch.Tensor:
        r"""
        The antiderivative of the beta function, :math:`B(t) = \int \beta(t)dt`

        :param t: The times t, a tensor of shape (batch_size, 1)
        :return: :math:`B(t)`
        """
        pass


class LinearVariancePreservingSDE(VariancePreservingSDE):
    r"""
    A Variance Preserving SDE with a linear beta function, given by
    :math:`\beta(t) = (\beta_{max} - \beta_{min}) * t + \beta_{min}`
    """

    def __init__(self, beta_min: float, beta_max: float):
        r"""
        Constructs the LinearVariancePreservingSDE.

        :param beta_min: Minimum value of :math:`\beta`.
        :param beta_max: Maximum value of :math:`\beta`.
        """
        super().__init__()

        self._beta_min = beta_min
        self._beta_max = beta_max

    def _beta(self, t: torch.Tensor) -> torch.Tensor:
        return self._beta_min + t * (self._beta_max - self._beta_min)

    def _B(self, t: torch.Tensor) -> torch.Tensor:
        return self._beta_min * t + 0.5 * torch.square(t) * (self._beta_max - self._beta_min)


class StableDiffusionVPSDE(VariancePreservingSDE):

    def __init__(
            self,
            beta_min: float = 0.00085,
            beta_max : float = 0.012,
            interval: tuple[int, int] = (0, 999),
            ode: bool = False,
            seed: int = 0
    ):
        super().__init__(ode=ode, seed=seed)
        self._beta_min = beta_min
        self._beta_max = beta_max
        self._interval = interval
        self._n = interval[1] - interval[0] + 1

        self._betas = torch.linspace(0.00085 ** 0.5, 0.012 ** 0.5, self._n, dtype=torch.float32) ** 2
        alphas = 1.0 - self._betas
        self._alpha_bar = torch.cumprod(alphas, dim=0)  # index 0..999

    def get_index_from_t(self, t):
        rounded = torch.round(t)
        return torch.clamp(rounded, min=self._interval[0] * torch.ones_like(t), max=self._interval[1] * torch.ones_like(t)).to(torch.int)

    def _beta(self, t: torch.Tensor) -> torch.Tensor:
        index = self.get_index_from_t(t)
        return self._betas[index]

    def _B(self, t: torch.Tensor) -> torch.Tensor:
        pass

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        index = self.get_index_from_t(t)
        return torch.sqrt(1 - self._alpha_bar[index])

    def to(self, device: str) -> 'SDE':
        self._betas = self._betas.to(device)
        self._alpha_bar = self._alpha_bar.to(device)
        return super().to(device)
