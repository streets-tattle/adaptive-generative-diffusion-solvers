"""Definitions of the stochastic differential equations"""
from abc import abstractmethod, ABC
from typing import Callable

import torch


class SDE(ABC):
    r"""
    The Stochastic Differential Equation (SDE) abstract class. Assumes the SDE can be written in Ito form:
    :math:`dx = f(x, t)dt + g(t)dw`
    Where :math:`f(x,t)` is the drift, and :math:`g(t)` the diffusion.
    """

    def __init__(self, ode: bool = False, seed: int = 0):
        self._nfe = 0
        self._device = "cpu"
        self._ode = ode
        self.__original_ode = ode
        self._seed = seed
        self._rng = torch.Generator().manual_seed(self._seed)

    @property
    def ode(self) -> bool:
        return self._ode

    @ode.setter
    def ode(self, reverse_ode: bool):
        self._ode = reverse_ode

    @property
    def nfe(self):
        return self._nfe

    def set_seed(self, seed: int):
        self._rng = torch.Generator(self._rng.device).manual_seed(seed)

    def reset(self):
        self._nfe = 0
        self._ode = self.__original_ode

    def dummy_call(self, x):
        """
        Dummy call for counting NFE

        :param x: The given sample, a tensor of shape (batch_size, d)
        """
        self._nfe += x.shape[0]

    def sde(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the Ito parameters of the SDE for a given sample and time. The SDE is defined by the drift f(t) and
        the diffusion g(t).

        :param x: The given sample, a tensor of shape (batch_size, d)
        :param t: The given time, a tensor of shape (batch_size, 1)
        :param labels: Optional labels parameter to be passed into the drift
        :return: A tuple of (drift, diffusion)
        """
        self._nfe += x.shape[0]  # Add batch_size to NFE
        return self.drift(x, t, labels), self.diffusion(t)

    def __call__(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the Ito parameters of the SDE for a given sample and time. The SDE is defined by the drift f(t) and
        the diffusion g(t).

        :param x: The given sample, a tensor of shape (batch_size, d).
        :param t: The given time, a tensor of shape (batch_size, 1).
        :param labels: Optional labels parameter to be passed into the drift
        :return: A tuple of (drift, diffusion), with shapes (batch_size, d) and (batch_size, 1).
        """
        return self.sde(x, t)

    def step(self,
             x: torch.tensor,
             t: torch.tensor,
             dt: torch.tensor,
             w: torch.Tensor = None,
             labels: torch.Tensor = None
             ) -> torch.Tensor:
        r"""
        Takes a step of size dt.

        :param x: The given sample, a tensor of shape (batch_size, d)
        :param t: The given time, a tensor of shape (batch_size, 1)
        :param dt: The step size, a tensor of shape (batch_size, 1)
        :param w: The noise added. By default, this is set to :math:`\mathcal{N}(0, I)`. Provide if specific noise
                  needs to be added. A tensor of shape (batch_size, d)
        :param labels: Optional labels parameter to be passed into the drift
        :return: x + dx, a tensor of shape (batch_size, d)
        """
        if w is None:
            w = torch.randn_like(x, generator=self._rng).to(self._device)
        drift, diffusion = self.sde(x, t, labels)
        if self.ode:
            return drift * dt
        return drift * dt + diffusion * torch.sqrt(torch.abs(dt)) * w

    @abstractmethod
    def drift(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        """
        Gives the drift f(x, t) at given x, t of the SDE.

        :param x: The given sample, a tensor of shape (batch_size, d)
        :param t: The given time, a tensor of shape (batch_size, 1)
        :param labels: Optional labels parameter to indicate data labels
        :return: f(x, t), a tensor of shape (batch_size, d)
        """
        pass

    @abstractmethod
    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        """
        Gives the diffusion g(t) at a given time t.

        :param t: The given time, a tensor of shape (batch_size, 1)
        :return: g(t), a tensor of shape (batch_size, 1)
        """
        pass


class LinearDriftSDE(SDE, ABC):
    r"""
    This SDE represents a class of SDEs with drift linear in x. Specifically, :math:`f(x, t) = F(t)x` holds
    for this class of SDEs. This type of SDE has an analytical solution for the marginal.
    """

    def marginal(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the Gaussian marginal probability distribution of a given sample, at a given time t.
        Marginal distribution can be obtained by solving the Focker-Planck Equation.

        :param x: The given sample, a tensor of shape (batch_size, d)
        :param t: The given time, a tensor of shape (batch_size, 1)
        :return: A tuple of (mu, sigma), representing the mean and standard deviation of the sample. mu is a tensor
                 of shape (batch_size, d), sigma a tensor of shape (batch_size, 1)
        """
        return self.mu(x, t), self.sigma(t)

    def sample_noise(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Computes a sample x from the marginal :math:`p(x_t|x_0)`, returning the noise for training convenience.

        :param x: The given sample, a tensor of shape (batch_size, d)
        :param t: The times t from which should be sampled, a tensor of shape (batch_size, 1)
        :return: A tuple of (:math:`x \sim p(x_t|x_0), \epsilon`), both with shape (batch_size, d)
        """
        mu, sigma = self.marginal(x, t)
        noise = torch.randn_like(x, generator=self._rng).to(self._device)
        return mu * x + sigma * noise, noise

    def sample(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        r"""
        Computes a sample x from the marginal :math:`p(x_t|x_0)`.

        :param x: The given sample, a tensor of shape (batch_size, d)
        :param t: The times t from which should be sampled, a tensor of shape (batch_size, 1)
        :return: :math:`x \sim p(x_t|x_0)`, with shape (batch_size, d)
        """
        return self.sample_noise(x, t)[0]

    @abstractmethod
    def mu(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Computes the marginal mean of x at time t.

        :param x: The given sample, a tensor of shape (batch_size, d)
        :param t: The times t from which should be sampled, a tensor of shape (batch_size, 1)
        :return: The marginal mean of x at time t, a tensor of shape (batch_size, d)
        """
        pass

    @abstractmethod
    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """
        Computes the marginal standard deviation at time t.

        :param t: The times t from which should be sampled, a tensor of shape (batch_size, 1)
        :return: The marginal standard deviation at time t, a tensor of shape (batch_size, d)
        """
        pass


    def get_reverse_sde(self, score_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor], ode_threshold: float = 0) -> 'LinearDriftSDE':
        r"""
        Reverses the SDE to the form :math:`dx = (f(x, t) - g(t)^2 \nabla_x \log p_t(x))dt + g(t)dw`

        :param score_fn: The score function, a function s(x, t) for which :math:`s(x, t) \approx \nabla_x \log p_t(x)`
                         which takes a tensor of shape (batch_size, d) and (batch_size, 1) and maps it to another
                         tensor of (batch_size, d)
        :param ode_threshold: Time threshold for when to apply ODE over SDE.
        :return: The reversed SDE
        """
        parent = self

        # Construct ReverseSDE class as child from SDE, use parent drift and diffusion but update drift to
        # include the score
        class ReverseSDE(LinearDriftSDE):
            """A reversed SDE."""
            def __init__(self):
                super().__init__(ode=parent.ode)

                self._parent = parent
                self.to(parent._device)

            @property
            def parent(self) -> 'LinearDriftSDE':
                return parent

            def mu(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                return parent.mu(x, t)

            def sigma(self, t: torch.Tensor) -> torch.Tensor:
                return parent.sigma(t)

            def drift(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
                if self.ode:
                    return self.parent.drift(x, t) - 0.5 * torch.square(self.parent.diffusion(t)) * score_fn(x, t, labels)
                return self.parent.drift(x, t) - torch.square(self.parent.diffusion(t)) * score_fn(x, t, labels)

            def diffusion(self, t: torch.Tensor) -> torch.Tensor:
                return self.parent.diffusion(t)

            def step(self, x: torch.tensor, t: torch.tensor, dt: torch.tensor, w: torch.Tensor = None, labels: torch.Tensor = None) -> torch.Tensor:
                if w is None:
                    w = torch.randn_like(x, generator=self._rng).to(self._device)
                w[t.view(x.shape[0]) < ode_threshold] = torch.zeros(w[t.view(x.shape[0]) < ode_threshold].shape).to(self._device)
                if torch.any(t.view(x.shape[0]) < ode_threshold):
                    self.ode = True
                return super().step(x, t, dt, w, labels)

        return ReverseSDE()

    def to(self, device: str) -> 'SDE':
        self._device = device
        self._rng = torch.Generator(device).manual_seed(self._seed)
        return self
