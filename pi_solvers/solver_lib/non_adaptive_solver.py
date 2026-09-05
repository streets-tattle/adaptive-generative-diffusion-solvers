"""Definitions of non-adaptive solvers, includes Euler-Marayuma, Heun, and EDM solver."""
from typing import Callable

import numpy as np
import pandas as pd
import torch

from pi_solvers.sde_lib import SDE, EDMSDE
from pi_solvers.utils import broadcast_vector, compute_discretisation_interpolation
from pi_solvers.solver_lib.solvers import Solver


class EulerMarayumaSolver(Solver):
    """
    The Euler Marayuma solver uses a simple first order scheme and pre-determined time-discretisation to solve the SDE.
    """

    def __init__(self, sde: SDE, discretisation: torch.Tensor, seed: int = 0, **kwargs):
        r"""
        Constructs the EulerMarayuma Solver

        :param sde: The SDE the solver will have to solve.
        :param discretisation: The time steps :math:`(t_0, t_1, ..., t_n)` the solver will solve the SDE over.
        """
        super().__init__(sde, seed=seed)

        self._discretisation = discretisation
        self._time_steps = discretisation[1:] - discretisation[:-1]

    def solve(self, x: torch.tensor, labels: torch.Tensor = None,
              callback: Callable[[torch.Tensor, torch.Tensor], None] = None) -> torch.tensor:
        for i, dt in enumerate(self._time_steps):
            t = broadcast_vector((self._discretisation[i] * torch.ones(x.shape[0]).to(self._device)), x)
            x += self.sde.step(x, t, dt, labels=labels)

            if callback is not None:
                callback(x=x, t=t + dt)

        return x

    def to(self, device: str) -> Solver:
        super().to(device)
        self._discretisation.to(self._device)
        return self


class HeunSolver(EulerMarayumaSolver):

    def solve(self, x: torch.tensor, labels: torch.Tensor = None,
              callback: Callable[[torch.Tensor, torch.Tensor], None] = None) -> torch.tensor:
        for i, dt in enumerate(self._time_steps):
            t = broadcast_vector((self._discretisation[i] * torch.ones(x.shape[0])).to(self._device), x)
            w = torch.randn_like(x, generator=self._rng)
            dx_euler = self.sde.step(x, t, dt, w=w, labels=labels)

            if i < (self._time_steps.shape[0] - 1):
                dx_heun = self.sde.step(x + dx_euler, t + dt, dt, w=w, labels=labels)
                x += 0.5 * (dx_euler + dx_heun)
            else:
                x += dx_euler

            if callback is not None:
                callback(x=x, t=t + dt)

        return x


def get_edm_schedule(
        n_steps: int,
        t_min: float = 0.002,
        t_max: float = 80,
        rho: float = 7
):
    step_indices = torch.arange(n_steps)
    discretisation = (t_max ** (1 / rho) + step_indices / (n_steps - 1)
               * (t_min ** (1 / rho) - t_max ** (1 / rho))) ** rho
    discretisation = torch.cat([discretisation, torch.zeros_like(discretisation[:1])])
    return discretisation


def get_entropy_schedule(
        n_steps: int,
        entropy_checkpoint: str = "../../refs/img64_rescaled_entropic_time.pt"
):
    times = torch.load(entropy_checkpoint)["time"].flip(dims=(0,))
    xp = np.arange(0, times.shape[0])
    points = np.linspace(0, times.shape[0], n_steps - 1)
    discretisation = np.interp(points, xp, times.numpy())
    discretisation = np.concatenate([discretisation, np.array([0])])
    return torch.tensor(discretisation)


def get_pi_schedule(
        n_steps: int,
        n_ode_steps: int,
        pi_paths_file: str,
        t_max: float = 80,
        t_ode: float = 0.05,
        t_min: float = 0.002
):
    # Load in t csv
    ts = pd.read_csv(pi_paths_file).to_numpy()
    # Add start time to histogram
    ts[:, 0] = np.full(ts.shape[0], t_max)

    # Get average schedule from the mean of the interpolation of all paths
    # Add one step as the last step is removed to not repeat a step from t_ode to t_ode
    _, pi_interpolation = compute_discretisation_interpolation(ts, n_steps + 1, t_min=t_min)
    pi_schedule = pi_interpolation.mean(axis=0)

    if t_ode == t_max:
        return torch.cat([pi_schedule, 0])

    # Add EDM schedule for the last few steps
    pi_schedule = torch.tensor(pi_schedule)[:-1]
    edm_end = get_edm_schedule(n_ode_steps, t_min, t_ode)

    return torch.cat([pi_schedule, edm_end])


# Adapted from EDM2: https://github.com/NVlabs/edm2/tree/main
class EDMSolver(Solver):
    """
    Implementation of the sampler implemented in the paper Elucidating the design space for
    diffusion models (EDM). Is not strictly an SDE solver. This class ignores the SDE completely
    and just solves an ODE with potentially some parametrised noise added. Class created for
    compatibility with the rest of the framework of this repository.
    """

    def __init__(
            self,
            sde: SDE,
            discretisation: torch.Tensor,
            model: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
            g_model: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] = None,
            guidance: int = 1,
            S_churn: float = 0,
            S_min: float = 0,
            S_max: float = float('inf'),
            S_noise: float = 1,
            dtype: torch.dtype = torch.float32,
            seed: int = 0,
            **kwargs
    ):
        r"""
        Construct the EDM solver.

        :param discretisation: The time steps :math:`(t_0, t_1, ..., t_n)` the solver will solve the SDE over.
        :param model: The denoiser network :math:`D_\theta(x, t)`
        :param g_model: The optional guidance network.
        :param guidance: True if guidance should be enabled and a guidance network is passed.
        :param S_churn: How much the next time step will be upscaled for churning.
        :param S_min: Minimum time step for enabling churn.
        :param S_max: Maximum time step for enabling churn.
        :param S_noise: How much noise should be added when churn.
        :param dtype: Dtype of all tensors.
        """
        super().__init__(sde, seed=seed)
        self._discretisation = discretisation
        self._model = model
        self._g_model = g_model
        self._guidance = guidance
        self._S_churn = S_churn
        self._S_min = S_min
        self._S_max = S_max
        self._S_noise = S_noise
        self._dtype = dtype
    
    def solve(self, x: torch.Tensor, labels: torch.Tensor = None,
              callback: Callable[[torch.Tensor, torch.Tensor], None] = None) -> torch.tensor:
        n_steps = self._discretisation.shape[0] - 1
        x_next = x.to(self._dtype)
        for i, (t_cur, t_next) in enumerate(zip(self._discretisation[:-1], self._discretisation[1:])):  # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            if self._S_churn > 0 and self._S_min <= t_cur <= self._S_max:
                gamma = min(self._S_churn / n_steps, 2**0.5 - 1)
                t_hat = t_cur + gamma * t_cur
                x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * self._S_noise * torch.randn_like(x_cur, generator=self._rng)
            else:
                t_hat = t_cur
                x_hat = x_cur

            # Euler step.
            self.sde.dummy_call(x_cur)
            t_hat = broadcast_vector(t_hat * torch.ones(x_hat.shape[0]).to(self._device), x_hat)
            d_cur = (x_hat - self.denoise(x_hat, t_hat, labels)) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < n_steps - 1:
                self.sde.dummy_call(x_cur)
                t_next = broadcast_vector(t_next * torch.ones(x_hat.shape[0]).to(self._device), x_hat)
                d_prime = (x_next - self.denoise(x_next, t_next, labels)) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next
    
    def denoise(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        Dx = self._model(x, t, labels).to(self._dtype)
        if self._guidance == 1:
            return Dx
        ref_Dx = self._g_model(x, t, labels).to(self._dtype)
        return ref_Dx.lerp(Dx, self._guidance)

    def to(self, device):
        super().to(device)
        self._discretisation =  self._discretisation.to(device)
        return self