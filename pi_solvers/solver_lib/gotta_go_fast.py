"""Adaptive solver based on Jolicoeur-Martineau et al. (2021)"""
from typing import Callable

import torch

from pi_solvers.solver_lib.solvers import Solver
from pi_solvers.sde_lib import SDE
from pi_solvers.utils import broadcast_vector


class GottaGoFast(Solver):
    """
    The GottaGoFast solver implements the custom solver from the paper
    Gotta Go Fast When Generating Data with Score-Based Models.
    It uses the extrapolated error to alter step sizes at runtime using a simple step size
    controller.
    """

    def __init__(
            self,
            sde: SDE,
            tau_a: float,
            tau_r: float,
            h_start: float,
            r: float,
            alpha: float,
            interval: tuple[float, float] = (1, 0),
            max_iter: int = 1000,
            max_increase: float = 5,
            max_decrease: float = 0.2,
            seed: int = 0,
            **kwargs
    ):
        """
        Constructs the GottaGoFast solver.
        
        :param sde: The SDE that needs to be solved.
        :param tau_a: Absolute tolerance
        :param tau_r: Relative tolerance
        :param h_start: Size of the starting step
        :param r: Hyperparameter for integration
        :param alpha: Safety factor
        :param interval: Interval of integration
        """
        super().__init__(sde, seed)
        self._tau_a = torch.tensor([tau_a])
        self._tau_r = torch.tensor([tau_r])
        self._r = r
        self._alpha = alpha
        self._start_time = float(interval[0])
        self._end_time = float(interval[1])
        self._h_start = abs(h_start) * (self._end_time - self._start_time) / abs(self._end_time - self._start_time)
        self._max_iter = max_iter
        self._max_increase = torch.Tensor([max_increase])
        self._max_decrease = torch.Tensor([max_decrease])

    def solve(self, x: torch.tensor, labels: torch.Tensor = None,
              callback: Callable[[torch.Tensor, torch.Tensor], None] = None) -> torch.tensor:
        t = broadcast_vector(torch.full((x.shape[0],), self._start_time), x).to(
            self._device)  # Initialise batch_size times, starting at 1
        h = broadcast_vector(torch.full((x.shape[0],), self._h_start), x).to(self._device)
        x_first_prev_full = x.clone()
        prev_rejected_full = torch.zeros(x.shape[0], dtype=torch.bool).to(self._device)
        prev_w_full = torch.zeros_like(x)
        error = torch.zeros(x.shape[0]).to(self._device)
        end_condition = broadcast_vector(torch.full((x.shape[0],), self._end_time), x).to(self._device)
        if labels is None:
            labels = torch.zeros(x.shape[0],).to(self._device)
        x_full, t_full, h_full = x, t, h
        i = 0

        # Loop until all times in the batch are equal to the end time
        while torch.any(not_finished := (t_full != end_condition)) and i < self._max_iter:
            # Work with the unfinished subset of x, t, h
            not_finished = not_finished.reshape(-1)
            x, t, h = x_full[not_finished], t_full[not_finished], h_full[not_finished]
            x_first_prev = x_first_prev_full[not_finished]
            prev_rejected = prev_rejected_full[not_finished]
            prev_w = prev_w_full[not_finished]

            # Perform Euler and Heun step
            w = torch.randn_like(x, generator=self._rng)
            w[prev_rejected] = prev_w[prev_rejected]

            dx_euler = self.sde.step(x, t, h, w, labels=labels[not_finished])

            # Multiplying by ((t + h) > 0) to make sure that this equals the euler step for points
            # taking the final step, so network not evaluated at t=0.
            dx_heun = self.sde.step(x + dx_euler * ((t + h) > 0), t + h * ((t + h) > 0), h, w, labels=labels[not_finished])

            # Compute first and second order x
            x_first = x + dx_euler
            x_second = x + 0.5 * (dx_euler + dx_heun)

            # Compute extrapolated error
            error[not_finished] = self._error(x_first, x_first_prev, x_second)

            # Update x and t
            not_rejected = error[not_finished] < 1
            x[not_rejected] = x_second[not_rejected]
            t[not_rejected] = t[not_rejected] + h[not_rejected]
            x_first_prev[not_rejected] = x_first[not_rejected]

            # Prevent new sampling of w for rejections
            prev_rejected = torch.logical_not(not_rejected)
            prev_w[prev_rejected] = w[prev_rejected]

            # Get next step size
            h = self._get_next_step(h, error[not_finished])

            # Bound steps such that no step exceeding end condition will be taken
            h = torch.maximum(h, end_condition[not_finished] - t)

            # Update the full matrices
            x_full[not_finished], t_full[not_finished], h_full[not_finished] = x, t, h
            x_first_prev_full[not_finished] = x_first_prev
            prev_rejected_full[not_finished] = prev_rejected
            prev_w_full[not_finished] = prev_w

            if callback is not None:
                callback(x=x_full, t=t_full, h=h_full, error=error)
            i += 1

        return x_full

    def _error(self, x_first: torch.Tensor, x_first_prev: torch.Tensor, x_second: torch.Tensor):
        d = sum(x_first.shape[1:])
        delta = torch.max(self._tau_a, self._tau_r * torch.max(torch.abs(x_first), torch.abs(x_first_prev)))
        return 1 / (d**0.5) * torch.sum((((x_first - x_second) / delta)**2).view(x_first.shape[0], -1), dim=1)

    def _get_next_step(self, h: torch.Tensor, error: torch.Tensor):
        error_pow = torch.where(h == 0, h, broadcast_vector(error, h) ** -self._r)
        return h * torch.max(self._max_decrease, torch.min(self._max_increase, self._alpha * error_pow))

    def to(self, device: torch.device | str) -> 'Solver':
        super().to(device)
        self._tau_r = self._tau_r.to(device)
        self._tau_a = self._tau_a.to(device)
        self._max_increase = self._max_increase.to(self._device)
        self._max_decrease = self._max_decrease.to(self._device)
        return self
