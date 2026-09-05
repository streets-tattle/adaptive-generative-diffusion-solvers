"""The proportional integral solver, based on Ilie et al (2015)."""
from typing import Callable
import torch

from pi_solvers.solver_lib import HeunSolver, get_edm_schedule
from pi_solvers.solver_lib.solvers import Solver
from pi_solvers.sde_lib import SDE, LinearDriftSDE
from pi_solvers.utils import broadcast_vector


class PISolver(Solver):
    """
    The PISolver uses an adaptive proportional-integral method to solve the SDE. It dynamically adjusts the step size
    based on local measured error for each sample in the batch. This error is computed using an embedded stochastic
    Runge-Kutta method of second order, extrapolating this and comparing it with the first-order error.
    """

    def __init__(self,
                 sde: LinearDriftSDE,
                 ki: float,
                 kp: float,
                 tau_a: float,
                 tau_r: float,
                 alpha: float,
                 h_start: float,
                 max_increase: float,
                 max_decrease: float,
                 interval: tuple[float, float] = (1, 0),
                 max_iter: int = 10000,
                 abs_error: bool = False,
                 batch_norm: bool = False,
                 seed: int = 0,
                 **kwargs
                 ):
        r"""
        Constructs the PISolver.

        :param sde: The SDE the solver will have to solve.
        :param ki: The integral constant, determining influence of integral component.
        :param kp: The proportional constant, determining influence of integral component.
        :param tau_a: The absolute tolerance tau_a, the minimum tolerance of the solver.
        :param tau_r: The relative tolerance tau_r, the factor that scales tolerance by pixel magnitude
        :param alpha: The safety factor, for which :math:`\alpha \leq 1`, reduces chance of rejecting next step size.
        :param h_start: The starting step size.
        :param max_increase: A factor determining the maximum increase of the step size for each step.
        :param max_decrease: A factor determining the maximum decrease of the step size for each step.
        :param interval: The interval over which the SDE is computed
        """
        super().__init__(sde, seed=seed)
        self._ki = ki
        self._kp = kp
        self._tau_a = tau_a
        self._tau_r = tau_r
        self._alpha = alpha
        self._start_time = float(interval[0])
        self._end_time = float(interval[1])
        self._h_start = abs(h_start) * (self._end_time - self._start_time) / abs(self._end_time - self._start_time)
        self._max_increase = torch.Tensor([max_increase])
        self._max_decrease = torch.Tensor([max_decrease])
        self._max_iter = max_iter
        self._abs_error = abs_error
        self._batch_norm = batch_norm

    def solve(self, x: torch.Tensor, labels: torch.Tensor = None,
              callback: Callable[[torch.Tensor, torch.Tensor], None] = None) -> torch.Tensor:
        t = broadcast_vector(torch.full((x.shape[0],), self._start_time), x).to(self._device)  # Initialise batch_size times, starting at 1
        h = broadcast_vector(torch.full((x.shape[0],), self._h_start), x).to(self._device)
        error = torch.full((x.shape[0],), self._alpha * self._tau_a).to(self._device)
        end_condition = broadcast_vector(torch.full((x.shape[0],), self._end_time), x).to(self._device)
        if labels is None:
            labels = torch.zeros(x.shape[0],).to(self._device)
        reject_count = 0
        not_reject_count = 0
        x_full, t_full, h_full = x, t, h
        i = 0

        # Loop until all times in the batch are equal to the end time
        while torch.any(not_finished := (t_full > end_condition)) and i < self._max_iter:
            # Work with the unfinished subset of x, t, h
            not_finished = not_finished.reshape(-1)
            x, t, h = x_full[not_finished], t_full[not_finished], h_full[not_finished]

            # Perform Euler and Heun step
            w = torch.randn_like(x, generator=self._rng)
            dx_euler = self.sde.step(x, t, h, w, labels=labels[not_finished])

            # Multiplying by ((t + h) > 0) to make sure that this equals the euler step for points
            # taking the final step, so network not evaluated at t=0.
            dx_heun = self.sde.step(x + dx_euler * ((t + h) > 0), t + h * ((t + h) > 0), h, w, labels=labels[not_finished])

            # Compute first and second order x
            x_first = x + dx_euler
            x_second = x + 0.5 * (dx_euler + dx_heun)

            # Compute extrapolated error
            old_error = error.clone()
            new_error = self._error(x_first, x_second, t)

            if self._batch_norm:
                new_error = torch.mean(new_error) * torch.ones_like(new_error)

            error[not_finished] = new_error

            # Update x and t
            not_rejected = error[not_finished] < 1000
            x[not_rejected] = x_second[not_rejected]
            t[not_rejected] = t[not_rejected] + h[not_rejected]

            reject_count += torch.sum(error > 1)
            not_reject_count += torch.sum(not_rejected)

            # Get next step size
            h = self._get_next_step(error[not_finished], old_error[not_finished], h)

            # Bound steps such that no step exceeding end condition will be taken
            h = torch.maximum(h, end_condition[not_finished] - t)

            # Update the full matrices
            x_full[not_finished], t_full[not_finished], h_full[not_finished] = x, t, h

            if callback is not None:
                callback(x=x_full, t=t_full, h=h_full, error=error)
            i += 1

        # print(reject_count / (reject_count + not_reject_count))
        return x_full

    def _error(self, x_first: torch.Tensor, x_second: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Computes the local extrapolated error by computing the MSE divided by the tolerance.

        :param x_first: The first order estimate of x, tensor of shape (batch_size, d).
        :param x_second: The higher order estimate of x, tensor of shape (batch_size, d).
        :return: The error, a tensor of shape (batch_size, 1)
        """
        d = sum(x_first.shape[1:])
        if d != 1 and not self._abs_error:
            norm = self._tau_a + self._tau_r * self.sde.sigma(t)
        elif self._abs_error:
            norm = self._tau_a + self._tau_r * torch.abs(x_second)
        else:
            norm = broadcast_vector(self._tau_a + self._tau_r * torch.abs(x_second), x_second)
        squared_error = torch.square((x_second - x_first) / norm)
        return torch.sqrt(torch.sum(squared_error.view(x_first.shape[0], -1), dim=1) / d)

    def _get_next_step(self, error: torch.Tensor, error_previous, h: torch.Tensor) -> torch.Tensor:
        """
        Computes the next step h for each x based on the error using a PI controller.

        :param error: Current error, tensor of shape (batch_size, 1).
        :param error_previous: Error of previous step, tensor of shape (batch_size, 1).
        :param h: Current step size, tensor of shape (batch_size, 1).
        :return: The new step size, a tensor of shape (batch_size, 1).
        """
        integral = (self._alpha * self._tau_a / error)**(self._ki + self._kp)
        proportional = (error_previous / (self._alpha * self._tau_a))**self._kp
        return h * broadcast_vector(torch.clamp(integral * proportional, min=self._max_decrease, max=self._max_increase), h)

    def to(self, device: str) -> Solver:
        super().to(device)
        self._max_increase = self._max_increase.to(self._device)
        self._max_decrease = self._max_decrease.to(self._device)
        return self


def construct_heun_end_adaptive_solver(adaptive_solver_class,
                                       sde: SDE,
                                       ode_threshold: float,
                                       n_ode_steps: int,
                                       interval: tuple[float, float] = (80, 0.002),
                                       rho: float = 7,
                                       **kwargs
                                       ):
    class HeunEndSolver(adaptive_solver_class):

        def __init__(self):
            pi_interval = (interval[0], ode_threshold)
            super().__init__(sde, interval=pi_interval, **kwargs)

            discretisation = get_edm_schedule(n_ode_steps, interval[1], ode_threshold, rho=rho)
            self._ode_solver = HeunSolver(sde, discretisation)

        def solve(self, x: torch.Tensor, labels: torch.Tensor = None,
                  callback: Callable[[torch.Tensor, torch.Tensor], None] = None) -> torch.Tensor:
            x_sde = super().solve(x, labels, callback)
            self.sde.ode = True
            return self._ode_solver.solve(x_sde, labels)

        def to(self, device: str) -> Solver:
            super().to(device)
            self._ode_solver.to(device)
            return self

    return HeunEndSolver()
