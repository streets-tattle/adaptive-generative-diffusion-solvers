import PIL
import os
from abc import ABC, abstractmethod
import pathlib
import re
import random
import scienceplots

import torch
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np

from pi_solvers import sde_lib
from pi_solvers.solver_lib import construct_heun_end_adaptive_solver, PISolver
from pi_solvers.utils import plot_images, load_edm_checkpoint
from pi_solvers.utils.data_logger import RejectCounter
from pi_solvers.evaluation import feature_vector, metrics


class Rater(ABC):

    @abstractmethod
    def rate(self, eval_point, images: list[str]) -> float:
        pass


class ManualRater(Rater):

    def __init__(self, n_cols: int):
        self.n_cols = n_cols

    def rate(self, eval_point, images: list[str]) -> float:
        # Plot images
        plot_images(images, self.n_cols).show()

        # Get image rating
        while True:
            try:
                rating = int(input("What is the quality of the following images (1-5):\n"))
                assert 0 < rating <= 5
                break
            except (ValueError, AssertionError):
                print("Please provide an integer from 1 to 5")

        return float(rating)


class MetricRater(Rater):

    def __init__(
            self,
            metric: metrics.Metrics,
            ref: torch.Tensor,
            batch_size: int = 64,
            device: str | torch.device = "cuda"
    ):
        self._metric = metric
        self._ref = ref
        self._batch_size = batch_size
        self._device = device

    def rate(self, eval_point: str, images: list[str]) -> float:
        features = feature_vector.detect_image_features(
            image_dir=eval_point, batch_size=self._batch_size, device=self._device
        )
        return self._metric(self._ref, features)


def image_path_iterator(target_path: pathlib.Path) -> list[tuple[str, list[str]]]:
    subdirs = []
    for subdir in target_path.iterdir():
        if not subdir.is_file():
            images = []
            for image_path in pathlib.Path(subdir).iterdir():
                images.append(image_path)
            subdirs.append((str(subdir), images))
    return subdirs


def evaluate_images(target_path: str, rater: Rater, seed = 42) -> pd.DataFrame:
    random.seed(seed)

    df = pd.DataFrame(columns=["tau_a", "tau_r", "rating"])

    paths = image_path_iterator(pathlib.Path(target_path))
    shuffled_paths = random.sample(paths, len(paths))

    for i, (eval_point, images) in tqdm.tqdm(enumerate(shuffled_paths)):
        rating = rater.rate(eval_point, images)

        # Extract tau_a, tau_r from eval point
        tau_a = float(re.search(r"abs_\d+.?\d*", eval_point).group(0)[4:])
        tau_r = float(re.search(r"rel_\d+.?\d*", eval_point).group(0)[4:])

        # Write to df in form tau_a, tau_r, rating
        df.loc[i] = [tau_a, tau_r, rating]

    return df


def get_ratings_grid(grid, rating_df: pd.DataFrame) -> torch.Tensor:
    av, rv = grid
    ratings = torch.zeros_like(av)

    for i in range(av.shape[1]):
        for j in range(av.shape[0]):
            tau_a, tau_r = round(float(av[i, j]), 3), round(float(rv[i, j]), 3)
            ratings[i, j] = rating_df[np.logical_and(rating_df["tau_a"] == tau_a, rating_df["tau_r"] == tau_r)]["rating"].iloc[0]

    return ratings


def create_grid(tau_a_range: tuple[float, float], tau_r_range: tuple[float, float], resolution: int) -> tuple[
    torch.Tensor, torch.Tensor]:
    tau_as = torch.linspace(tau_a_range[0], tau_a_range[1], resolution)
    tau_rs = torch.linspace(tau_r_range[0], tau_r_range[1], resolution)
    av, rv = torch.meshgrid((tau_as, tau_rs))
    return av, rv


def plot_grid(
        grid: tuple[torch.Tensor, torch.Tensor],
        data: torch.Tensor,
        outdir: str,
        name: str,
        title: str = None,
        gamma: float = 1
):
    # Create and save plots
    plt.figure()
    plt.style.use("science")
    mesh = plt.pcolormesh(grid[0], grid[1], data, cmap='inferno', norm=PowerNorm(gamma=gamma))
    plt.colorbar(mesh, label=f'{name}')
    plt.xlabel(r"$\tau_a$", fontsize=15)
    plt.ylabel(r"$\tau_r$", fontsize=15)
    if not title:
        title = rf"{name} as a function of $\tau_a$ and $\tau_r$"
    # plt.title(title)
    plt.savefig(outdir + f"/{name}.png")


def apply_over_grid(
        tau_a_min: float,
        tau_a_max: float,
        tau_r_min: float,
        tau_r_max: float,
        resolution: int,
        outdir: str,
        model: str,
        ode: bool,
        seed: int,
        batch_size: int,
        device: torch.device | str,
        **pi_kwargs) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:

    # Set seed
    torch.random.manual_seed(seed)

    # Load model
    model, encoder = load_edm_checkpoint(model)
    model = model.to(device)

    # Create SDE
    sde = sde_lib.EDMSDE(ode=ode).to(device)
    rsde = sde.get_reverse_sde(model).to(device)

    # Create grid
    grid = create_grid((tau_a_min, tau_a_max), (tau_r_min, tau_r_max), resolution)

    # Sample noise and labels
    x = torch.zeros((batch_size, model.img_channels, model.img_resolution, model.img_resolution)).to(device)
    noise = torch.randn_like(x) * 80
    labels = torch.eye(model.label_dim, device=device)[
        torch.randint(high=model.label_dim, size=(batch_size,), device=device)]

    nfes = torch.zeros_like(grid[0])
    reject_rate = torch.zeros_like(grid[0])

    # Loop over grid
    for i in tqdm.tqdm(range(grid[0].shape[1])):
        for j in tqdm.tqdm(range(grid[0].shape[0])):
            torch.random.manual_seed(seed)

            tau_a, tau_r = float(grid[0][i, j]), float(grid[1][i, j])
            rsde.reset()

            reject_counter = RejectCounter()

            solver = construct_heun_end_adaptive_solver(
                rsde,
                adaptive_solver_class=PISolver,
                tau_a=tau_a,
                tau_r=tau_r,
                seed=seed,
                **pi_kwargs
            ).to(device)

            images = solver.solve(noise.clone(), labels=labels, callback=reject_counter)

            for k, (image, label) in enumerate(zip(encoder.decode(images).permute(0, 2, 3, 1).cpu().numpy(), labels)):
                label = torch.argmax(label)
                dir_path = os.path.join(outdir, f"abs_{round(tau_a, 3)}_rel_{round(tau_r, 3)}")
                os.makedirs(dir_path, exist_ok=True)
                PIL.Image.fromarray(image, "RGB").save(os.path.join(dir_path, f"{k}_{label}.png"))

            nfes[i, j] = rsde.nfe / batch_size
            reject_rate[i, j] = reject_counter.reject_rate()

    return nfes, reject_rate, grid
