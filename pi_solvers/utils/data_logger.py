from abc import abstractmethod, ABC
import os

import pandas as pd
import torch


class DataLogger(ABC):

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @abstractmethod
    def write(self):
        pass


class PIDataLogger(DataLogger):

    def __init__(self, write_path: str, max_iter: int = 1000, batch_size: int = 64, device: torch.device | str = "cpu"):
        self._write_path = write_path

        self._batch_size = batch_size
        self._ts = torch.zeros(batch_size, max_iter).to(device)
        self._hs = torch.zeros(batch_size, max_iter).to(device)
        self._errors = torch.zeros(batch_size, max_iter).to(device)

        self._i = 0
        self._device = device

    def __call__(self, x: torch.Tensor, t: torch.Tensor, h: torch.Tensor, error: torch.Tensor, *args, **kwargs):
        return self.callback(x, t, h, error)

    def callback(self, x: torch.Tensor, t: torch.Tensor, h: torch.Tensor, error: torch.Tensor):
        batch_size = x.shape[0]

        self._ts[:batch_size, self._i] = t.to(self._device).reshape(batch_size)
        self._hs[:batch_size, self._i] = h.to(self._device).reshape(batch_size)
        self._errors [:batch_size, self._i] = error.to(self._device).reshape(batch_size)

        self._i += 1

    def write(self):
        os.makedirs(self._write_path, exist_ok=True)
        self._i = 0

        ts, hs, errors = pd.DataFrame(self._ts.numpy()), pd.DataFrame(self._hs.numpy()), pd.DataFrame(self._errors.numpy())
        t_path, h_path, error_path = self._write_path + "_t.csv", self._write_path + "_h.csv", self._write_path + "_error.csv"

        ts.to_csv(t_path, mode="a", header=False if os.path.exists(t_path) else True)
        hs.to_csv(h_path, mode="a", header=False if os.path.exists(h_path) else True)
        errors.to_csv(error_path, mode="a", header=False if os.path.exists(error_path) else True)

        self._ts = torch.zeros_like(self._ts)
        self._hs = torch.zeros_like(self._hs)
        self._errors = torch.zeros_like(self._errors)


class RejectCounter(DataLogger):
    def __init__(self):
        self.__reject_count = 0
        self.__total = 0

    def __call__(self, error: torch.Tensor, *args, **kwargs):
        return self.callback(error)

    def callback(self, error: torch.Tensor):
        self.__total += error.shape[0]
        self.__reject_count += int(torch.sum(error > 1))

    def reject_rate(self):
        return self.__reject_count / self.__total

    def write(self):
        pass


class LastTimeLogger(DataLogger):
    def __init__(self):
        self._t = -1

    @property
    def t(self):
        return self._t

    def __call__(self, t: torch.Tensor, *args, **kwargs):
        self._t = t

    def write(self):
        pass
