"""Base interfaces for models."""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseIRTModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor):
        raise NotImplementedError

    @abstractmethod
    def loss_function(self, x: torch.Tensor, recon_x: torch.Tensor, z: torch.Tensor):
        raise NotImplementedError
