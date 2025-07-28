import torch
from torch import nn


class QuantileLoss(nn.Module):
    def __init__(
        self,
        manual_quantiles=None,
        auto_quantiles=3,
        reduction="mean",
    ) -> None:
        """Quantile Loss as defined in:

        R. Wen, K. Torkkola, B. Narayanaswamy, et D. Madeka,
        "A Multi-Horizon Quantile Recurrent Forecaster", arXiv:1711.11053.

        You can use `auto_quantiles` to generate the desired number of quantiles.
        For example, `auto_quantiles=3` will set the quantiles to [0.25, 0.5, 0.75].

        Note that this means that the 0.5 quantile is only present on even values of `auto_quantiles`.

        Note: `manual_quantiles` will always override `auto_quantiles` when provided.

        Parameters
        ----------
        manual_quantiles : list[float], optional
            If provided, the quantile points as floats, by default None
        auto_quantiles : int, optional
            The length of the evenly-spaced quantile list to generate, by default 3
        reduction : str, optional
            How to reduce the batch dimension. Must be one of "mean", "sum", by default "mean"


        Parameters of the forward function
        ----------
        model_out : torch.Tensor
            model output, of shape (batch_size, n_quantiles)
        target : torch.Tensor
            point targets, of shape (batch_size, 1)
        """
        super().__init__()

        if manual_quantiles is not None:
            self.quantiles = manual_quantiles
        else:
            auto_quantiles = int(auto_quantiles)
            assert auto_quantiles > 0, (
                f"If manual_quantiles is None, auto_quantiles must be a valid number of quantiles (got {auto_quantiles=})"
            )
            # Generate the desired number of quantiles
            gap = 1 / (auto_quantiles + 1)
            self.quantiles = [i * gap for i in range(1, auto_quantiles + 1)]

        assert reduction in [
            "mean",
            "sum",
        ], f"reduction must be one of ['mean', 'sum'], got '{reduction}'"
        self.reduction = reduction

    def _ql(self, y: torch.Tensor, y_hat: torch.Tensor, q: float):
        """Quantile loss function"""
        z = torch.zeros_like(y)
        return q * torch.maximum((y - y_hat), z) + (1 - q) * torch.maximum(
            (y_hat - y), z
        )

    def forward(self, model_out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        model_out : torch.Tensor
            model output, of shape (batch_size, n_quantiles)
        target : torch.Tensor
            point targets, of shape (batch_size, 1)

        Returns
        -------
        torch.Tensor
            Reduced quantile loss
        """
        losses = []
        for i, q in enumerate(self.quantiles):
            # TODO vectorize
            losses.append(self._ql(target, model_out[..., i], q))

        all_losses = torch.cat(losses)  # Shape (batch_size, n_quantiles)
        quantile_losses = torch.sum(all_losses, dim=-1)  # Shape (batch_size)
        if self.reduction == "mean":
            return torch.mean(quantile_losses)  # Shape (1)
        elif self.reduction == "sum":
            return torch.sum(quantile_losses)  # Shape (1)
