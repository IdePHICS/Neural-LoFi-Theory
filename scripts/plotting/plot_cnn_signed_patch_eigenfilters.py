from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config_utils import with_config
from _plot_utils import setup_style

from train_hierarchically.datasets import DatasetConfig, build_dataset
from train_hierarchically.utils.helpers import resolve_device, set_seed

log = logging.getLogger(__name__)


def _build_loader(cfg) -> DataLoader:
    raw_classes = cfg.dataset.get("classes")
    classes: list[int] | None = (
        cast(list[int], OmegaConf.to_object(raw_classes)) if raw_classes else None
    )

    ds_cfg = DatasetConfig(
        dataset_type=cfg.dataset.name,
        split=cfg.split,
        root=cfg.root,
        n_samples=cfg.dataset.n_train,
        seed=cfg.seed,
        flatten=False,
        class_preset=cfg.dataset.get("preset"),
        classes=classes,
        remap_labels=cfg.remap_labels,
    )
    ds = build_dataset(ds_cfg)

    return DataLoader(
        ds,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
    )


def _channel_patch_matrix(
    x_channel: torch.Tensor,
    kernel_size: int,
    padding: int,
    stride: int,
) -> tuple[torch.Tensor, int]:
    """Return flattened channel patches as (N*S, k*k) and S locations/sample."""
    x = x_channel.unsqueeze(1)
    if padding > 0:
        x = F.pad(x, [padding] * 4, mode="constant", value=0.0)

    patches = x.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
    n_locations = int(patches.shape[2] * patches.shape[3])
    z = patches.contiguous().view(x.shape[0] * n_locations, kernel_size * kernel_size)
    return z, n_locations


def _top_signed_eigens(
    loader: DataLoader,
    device: torch.device,
    kernel_size: int,
    padding: int,
    stride: int,
    n_top: int,
    channels_to_use: int,
) -> tuple[list[np.ndarray], list[np.ndarray], int]:
    dim = kernel_size * kernel_size
    signed_cov = [
        torch.zeros((dim, dim), device=device, dtype=torch.float32)
        for _ in range(channels_to_use)
    ]
    total_locations = 0

    for x, y in tqdm(loader, desc="Accumulating signed covariance", leave=False):
        x_dev = x.to(device=device, dtype=torch.float32)
        y_dev = y.to(device=device, dtype=torch.float32)
        y_pm1 = torch.where(y_dev > 0.0, 1.0, -1.0)

        for ch in range(channels_to_use):
            z, n_locations = _channel_patch_matrix(
                x_dev[:, ch, :, :],
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
            )
            y_expanded = y_pm1.repeat_interleave(n_locations)
            signed_cov[ch] += (y_expanded[:, None] * z).T @ z

        total_locations += int(x_dev.shape[0]) * n_locations

    if total_locations <= 0:
        msg = "No samples were processed; cannot compute covariance."
        raise RuntimeError(msg)

    eigenvalues_out: list[np.ndarray] = []
    eigenvectors_out: list[np.ndarray] = []

    for ch in range(channels_to_use):
        c_signed = signed_cov[ch] / float(total_locations)
        # MPS does not currently support eigh for this op path; diagonalize on CPU.
        c_signed_cpu = c_signed.to(device="cpu")
        eigvals, eigvecs = torch.linalg.eigh(c_signed_cpu)
        order = torch.argsort(torch.abs(eigvals), descending=True)[:n_top]
        top_vals = eigvals[order].detach().cpu().numpy()
        top_vecs = eigvecs[:, order].detach().cpu().numpy()

        eigenvalues_out.append(top_vals)
        eigenvectors_out.append(top_vecs)

    return eigenvalues_out, eigenvectors_out, total_locations


_CHANNEL_LABELS = ["R", "G", "B"]
_CHANNEL_COLORS = ["#D62728", "#2CA02C", "#1F77B4"]
_CHANNEL_CMAPS = [
    LinearSegmentedColormap.from_list("red_div", ["#D62728", "#F7F7F7", "#17BECF"]),
    LinearSegmentedColormap.from_list("green_div", ["#2E8B57", "#F7F7F7", "#E67E22"]),
    LinearSegmentedColormap.from_list("blue_div", ["#1F77B4", "#F7F7F7", "#F1C40F"]),
]

_IMGS_DIR = Path(__file__).resolve().parents[2] / "imgs" / "cnn_signed_patch_eigenfilters"


def _plot_all_channels_eigenvectors(
    out_dir: Path,
    eigenvalues_by_ch: list[np.ndarray],
    eigenvectors_by_ch: list[np.ndarray],
    kernel_size: int,
    cmap: str,
    dataset_name: str,
) -> None:
    setup_style()

    n_channels = len(eigenvectors_by_ch)
    n_top = int(eigenvectors_by_ch[0].shape[1])

    # Layout: rows = eigenvectors (n_top), cols = channels (n_channels)
    # Figure height = 15 cm (will be scaled down in LaTeX)
    height_cm = 15.0
    height_inches = height_cm / 2.54
    cell_size_inches = height_inches / n_top
    width_inches = cell_size_inches * n_channels

    fig, axes = plt.subplots(
        n_top,
        n_channels,
        figsize=(width_inches, height_inches),
        squeeze=False,
    )

    for ch in range(n_channels):
        eigen_mats = eigenvectors_by_ch[ch].T.reshape(-1, kernel_size, kernel_size)
        col_max = float(np.max(np.abs(eigen_mats)))
        vmax = col_max if col_max > 0.0 else 1.0
        ch_cmap = _CHANNEL_CMAPS[ch] if ch < len(_CHANNEL_CMAPS) else cmap
        ch_color = _CHANNEL_COLORS[ch] if ch < len(_CHANNEL_COLORS) else "black"
        ch_label = _CHANNEL_LABELS[ch] if ch < len(_CHANNEL_LABELS) else f"C{ch}"

        for idx in range(n_top):
            ax = axes[idx][ch]
            ax.imshow(
                eigen_mats[idx],
                cmap=ch_cmap,
                vmin=-vmax,
                vmax=vmax,
                origin="upper",
                aspect="auto",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.margins(0)
            for spine in ax.spines.values():
                spine.set_linewidth(0.3)

        axes[0][ch].set_title(ch_label, color=ch_color, fontsize=33, fontweight="bold", pad=6)

    fig.subplots_adjust(left=0, right=1, top=0.98, bottom=0)
    fig.get_axes()[0].get_gridspec().update(wspace=0, hspace=0)

    out_png = out_dir / "signed_patch_eigenfilters_all_channels.png"
    out_pdf = out_dir / "signed_patch_eigenfilters_all_channels.pdf"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")

    _IMGS_DIR.mkdir(parents=True, exist_ok=True)
    imgs_png = _IMGS_DIR / f"{dataset_name}.png"
    imgs_pdf = _IMGS_DIR / f"{dataset_name}.pdf"
    fig.savefig(imgs_png, dpi=160, bbox_inches="tight")
    fig.savefig(imgs_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {imgs_png} and {imgs_pdf}")


def _save_channel_arrays(
    out_dir: Path,
    channel_idx: int,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
) -> None:
    vals_path = out_dir / f"signed_patch_eigenvalues_channel_{channel_idx}.csv"
    vecs_path = out_dir / f"signed_patch_eigenvectors_channel_{channel_idx}.csv"

    np.savetxt(
        vals_path,
        eigenvalues.reshape(1, -1),
        delimiter=",",
        header=",".join([f"lambda_{i + 1}" for i in range(eigenvalues.shape[0])]),
        comments="",
    )
    np.savetxt(
        vecs_path,
        eigenvectors,
        delimiter=",",
        header=",".join([f"v_{i + 1}" for i in range(eigenvectors.shape[1])]),
        comments="",
    )


@with_config("conf/plot_cnn_signed_patch_eigenfilters.yaml")
def main(cfg, out_dir: Path) -> None:
    set_seed(int(cfg.seed))
    device = resolve_device(str(cfg.device))

    if int(cfg.kernel_size) <= 0:
        raise ValueError("kernel_size must be a positive integer.")

    loader = _build_loader(cfg)
    log.info(
        "Computing signed patch covariance on %s with batch_size=%d",
        device,
        int(cfg.batch_size),
    )

    first_batch = next(iter(loader))
    x0, _ = first_batch
    if x0.ndim != 4:
        raise ValueError(
            "Expected image tensor (N,C,H,W), "
            f"got shape={tuple(x0.shape)}"
        )

    n_input_channels = int(x0.shape[1])
    if n_input_channels < 3:
        raise ValueError(
            f"Expected at least 3 channels, got {n_input_channels}."
        )
    channels_to_use = int(cfg.channels_to_use)
    if channels_to_use > n_input_channels:
        raise ValueError(
            f"channels_to_use={channels_to_use} > input channels={n_input_channels}."
        )

    eigenvalues_by_ch, eigenvectors_by_ch, n_total_locations = _top_signed_eigens(
        loader=loader,
        device=device,
        kernel_size=int(cfg.kernel_size),
        padding=int(cfg.padding),
        stride=int(cfg.stride),
        n_top=int(cfg.n_top),
        channels_to_use=channels_to_use,
    )

    metadata_path = out_dir / "metadata.txt"
    metadata_lines = [
        f"dataset={cfg.dataset.name}",
        f"preset={cfg.dataset.get('preset')}",
        f"split={cfg.split}",
        f"device={device}",
        f"n_train={cfg.dataset.n_train}",
        f"batch_size={cfg.batch_size}",
        f"kernel_size={cfg.kernel_size}",
        f"padding={cfg.padding}",
        f"stride={cfg.stride}",
        f"channels_to_use={channels_to_use}",
        f"n_total_locations={n_total_locations}",
    ]
    metadata_path.write_text("\n".join(metadata_lines) + "\n", encoding="utf-8")

    for ch in range(channels_to_use):
        _save_channel_arrays(
            out_dir=out_dir,
            channel_idx=ch,
            eigenvalues=eigenvalues_by_ch[ch],
            eigenvectors=eigenvectors_by_ch[ch],
        )

    _plot_all_channels_eigenvectors(
        out_dir=out_dir,
        eigenvalues_by_ch=eigenvalues_by_ch,
        eigenvectors_by_ch=eigenvectors_by_ch,
        kernel_size=int(cfg.kernel_size),
        cmap=str(cfg.cmap),
        dataset_name=str(cfg.dataset.name),
    )

    log.info("Saved outputs to %s", out_dir)


if __name__ == "__main__":
    main()
