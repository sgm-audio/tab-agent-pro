"""Compute device resolution for local GPU backends.

Tab Agent runs entirely on the end user's machine — there is no cloud
inference. This module resolves a user's ``DEVICE`` preference (or auto-detects)
into:

* a **torch device** string used by Demucs stem separation
* a list of **ONNX execution providers** used by Basic Pitch transcription
* a human-readable label for logs and the web UI

Supported backends:

* ``cuda``   — NVIDIA CUDA (torch and ONNX CUDA EP)
* ``rocm``   — AMD ROCm (torch reports ROCm as CUDA via HIP; ONNX ROCm EP)
* ``vulkan`` — AMD/Intel/NVIDIA Vulkan (ONNX Vulkan EP only; Demucs stays on CPU)
* ``mps``    — Apple Silicon Metal
* ``directml`` — Windows DirectML (ONNX DirectML EP only; Demucs stays on CPU)
* ``cpu``    — plain CPU (default, works everywhere)

Basic Pitch hardcodes ``providers=["CPUExecutionProvider"]`` upstream, so to
actually run Basic Pitch on a GPU the matching ONNX Runtime package must be
installed (``onnxruntime-gpu`` for CUDA, ``onnxruntime-rocm`` for ROCm,
``onnxruntime-vulkan`` for Vulkan, ``onnxruntime-directml`` for DirectML).
This module applies a small, isolated override so Basic Pitch picks up the
requested GPU provider whenever one is available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from monitoring import get_logger

logger = get_logger("device")

_VALID_BACKENDS = ("auto", "cpu", "cuda", "rocm", "vulkan", "mps", "directml")

# Ordered for the UI: most capable/common first.
DEVICE_BACKENDS = [
    ("auto", "Auto (detect best)"),
    ("cuda", "NVIDIA CUDA"),
    ("rocm", "AMD ROCm"),
    ("vulkan", "AMD Vulkan"),
    ("mps", "Apple Silicon (MPS)"),
    ("directml", "Windows DirectML"),
    ("cpu", "CPU"),
]

# Human-readable labels for each backend.
_LABELS = {
    "cpu": "CPU",
    "cuda": "NVIDIA CUDA",
    "rocm": "AMD ROCm (HIP)",
    "vulkan": "Vulkan",
    "mps": "Apple Metal (MPS)",
    "directml": "Windows DirectML",
}


@dataclass(frozen=True)
class DeviceConfig:
    """Resolved compute configuration."""

    backend: str
    """One of: cuda, rocm, vulkan, mps, directml, cpu."""

    torch_device: str
    """Device string passed to torch/Demucs: cuda, cpu, or mps."""

    onnx_providers: tuple[str, ...]
    """ONNX Runtime execution providers, GPU first then CPU fallback."""

    label: str
    """Human-readable name shown in logs and the UI."""

    requested: str
    """The ``DEVICE`` value the user asked for (``auto`` if unset)."""

    resolved_from: str
    """``explicit``, ``auto-detected``, or ``fallback``."""


def _torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _torch_mps_available() -> bool:
    try:
        import torch

        return bool(getattr(torch, "backends", None) and torch.backends.mps.is_available())  # type: ignore[attr-defined]
    except Exception:
        return False


def _is_rocm_build() -> bool:
    """PyTorch built with ROCm sets ``torch.version.hip``; CUDA builds set
    ``torch.version.cuda`` instead. They are mutually exclusive at build time."""
    try:
        import torch

        return bool(getattr(torch.version, "hip", None))
    except Exception:
        return False


def _onnx_providers_available() -> tuple[str, ...]:
    try:
        import onnxruntime as ort

        return tuple(ort.get_available_providers())
    except Exception:
        return ()


def _providers_for(backend: str, available: tuple[str, ...]) -> tuple[str, ...]:
    """Return the ONNX provider list for a backend, GPU first, CPU last."""
    first: str | None = None
    if backend == "cuda" and "CUDAExecutionProvider" in available:
        first = "CUDAExecutionProvider"
    elif backend == "rocm":
        if "ROCmExecutionProvider" in available:
            first = "ROCmExecutionProvider"
        elif "CUDAExecutionProvider" in available:
            # ROCm torch maps to CUDA EP under the hood.
            first = "CUDAExecutionProvider"
    elif backend == "vulkan" and "VulkanExecutionProvider" in available:
        first = "VulkanExecutionProvider"
    elif backend == "directml" and "DmlExecutionProvider" in available:
        first = "DmlExecutionProvider"
    elif backend == "mps" and "CoreMLExecutionProvider" in available:
        first = "CoreMLExecutionProvider"
    if first:
        return (first, "CPUExecutionProvider")
    return ("CPUExecutionProvider",)


def resolve_device(pref: str | None = None) -> DeviceConfig:
    """Resolve a device preference into a concrete :class:`DeviceConfig`.

    Parameters
    ----------
    pref:
        User-requested device (``auto``, ``cpu``, ``cuda``, ``rocm``,
        ``vulkan``, ``mps``, ``directml``). Defaults to the ``DEVICE`` env
        var, then ``auto``.

    Returns
    -------
    DeviceConfig
        The resolved configuration. An explicit GPU request that cannot be
        satisfied falls back to CPU with a logged warning rather than raising,
        so a transcription run never dies on device selection alone.
    """
    requested = (pref or os_getenv("DEVICE") or "auto").strip().lower()
    if requested not in _VALID_BACKENDS:
        logger.warning("device_unknown", requested=requested)
        requested = "auto"

    available = _onnx_providers_available()
    cuda = _torch_cuda_available()
    mps = _torch_mps_available()
    rocm_build = _is_rocm_build()

    def build(backend: str, source: str) -> DeviceConfig:
        torch_device = _torch_device_for(backend)
        return DeviceConfig(
            backend=backend,
            torch_device=torch_device,
            onnx_providers=_providers_for(backend, available),
            label=_LABELS[backend],
            requested=requested,
            resolved_from=source,
        )

    def cannot_satisfy(backend: str) -> DeviceConfig:
        logger.warning(
            "device_unavailable",
            requested=requested,
            backend=backend,
            fallback="cpu",
        )
        return build("cpu", "fallback")

    if requested == "cuda":
        if not cuda:
            return cannot_satisfy("cuda")
        return build("rocm" if rocm_build else "cuda", "explicit")
    if requested == "rocm":
        if not cuda or not rocm_build:
            return cannot_satisfy("rocm")
        return build("rocm", "explicit")
    if requested == "vulkan":
        if "VulkanExecutionProvider" not in available:
            return cannot_satisfy("vulkan")
        return build("vulkan", "explicit")
    if requested == "mps":
        if not mps:
            return cannot_satisfy("mps")
        return build("mps", "explicit")
    if requested == "directml":
        if "DmlExecutionProvider" not in available:
            return cannot_satisfy("directml")
        return build("directml", "explicit")
    if requested == "cpu":
        return build("cpu", "explicit")

    # auto: pick the best available backend
    if cuda:
        return build("rocm" if rocm_build else "cuda", "auto-detected")
    if mps:
        return build("mps", "auto-detected")
    if "VulkanExecutionProvider" in available:
        return build("vulkan", "auto-detected")
    if "DmlExecutionProvider" in available:
        return build("directml", "auto-detected")
    return build("cpu", "auto-detected")


def _torch_device_for(backend: str) -> str:
    """Map a backend to a torch device string.

    Demucs/torch only speak CUDA/CPU/MPS. ROCm builds accept ``cuda``
    (HIP presents as CUDA), and Vulkan/DirectML have no torch backend, so
    Demucs falls back to CPU while Basic Pitch uses the ONNX GPU provider.
    """
    if backend == "mps":
        return "mps"
    if backend in ("cuda", "rocm"):
        return "cuda"
    return "cpu"


def install_onnx_provider_override(config: DeviceConfig) -> None:
    """Patch ``onnxruntime.InferenceSession`` so Basic Pitch (which hardcodes
    ``providers=["CPUExecutionProvider"]`` upstream) uses the resolved GPU
    provider when one is available.

    Safe to call multiple times; only installs the override once. A CPU-only
    config leaves ONNX Runtime untouched.
    """
    if config.backend == "cpu":
        return

    try:
        import onnxruntime as ort
    except Exception:
        logger.warning("onnxruntime_missing", detail="cannot override ONNX providers")
        return

    if getattr(ort.InferenceSession, "_tab_agent_patched", False):
        return

    real_session = ort.InferenceSession
    wanted = config.onnx_providers

    def patched(path_or_bytes, providers=None, sess_options=None, **kwargs):
        # Basic Pitch explicitly passes CPU-only providers; upgrade them to the
        # requested GPU provider(s) when available.
        if providers == ["CPUExecutionProvider"] or providers is None:
            providers = list(wanted)
        return real_session(path_or_bytes, providers=providers, sess_options=sess_options, **kwargs)

    patched._tab_agent_patched = True  # type: ignore[attr-defined]
    ort.InferenceSession = patched  # type: ignore[assignment]
    logger.info(
        "onnx_provider_override_installed",
        backend=config.backend,
        providers=",".join(wanted),
    )


def os_getenv(name: str, default: str | None = None) -> str | None:
    """Thin wrapper so :func:`resolve_device` can be monkeypatched in tests."""
    import os

    return os.getenv(name, default)
