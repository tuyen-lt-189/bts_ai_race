#!/usr/bin/env bash
#
# Native 3DGUT renderer matching the 3DGRUT v2 training launcher.
#
# Expected layout:
#
#   ~/ai-tuyen/
#   ├── bts_ai_race/
#   │   ├── setup_and_train_3dgrut.sh
#   │   └── render_3dgrut.sh
#   ├── dataset/phase1/public_set/HCM0181/
#   │   ├── train/images
#   │   ├── train/sparse/0
#   │   └── test/
#   │       ├── test_poses.csv
#   │       └── images                 # optional GT for evaluation
#   └── 3dgrut_workspace/3dgrut/
#       ├── .venv
#       └── runs
#
# Usage:
#
#   bash render_3dgrut.sh HCM0181
#   CHECKPOINT_STEP=30000 bash render_3dgrut.sh HCM0181
#   CHECKPOINT_STEP=last bash render_3dgrut.sh HCM0181
#   ENABLE_EVALUATION=false bash render_3dgrut.sh HCM0181
#
set -Eeuo pipefail
IFS=$'\n\t'

on_error() {
    local exit_code=$?
    echo
    echo "============================================================" >&2
    echo "RENDER FAILED at line ${BASH_LINENO[0]} (exit ${exit_code})" >&2
    echo "Log: ${LOG_FILE:-not-created}" >&2
    echo "============================================================" >&2
    exit "$exit_code"
}
trap on_error ERR

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
AI_TUYEN_ROOT="$(cd "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd)"

SCENE_NAME="${1:-${SCENE_NAME:-HCM0181}}"
PUBLIC_SET_ROOT="${PUBLIC_SET_ROOT:-$AI_TUYEN_ROOT/dataset/phase1/public_set}"
DATASET_SOURCE="${DATASET_SOURCE:-}"

WORK_ROOT="${WORK_ROOT:-$AI_TUYEN_ROOT/3dgrut_workspace}"
REPO_DIR="${REPO_DIR:-$WORK_ROOT/3dgrut}"
OUT_ROOT="${OUT_ROOT:-$REPO_DIR/runs}"
VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv}"

# The official installer places slangc in .venv/bin. Rendering uses Python
# through an absolute path, so explicitly expose the whole environment too.
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$VENV_DIR}"
export UV_PYTHON="${UV_PYTHON:-$VENV_DIR/bin/python}"
export PATH="$VENV_DIR/bin:$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export LD_LIBRARY_PATH="$VENV_DIR/lib:$VENV_DIR/lib64:${LD_LIBRARY_PATH:-}"

# Slang compiler installed by the official 3DGRUT environment setup.
# Keep this as an absolute path because 3DGRUT launches `slangc` from a
# Python subprocess while building the 3DGUT plugin.
SLANGC_BIN="${SLANGC_BIN:-$VENV_DIR/bin/slangc}"
SLANGC_EXPECTED_VERSION="${SLANGC_EXPECTED_VERSION:-2026.5.2}"

# HARD GPU LOCK:
# Always use physical GPU 1. CUDA masks it as logical cuda:0 inside Python.
unset CUDA_VISIBLE_DEVICES
unset CUDA_DEVICE_ORDER
readonly GPU_ID=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export PHYSICAL_GPU_ID=1

APPEARANCE_MODE="${APPEARANCE_MODE:-native_distortion}"
MAX_STEPS="${MAX_STEPS:-35000}"
CAP_MAX="${CAP_MAX:-2000000}"
NHT_FEATURE_DIM="${NHT_FEATURE_DIM:-48}"
RUN_NAME="${RUN_NAME:-${SCENE_NAME}_3dgrut_v2_3dgut_mcmc_nht_${APPEARANCE_MODE}_$((MAX_STEPS / 1000))k_$((CAP_MAX / 1000000))m_fd${NHT_FEATURE_DIM}}"

# Use an isolated cache compiled from the exact checkpoint 3DGUT config.
# This avoids reusing the earlier experimental 3DGRT/safe-culling extensions.
TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$WORK_ROOT/torch_extensions/native_3dgut_${SCENE_NAME}_fd${NHT_FEATURE_DIM}}"
mkdir -p "$TORCH_EXTENSIONS_DIR"
export TORCH_EXTENSIONS_DIR

# last = newest ckpt_last.pt, or newest exact checkpoint if no last exists.
CHECKPOINT_STEP="${CHECKPOINT_STEP:-last}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"

LPIPS_NET="${LPIPS_NET:-vgg}"
ENABLE_EVALUATION="${ENABLE_EVALUATION:-auto}"
USE_NATIVE_DISTORTION="${USE_NATIVE_DISTORTION:-true}"
USE_FEATURE_DECODER_EMA="${USE_FEATURE_DECODER_EMA:-true}"

OVERWRITE_RENDER="${OVERWRITE_RENDER:-false}"
SAVE_ALPHA="${SAVE_ALPHA:-false}"
MAX_IMAGES="${MAX_IMAGES:-}"
FORCE_OUTPUT_EXTENSION="${FORCE_OUTPUT_EXTENSION:-}"
RESIZE_PREDICTION_TO_GT="${RESIZE_PREDICTION_TO_GT:-false}"
CONTINUE_ON_EVAL_ERROR="${CONTINUE_ON_EVAL_ERROR:-true}"

# Continue a partially rendered set by keeping existing images.
# Set OVERWRITE_RENDER=true to regenerate every image.

# Avoid the known tiny-cuda-nn RTC warning on this server.
TCNN_JIT_FUSION="${TCNN_JIT_FUSION:-0}"
export TCNN_JIT_FUSION

# Preserve tqdm/Rich output while also writing a log.
USE_PTY_LOGGING="${USE_PTY_LOGGING:-true}"
PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_ALLOC_CONF
export PYTHONUNBUFFERED=1

CUDA_DEBUG_SYNC="${CUDA_DEBUG_SYNC:-false}"
if [[ "$CUDA_DEBUG_SYNC" == "true" ]]; then
    export CUDA_LAUNCH_BLOCKING=1
fi

# =============================================================================
# LOGGING
# =============================================================================

mkdir -p "$WORK_ROOT/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-$WORK_ROOT/logs/render_${SCENE_NAME}_${CHECKPOINT_STEP}_${TIMESTAMP}.log}"

if [[ "$USE_PTY_LOGGING" == "true" \
   && -z "${_3DGRUT_RENDER_PTY_ACTIVE:-}" \
   && -t 1 \
   && -x "$(command -v script 2>/dev/null || true)" ]]; then
    export _3DGRUT_RENDER_PTY_ACTIVE=1
    export LOG_FILE
    printf -v _3DGRUT_RENDER_REEXEC '%q ' bash "$0" "$@"
    exec script -q -f -e -c "$_3DGRUT_RENDER_REEXEC" "$LOG_FILE"
fi

if [[ -z "${_3DGRUT_RENDER_PTY_ACTIVE:-}" ]]; then
    exec > >(tee -a "$LOG_FILE") 2>&1
fi

# =============================================================================
# HELPERS
# =============================================================================

die() {
    echo "ERROR: $*" >&2
    exit 1
}

have_command() {
    command -v "$1" >/dev/null 2>&1
}

auto_find_dataset() {
    if [[ -n "$DATASET_SOURCE" && -d "$DATASET_SOURCE" ]]; then
        return 0
    fi

    local root=""
    local candidate=""
    for root in \
        "$PUBLIC_SET_ROOT" \
        "$AI_TUYEN_ROOT/dataset/phase1/private_set1" \
        "$AI_TUYEN_ROOT/dataset/phase1"
    do
        [[ -d "$root" ]] || continue

        if [[ -d "$root/$SCENE_NAME" ]]; then
            DATASET_SOURCE="$root/$SCENE_NAME"
            return 0
        fi

        candidate="$(
            find "$root" \
                -mindepth 1 \
                -maxdepth 1 \
                -type d \
                -iname "$SCENE_NAME" \
                -print \
                -quit 2>/dev/null || true
        )"
        if [[ -n "$candidate" ]]; then
            DATASET_SOURCE="$candidate"
            return 0
        fi
    done
}

prepare_dataset_view() {
    auto_find_dataset
    [[ -n "$DATASET_SOURCE" ]] || die \
        "Scene '$SCENE_NAME' was not found under $PUBLIC_SET_ROOT"

    DATASET_SOURCE="$(readlink -f "$DATASET_SOURCE")"
    local ready_root=""

    if [[ -d "$DATASET_SOURCE/images" && -d "$DATASET_SOURCE/sparse/0" ]]; then
        ready_root="$DATASET_SOURCE"
    elif [[ -d "$DATASET_SOURCE/train/images" \
         && -d "$DATASET_SOURCE/train/sparse/0" ]]; then
        local view_root="$WORK_ROOT/dataset_views/$SCENE_NAME"
        rm -rf "$view_root"
        mkdir -p "$view_root"
        ln -s "$DATASET_SOURCE/train/images" "$view_root/images"
        ln -s "$DATASET_SOURCE/train/sparse" "$view_root/sparse"
        if [[ -d "$DATASET_SOURCE/test" ]]; then
            ln -s "$DATASET_SOURCE/test" "$view_root/test"
        fi
        ready_root="$view_root"
    else
        die \
            "Expected images+sparse/0 or train/images+train/sparse/0 in $DATASET_SOURCE"
    fi

    [[ -f "$ready_root/sparse/0/cameras.bin" ]] || die \
        "Missing cameras.bin: $ready_root/sparse/0/cameras.bin"
    [[ -f "$ready_root/test/test_poses.csv" ]] || die \
        "Missing test_poses.csv: $ready_root/test/test_poses.csv"

    mkdir -p "$REPO_DIR/data"
    local target="$REPO_DIR/data/$SCENE_NAME"

    if [[ -L "$target" ]]; then
        rm -f "$target"
    elif [[ -e "$target" \
         && "$(readlink -f "$target")" != "$(readlink -f "$ready_root")" ]]; then
        die "A different real dataset path already exists at: $target"
    fi

    if [[ ! -e "$target" ]]; then
        ln -s "$ready_root" "$target"
    fi

    export DATA_ROOT="$REPO_DIR/data"
    export SCENE_ROOT="$target"
    export TEST_ROOT="$target/test"
    export TEST_CSV_PATH="$target/test/test_poses.csv"
    export GT_DIR="$target/test/images"
    export CAMERAS_BIN_PATH="$target/sparse/0/cameras.bin"

    echo "[Data] Scene         : $SCENE_NAME"
    echo "[Data] Source        : $DATASET_SOURCE"
    echo "[Data] Render view   : $ready_root"
    echo "[Data] Repository    : $target"
    echo "[Data] Test CSV      : $TEST_CSV_PATH"
    if [[ -d "$GT_DIR" ]]; then
        echo "[Data] Test GT       : $GT_DIR"
    else
        echo "[Data] Test GT       : not present; render-only mode available"
    fi
}

check_runtime() {
    [[ -d "$REPO_DIR/.git" ]] || die \
        "3DGRUT repository not found at $REPO_DIR. Run setup_and_train_3dgrut.sh first."
    [[ -x "$VENV_DIR/bin/python" ]] || die \
        "3DGRUT .venv not found at $VENV_DIR. Run training setup first."
    [[ -f "$REPO_DIR/render.py" ]] || die \
        "Invalid 3DGRUT repository: $REPO_DIR/render.py is missing."

    # Restore PATH and compiler variables persisted by install_env_uv.sh.
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    export UV_PROJECT_ENVIRONMENT="$VENV_DIR"
    export UV_PYTHON="$VENV_DIR/bin/python"
    export PATH="$VENV_DIR/bin:$PATH"
    hash -r

    echo "[Runtime] Venv          : $VENV_DIR"
    echo "[Runtime] Python        : $(command -v python)"
    echo "[Runtime] PATH head     : ${PATH%%:*}"
    echo "[Runtime] Slang target  : $SLANGC_BIN"

    command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is unavailable."
    nvidia-smi -i 1 --query-gpu=index --format=csv,noheader,nounits \
        >/dev/null 2>&1 || die "Physical GPU 1 is unavailable."

    echo "[GPU] Hard-locked physical GPU: 1"
    nvidia-smi -i 1 \
        --query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.used,memory.free \
        --format=csv,noheader

    env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
        "$VENV_DIR/bin/python" - <<'PYVERIFY'
import os
import torch

print("[Python] Version:", os.sys.version.split()[0])
print("[PyTorch] Version:", torch.__version__)
print("[PyTorch] CUDA:", torch.version.cuda)
print("[PyTorch] CUDA available:", torch.cuda.is_available())
print("[PyTorch] CUDA_DEVICE_ORDER:", os.environ.get("CUDA_DEVICE_ORDER"))
print("[PyTorch] CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
    raise SystemExit("GPU lock failed: CUDA_VISIBLE_DEVICES must be 1.")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access physical GPU 1.")
if torch.cuda.device_count() != 1:
    raise SystemExit(
        f"Expected exactly one visible GPU, got {torch.cuda.device_count()}."
    )

torch.cuda.set_device(0)
print("[PyTorch] Logical device:", torch.cuda.current_device())
print("[PyTorch] Logical cuda:0:", torch.cuda.get_device_name(0))
PYVERIFY
}

ensure_slangc() {
    local actual_version=""

    [[ -f "$SLANGC_BIN" ]] || die \
        "slangc file is missing: $SLANGC_BIN
Install only this component with:
  cd $REPO_DIR
  source $VENV_DIR/bin/activate
  export UV_PROJECT_ENVIRONMENT=$VENV_DIR
  bash scripts/install_slangc.sh"

    [[ -x "$SLANGC_BIN" ]] || die \
        "slangc exists but is not executable: $SLANGC_BIN
Run:
  chmod +x $SLANGC_BIN"

    # setup_3dgut.py invokes the literal command `slangc`.
    export PATH="$(dirname "$SLANGC_BIN"):$PATH"
    export LD_LIBRARY_PATH="$VENV_DIR/lib:$VENV_DIR/lib64:${LD_LIBRARY_PATH:-}"
    hash -r

    local resolved_slangc
    resolved_slangc="$(command -v slangc 2>/dev/null || true)"
    [[ -n "$resolved_slangc" ]] || die \
        "slangc exists but cannot be resolved through PATH."

    [[ "$(readlink -f "$resolved_slangc")" == "$(readlink -f "$SLANGC_BIN")" ]] || die \
        "PATH resolves a different slangc:
  expected: $SLANGC_BIN
  resolved: $resolved_slangc"

    if ! actual_version="$("$SLANGC_BIN" -version 2>&1 | head -n 1 | tr -d '\r')"; then
        die \
            "slangc exists but cannot run: $SLANGC_BIN
Check shared libraries with:
  ldd $SLANGC_BIN | grep 'not found'"
    fi

    echo "[Slang] Executable     : $SLANGC_BIN"
    echo "[Slang] PATH resolved  : $resolved_slangc"
    echo "[Slang] Version        : $actual_version"
    echo "[Slang] Expected       : $SLANGC_EXPECTED_VERSION"

    if [[ "$actual_version" != "$SLANGC_EXPECTED_VERSION" ]]; then
        echo "WARNING: slangc version differs from the repository-pinned version." >&2
        echo "WARNING: render will continue because the compiler is runnable." >&2
    fi

    SLANGC_BIN="$SLANGC_BIN" "$VENV_DIR/bin/python" - <<'PYSLANG'
import os
import pathlib
import shutil
import subprocess

expected = pathlib.Path(os.environ["SLANGC_BIN"]).resolve()
resolved_raw = shutil.which("slangc")
if not resolved_raw:
    raise SystemExit("Python subprocess environment cannot resolve slangc.")

resolved = pathlib.Path(resolved_raw).resolve()
if resolved != expected:
    raise SystemExit(
        f"Python resolved a different slangc: expected={expected}, resolved={resolved}"
    )

version = subprocess.check_output(
    ["slangc", "-version"],
    text=True,
    stderr=subprocess.STDOUT,
).strip()

print("[Slang] Python lookup   :", resolved)
print("[Slang] Subprocess test :", version)
PYSLANG
}

patch_tcnn_jit_control() {
    local feature_file="$REPO_DIR/threedgrut/model/feature_decoder.py"
    [[ -f "$feature_file" ]] || die "Feature decoder not found: $feature_file"

    "$VENV_DIR/bin/python" - "$feature_file" <<'PYPATCH'
from pathlib import Path
import py_compile
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
marker = "# BEGIN TCNN JIT CONTROL PATCH"

if marker not in text:
    if "import tinycudann as tcnn\nimport torch\n" in text:
        text = text.replace(
            "import tinycudann as tcnn\nimport torch\n",
            "import os\n\nimport tinycudann as tcnn\nimport torch\n",
            1,
        )

    old = """        if hasattr(tcnn, "supports_jit_fusion"):
            self.network.jit_fusion = tcnn.supports_jit_fusion()
"""
    new = """        # BEGIN TCNN JIT CONTROL PATCH
        if hasattr(tcnn, "supports_jit_fusion"):
            raw_jit = os.environ.get("TCNN_JIT_FUSION", "0").strip().lower()
            requested_jit = raw_jit in {"1", "true", "yes", "y", "on"}
            self.network.jit_fusion = bool(
                requested_jit and tcnn.supports_jit_fusion()
            )
        # END TCNN JIT CONTROL PATCH
"""
    if old not in text:
        raise RuntimeError(
            "Could not patch tiny-cuda-nn JIT control; source anchor changed."
        )
    backup = path.with_name(path.name + ".pre_render_tcnn_jit.bak")
    if not backup.exists():
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    text = text.replace(old, new, 1)
    path.write_text(text, encoding="utf-8")
    py_compile.compile(str(path), doraise=True)
    print(f"[Patch] tiny-cuda-nn JIT control applied: {path}")
else:
    print("[Patch] tiny-cuda-nn JIT control already present.")
PYPATCH
}

write_embedded_renderer() {
    RENDER_LAUNCHER="$REPO_DIR/render_3dgrut_v2_embedded.py"
    cat > "$RENDER_LAUNCHER" <<'__RENDER_3DGRUT_V2_PY__'
#!/usr/bin/env python3
"""
Render AI Race test_poses.csv from a native 3DGRUT v2 3DGUT-MCMC-NHT
checkpoint and evaluate with the AI Race score.

Features:
- Auto-discovers the latest run manifest produced by train_hcm0204_3dgrut_v2.py.
- Supports explicit intermediate checkpoints (30k, 32.5k, 35k).
- Builds native OpenCV/SIMPLE_RADIAL camera rays from COLMAP cameras.bin.
- Loads NHT FeatureDecoder and applies EMA weights.
- Loads PPISP automatically when present in the checkpoint.
- Writes checkpoint-specific render folders to prevent stale-image reuse.
- Computes PSNR, SSIM, LPIPS-VGG/Alex, MSE, MAE and the official score.

Place this file in the root of nv-tlabs/3dgrut next to render.py.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any, NoReturn

import ncore.sensors
import numpy as np
import torch
from ncore.data import OpenCVPinholeCameraModelParameters, ShutterType
from PIL import Image
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm

from threedgrut.datasets.protocols import Batch
from threedgrut.datasets.utils import create_pixel_coords, read_colmap_intrinsics_binary
from threedgrut.model.feature_decoder import FeatureDecoder
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.utils.render import apply_background, apply_feature_decoder, apply_post_processing


# =============================================================================
# USER CONFIGURATION
# =============================================================================

def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_optional_int(name: str, default: int | None = None) -> int | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


SCENE_NAME = os.environ.get("SCENE_NAME", "HCM0181")
SCRIPT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("DATA_ROOT", str(SCRIPT_ROOT / "data"))).expanduser().resolve()
ORIGINAL_SCENE_ROOT = Path(
    os.environ.get("SCENE_ROOT", str(DATA_ROOT / SCENE_NAME))
).expanduser().resolve()
TEST_ROOT = Path(
    os.environ.get("TEST_ROOT", str(ORIGINAL_SCENE_ROOT / "test"))
).expanduser().resolve()
TEST_CSV_PATH = Path(
    os.environ.get("TEST_CSV_PATH", str(TEST_ROOT / "test_poses.csv"))
).expanduser().resolve()
GT_DIR = Path(
    os.environ.get("GT_DIR", str(TEST_ROOT / "images"))
).expanduser().resolve()
CAMERAS_BIN_PATH = Path(
    os.environ.get(
        "CAMERAS_BIN_PATH",
        str(ORIGINAL_SCENE_ROOT / "sparse" / "0" / "cameras.bin"),
    )
).expanduser().resolve()

APPEARANCE_MODE = os.environ.get("APPEARANCE_MODE", "native_distortion")
MAX_STEPS = int(os.environ.get("MAX_STEPS", "35000"))
CAP_MAX = int(os.environ.get("CAP_MAX", "2000000"))
NHT_FEATURE_DIM = int(os.environ.get("NHT_FEATURE_DIM", "48"))
RUN_NAME = os.environ.get(
    "RUN_NAME",
    (
        f"{SCENE_NAME}_3dgrut_v2_3dgut_mcmc_nht_"
        f"{APPEARANCE_MODE}_{MAX_STEPS // 1000}k_"
        f"{CAP_MAX // 1_000_000}m_fd{NHT_FEATURE_DIM}"
    ),
)
EXPERIMENT_DIR = Path(
    os.environ.get("EXPERIMENT_DIR", str(SCRIPT_ROOT / "runs" / RUN_NAME))
).expanduser().resolve()

# CHECKPOINT_STEP accepts: last/latest/none or an exact integer step.
_checkpoint_step_raw = os.environ.get("CHECKPOINT_STEP", "last").strip().lower()
CHECKPOINT_STEP: int | None = (
    None
    if _checkpoint_step_raw in {"", "last", "latest", "none"}
    else int(_checkpoint_step_raw)
)
_checkpoint_path_raw = os.environ.get("CHECKPOINT_PATH", "").strip()
CHECKPOINT_PATH: Path | None = (
    Path(_checkpoint_path_raw).expanduser().resolve()
    if _checkpoint_path_raw
    else None
)

# CUDA_VISIBLE_DEVICES selects the physical GPU in the shell. Inside this
# process the selected device is logical cuda:0.
GPU_ID = int(os.environ.get("RENDER_LOGICAL_GPU_ID", "0"))
CAMERA_ID = env_optional_int("CAMERA_ID")
USE_NATIVE_DISTORTION = env_bool("USE_NATIVE_DISTORTION", True)
USE_FEATURE_DECODER_EMA = env_bool("USE_FEATURE_DECODER_EMA", True)

LPIPS_NET = os.environ.get("LPIPS_NET", "vgg").strip().lower()
PSNR_MAX = float(os.environ.get("PSNR_MAX", "40.0"))
BACKGROUND_OVERRIDE: tuple[float, float, float] | None = None

OVERWRITE = env_bool("OVERWRITE_RENDER", False)
SAVE_ALPHA = env_bool("SAVE_ALPHA", False)
MAX_IMAGES = env_optional_int("MAX_IMAGES")
_force_extension_raw = os.environ.get("FORCE_OUTPUT_EXTENSION", "").strip()
FORCE_OUTPUT_EXTENSION: str | None = _force_extension_raw or None
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "95"))
PNG_COMPRESS_LEVEL = int(os.environ.get("PNG_COMPRESS_LEVEL", "3"))
RESIZE_PREDICTION_TO_GT = env_bool("RESIZE_PREDICTION_TO_GT", False)
CONTINUE_ON_EVAL_ERROR = env_bool("CONTINUE_ON_EVAL_ERROR", True)

_eval_raw = os.environ.get("ENABLE_EVALUATION", "auto").strip().lower()
ENABLE_EVALUATION = (
    GT_DIR.is_dir()
    if _eval_raw == "auto"
    else _eval_raw in {"1", "true", "yes", "y", "on"}
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# =============================================================================
# HELPERS
# =============================================================================


def fail(message: str) -> NoReturn:
    raise RuntimeError(message)


def locate_3dgrut_root() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in (here, here.parent):
        if (
            (candidate / "render.py").is_file()
            and (candidate / "threedgrut" / "render.py").is_file()
            and (candidate / "threedgrut" / "model" / "model.py").is_file()
        ):
            return candidate
    fail("Place this renderer in the root of the 3DGRUT repository.")


def checkpoint_step_from_path(path: Path) -> int:
    match = re.search(r"ckpt_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def checkpoint_run_dir(checkpoint: Path) -> Path:
    checkpoint = checkpoint.resolve()
    if checkpoint.parent.name.startswith("ours_"):
        return checkpoint.parent.parent
    return checkpoint.parent


def checkpoint_candidates(experiment_dir: Path) -> list[Path]:
    if not experiment_dir.is_dir():
        return []
    candidates = {
        *experiment_dir.rglob("ckpt_last.pt"),
        *experiment_dir.rglob("ckpt_*.pt"),
    }
    return sorted(
        (
            path.resolve()
            for path in candidates
            if path.is_file() and path.stat().st_size > 0
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def resolve_checkpoint() -> tuple[Path, Path]:
    if CHECKPOINT_PATH is not None:
        if not CHECKPOINT_PATH.is_file():
            fail(f"Explicit checkpoint does not exist: {CHECKPOINT_PATH}")
        return CHECKPOINT_PATH, checkpoint_run_dir(CHECKPOINT_PATH)

    candidates = checkpoint_candidates(EXPERIMENT_DIR)
    if not candidates:
        fail(
            f"No checkpoint was found under: {EXPERIMENT_DIR}\n"
            "Finish or interrupt training after enabling save-on-Ctrl+C, "
            "or provide CHECKPOINT_PATH explicitly."
        )

    if CHECKPOINT_STEP is None:
        last_candidates = [path for path in candidates if path.name == "ckpt_last.pt"]
        checkpoint = last_candidates[0] if last_candidates else candidates[0]
        return checkpoint, checkpoint_run_dir(checkpoint)

    exact_name = f"ckpt_{CHECKPOINT_STEP}.pt"
    exact_candidates = [path for path in candidates if path.name == exact_name]
    if not exact_candidates:
        available = sorted(
            {
                checkpoint_step_from_path(path)
                for path in candidates
                if checkpoint_step_from_path(path) >= 0
            }
        )
        fail(
            f"Checkpoint step {CHECKPOINT_STEP} was not found under: "
            f"{EXPERIMENT_DIR}\nAvailable exact steps: {available}; "
            f"last checkpoints: {sum(path.name == 'ckpt_last.pt' for path in candidates)}"
        )

    checkpoint = exact_candidates[0]
    return checkpoint, checkpoint_run_dir(checkpoint)


def normalize_header(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "image_name": ("image_name", "name", "filename", "file_name", "image"),
    "qw": ("qw", "q_w"), "qx": ("qx", "q_x"), "qy": ("qy", "q_y"), "qz": ("qz", "q_z"),
    "tx": ("tx", "t_x"), "ty": ("ty", "t_y"), "tz": ("tz", "t_z"),
    "fx": ("fx", "f_x"), "fy": ("fy", "f_y"),
    "cx": ("cx", "c_x"), "cy": ("cy", "c_y"),
    "width": ("width", "w", "image_width"),
    "height": ("height", "h", "image_height"),
}


def resolve_column(fieldnames: list[str], logical_name: str) -> str:
    mapping = {normalize_header(field): field for field in fieldnames}
    for alias in COLUMN_ALIASES[logical_name]:
        if alias in mapping:
            return mapping[alias]
    fail(f"Missing CSV column {logical_name!r}; available columns: {fieldnames}")


def parse_float(row: dict[str, str], column: str, row_number: int) -> float:
    try:
        value = float(row.get(column, ""))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Row {row_number}: invalid value in {column!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"Row {row_number}: non-finite value in {column!r}")
    return value


def parse_int(row: dict[str, str], column: str, row_number: int) -> int:
    value = parse_float(row, column, row_number)
    rounded = int(round(value))
    if rounded <= 0 or abs(value - rounded) > 1e-6:
        raise ValueError(f"Row {row_number}: {column!r} must be a positive integer")
    return rounded


def read_test_poses(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        fail(f"test_poses.csv does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            fail(f"CSV has no header: {path}")
        columns = {key: resolve_column(list(reader.fieldnames), key) for key in COLUMN_ALIASES}
        records: list[dict[str, Any]] = []
        for row_number, row in enumerate(reader, start=2):
            image_name = str(row.get(columns["image_name"], "")).strip()
            if not image_name:
                raise ValueError(f"Row {row_number}: empty image_name")
            record: dict[str, Any] = {
                "image_name": image_name,
                "width": parse_int(row, columns["width"], row_number),
                "height": parse_int(row, columns["height"], row_number),
            }
            for key in ("qw", "qx", "qy", "qz", "tx", "ty", "tz", "fx", "fy", "cx", "cy"):
                record[key] = parse_float(row, columns[key], row_number)
            records.append(record)
    if MAX_IMAGES is not None:
        records = records[:MAX_IMAGES]
    if not records:
        fail(f"No test poses found in: {path}")
    return records


def quaternion_to_rotation(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    q = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        fail("Encountered a zero-length quaternion.")
    qw, qx, qy, qz = q / norm
    return np.asarray([
        [1 - 2 * (qy*qy + qz*qz), 2 * (qx*qy - qz*qw), 2 * (qx*qz + qy*qw)],
        [2 * (qx*qy + qz*qw), 1 - 2 * (qx*qx + qz*qz), 2 * (qy*qz - qx*qw)],
        [2 * (qx*qz - qy*qw), 2 * (qy*qz + qx*qw), 1 - 2 * (qx*qx + qy*qy)],
    ], dtype=np.float32)


def build_c2w(record: dict[str, Any]) -> np.ndarray:
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = quaternion_to_rotation(record["qw"], record["qx"], record["qy"], record["qz"])
    w2c[:3, 3] = np.asarray([record["tx"], record["ty"], record["tz"]], dtype=np.float32)
    return np.linalg.inv(w2c).astype(np.float32)


# =============================================================================
# NATIVE CAMERA MODEL
# =============================================================================


def select_colmap_camera() -> Any:
    cameras = read_colmap_intrinsics_binary(str(CAMERAS_BIN_PATH))
    if CAMERA_ID is not None:
        if CAMERA_ID not in cameras:
            fail(f"CAMERA_ID={CAMERA_ID} not found; available IDs: {sorted(cameras)}")
        return cameras[CAMERA_ID]
    if len(cameras) != 1:
        fail(f"Expected one COLMAP camera, found {len(cameras)}. Set CAMERA_ID explicitly.")
    return next(iter(cameras.values()))


def distortion_from_colmap(camera: Any) -> dict[str, np.ndarray | str]:
    model = str(camera.model)
    params = np.asarray(camera.params, dtype=np.float32)
    radial = np.zeros(6, dtype=np.float32)
    tangential = np.zeros(2, dtype=np.float32)
    thin_prism = np.zeros(4, dtype=np.float32)

    if not USE_NATIVE_DISTORTION or model in {"SIMPLE_PINHOLE", "PINHOLE"}:
        pass
    elif model == "SIMPLE_RADIAL":
        radial[0] = params[3]
    elif model == "RADIAL":
        radial[:2] = params[3:5]
    elif model == "OPENCV":
        radial[:2] = params[4:6]
        tangential[:] = params[6:8]
    elif model == "FULL_OPENCV":
        radial[:] = params[[4, 5, 8, 9, 10, 11]]
        tangential[:] = params[6:8]
    else:
        fail(
            f"Custom CSV renderer currently supports pinhole/OpenCV distortion, got {model}. "
            "Use the official dataset renderer for fisheye cameras."
        )
    return {
        "model": model,
        "radial": radial,
        "tangential": tangential,
        "thin_prism": thin_prism,
    }


def camera_cache_key(record: dict[str, Any], distortion: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record["width"], record["height"],
        round(record["fx"], 10), round(record["fy"], 10),
        round(record["cx"], 10), round(record["cy"], 10),
        tuple(np.asarray(distortion["radial"]).tolist()),
        tuple(np.asarray(distortion["tangential"]).tolist()),
        tuple(np.asarray(distortion["thin_prism"]).tolist()),
    )


def build_native_camera_tensors(
    record: dict[str, Any],
    distortion: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], torch.Tensor]:
    width = int(record["width"])
    height = int(record["height"])
    params = OpenCVPinholeCameraModelParameters(
        resolution=np.array([width, height], dtype=np.uint64),
        shutter_type=ShutterType.GLOBAL,
        principal_point=np.array([record["cx"], record["cy"]], dtype=np.float32),
        focal_length=np.array([record["fx"], record["fy"]], dtype=np.float32),
        radial_coeffs=np.asarray(distortion["radial"], dtype=np.float32),
        tangential_coeffs=np.asarray(distortion["tangential"], dtype=np.float32),
        thin_prism_coeffs=np.asarray(distortion["thin_prism"], dtype=np.float32),
    )
    camera_model = ncore.sensors.CameraModel.from_parameters(
        params, device="cpu", dtype=torch.float32
    )
    u = np.tile(np.arange(width), height)
    v = np.arange(height).repeat(width)
    int_pixels = torch.tensor(np.stack([u, v], axis=1), dtype=torch.int32)
    image_points = camera_model.pixels_to_image_points(int_pixels)
    rays_dir = camera_model.image_points_to_camera_rays(image_points)
    rays_ori = torch.zeros_like(rays_dir)
    rays_ori = rays_ori.to(torch.float32).reshape(1, height, width, 3).to(device)
    rays_dir = rays_dir.to(torch.float32).reshape(1, height, width, 3).to(device)
    pixels = create_pixel_coords(width, height, device=device)
    return rays_ori, rays_dir, params.to_dict(), pixels


def make_batch(
    record: dict[str, Any],
    distortion: dict[str, Any],
    cache: dict[tuple[Any, ...], tuple[torch.Tensor, torch.Tensor, dict[str, Any], torch.Tensor]],
    device: torch.device,
) -> Batch:
    key = camera_cache_key(record, distortion)
    if key not in cache:
        cache[key] = build_native_camera_tensors(record, distortion, device)
    rays_ori, rays_dir, params_dict, pixels = cache[key]
    c2w = torch.from_numpy(build_c2w(record)).to(device=device, dtype=torch.float32).unsqueeze(0)
    return Batch(
        rays_ori=rays_ori,
        rays_dir=rays_dir,
        T_to_world=c2w,
        intrinsics_OpenCVPinholeCameraModelParameters=params_dict,
        camera_idx=0,
        frame_idx=-1,
        pixel_coords=pixels,
    )


# =============================================================================
# CHECKPOINT MODEL LOADING
# =============================================================================


def load_post_processing(checkpoint: dict[str, Any], conf: Any, device: torch.device) -> Any | None:
    if "post_processing" not in checkpoint:
        return None
    method = conf.post_processing.method
    if method == "linear-to-srgb":
        from threedgrut.utils.post_processing_linear_to_srgb import LinearToSrgbPostProcessing
        module = LinearToSrgbPostProcessing()
        module.load_state_dict(checkpoint["post_processing"]["module"])
        return module.to(device).eval()
    if method == "ppisp":
        from ppisp import PPISP, PPISPConfig
        use_controller = conf.post_processing.get("use_controller", True)
        distill_steps = conf.post_processing.get("n_distillation_steps", 5000)
        if use_controller and distill_steps > 0:
            activation_ratio = (conf.n_iterations - distill_steps) / conf.n_iterations
            controller_distillation = True
        elif use_controller:
            activation_ratio = 0.8
            controller_distillation = False
        else:
            activation_ratio = 0.0
            controller_distillation = False
        pp_config = PPISPConfig(
            use_controller=use_controller,
            controller_distillation=controller_distillation,
            controller_activation_ratio=activation_ratio,
        )
        return PPISP.from_state_dict(
            checkpoint["post_processing"]["module"], config=pp_config
        ).to(device).eval()
    return None


def load_feature_decoder(
    checkpoint: dict[str, Any],
    conf: Any,
    model: MixtureOfGaussians,
    device: torch.device,
) -> tuple[FeatureDecoder | None, str]:
    if "feature_decoder" not in checkpoint:
        return None, "none"
    dec = conf.model.nht_decoder
    decoder = FeatureDecoder(
        ray_feature_dim=model.ray_feature_dim,
        hidden_dim=dec.hidden_dim,
        num_layers=getattr(dec, "num_layers", 4),
        dir_encoding=getattr(dec, "dir_encoding", "SphericalHarmonics"),
        dir_encoding_degree=getattr(dec, "dir_encoding_degree", 3),
        sh_scale=getattr(dec, "sh_scale", 1.0),
        output_activation=getattr(dec, "output_activation", "Sigmoid"),
        ema_decay=getattr(dec, "ema_decay", 0.0),
        ema_start_step=getattr(dec, "ema_start_step", 0),
        unpremultiply_alpha=getattr(dec, "unpremultiply_alpha", False),
    ).to(device)
    decoder.load_state_dict(checkpoint["feature_decoder"]["module"])
    state_name = "module"
    ema_state = checkpoint["feature_decoder"].get("ema")
    if USE_FEATURE_DECODER_EMA and ema_state:
        decoder.load_ema_state_dict(ema_state)
        decoder.apply_ema_shadow()
        state_name = "ema"
    decoder.eval()
    return decoder, state_name


def move_model_to_render_device(
    model: MixtureOfGaussians,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    """
    Move only the model tensors to CUDA.

    The checkpoint stays on CPU so optimizer state does not consume render VRAM.
    init_from_checkpoint() assigns checkpoint tensors directly to the model, so
    an explicit model.to(device) is required before the native CUDA renderer.
    """
    model.to(device)
    model.requires_grad_(False)

    tensor_names = [
        "positions",
        "rotation",
        "scale",
        "density",
    ]
    if hasattr(model, "features"):
        tensor_names.append("features")
    if hasattr(model, "features_albedo"):
        tensor_names.append("features_albedo")
    if hasattr(model, "features_specular"):
        tensor_names.append("features_specular")

    report: dict[str, dict[str, Any]] = {}
    for name in tensor_names:
        tensor = getattr(model, name, None)
        if tensor is None:
            continue

        # Keep the storage contiguous before it is passed to the native plugin.
        if isinstance(tensor, torch.nn.Parameter):
            if not tensor.data.is_contiguous():
                tensor.data = tensor.data.contiguous()
        elif isinstance(tensor, torch.Tensor) and not tensor.is_contiguous():
            tensor = tensor.contiguous()
            setattr(model, name, tensor)

        tensor = getattr(model, name)
        report[name] = {
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "is_cuda": bool(tensor.is_cuda),
            "contiguous": bool(tensor.is_contiguous()),
        }

        if not tensor.is_cuda:
            fail(
                f"Model tensor {name!r} is still on {tensor.device}; "
                f"native 3DGUT requires CUDA tensors."
            )
        if tensor.device.index != device.index:
            fail(
                f"Model tensor {name!r} is on {tensor.device}, expected {device}."
            )
        if not tensor.is_contiguous():
            fail(f"Model tensor {name!r} is not contiguous.")

    if not report:
        fail("No Gaussian parameter tensors were found after checkpoint loading.")

    background_parameters = list(model.background.parameters())
    for parameter in background_parameters:
        if not parameter.is_cuda or parameter.device.index != device.index:
            fail(
                f"Background parameter is on {parameter.device}, expected {device}."
            )

    print("[Model] CUDA tensor verification:")
    for name, info in report.items():
        print(
            f"[Model]   {name:<20} "
            f"shape={info['shape']}, dtype={info['dtype']}, "
            f"device={info['device']}, contiguous={info['contiguous']}"
        )
    return report


def cuda_memory_line(device: torch.device, label: str) -> None:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    print(
        f"[CUDA] {label}: "
        f"free={free_bytes / 1024**3:.2f} GiB, "
        f"total={total_bytes / 1024**3:.2f} GiB, "
        f"allocated={allocated / 1024**3:.2f} GiB, "
        f"reserved={reserved / 1024**3:.2f} GiB"
    )


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[MixtureOfGaussians, FeatureDecoder | None, Any | None, Any, dict[str, Any]]:
    # Keep the full checkpoint and optimizer state on CPU. Only model tensors
    # are moved to CUDA after init_from_checkpoint().
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if "config" not in checkpoint:
        fail(f"Checkpoint has no config: {checkpoint_path}")

    conf = checkpoint["config"]
    checkpoint_backend = str(conf.render.method)
    if checkpoint_backend != "3dgut":
        fail(
            f"Expected a native 3DGUT checkpoint, got "
            f"render.method={checkpoint_backend}"
        )

    print("[Model] Checkpoint backend : native 3DGUT")
    print("[Model] Checkpoint load    : CPU")
    print("[Model] Render device      :", device)
    print("[Model] Extension cache    :", os.environ.get("TORCH_EXTENSIONS_DIR", "default"))

    cuda_memory_line(device, "before model construction")

    # Construct the exact backend/config used during training.
    model = MixtureOfGaussians(conf)
    model.init_from_checkpoint(checkpoint, setup_optimizer=False)

    # Critical fix: init_from_checkpoint assigns checkpoint tensors directly.
    # Since the checkpoint was loaded on CPU, explicitly move all registered
    # Gaussian/NHT parameters and the background to the render CUDA device.
    tensor_report = move_model_to_render_device(model, device)
    model.eval()

    cuda_memory_line(device, "after model tensors moved")

    # Native 3DGUT currently has no OptiX BVH, but call the common API to match
    # the official renderer and future repository changes.
    model.build_acc()
    torch.cuda.synchronize(device)

    cuda_memory_line(device, "after build_acc")

    decoder, decoder_state = load_feature_decoder(
        checkpoint,
        conf,
        model,
        device,
    )
    post_processing = load_post_processing(
        checkpoint,
        conf,
        device,
    )

    # Release CPU-only optimizer/checkpoint tensors before rendering.
    checkpoint_gaussians = int(checkpoint["positions"].shape[0])
    global_step = int(
        checkpoint.get(
            "global_step",
            checkpoint_step_from_path(checkpoint_path),
        )
    )
    feature_type = str(checkpoint.get("feature_type", "unknown"))
    particle_feature_dim = int(
        checkpoint.get("particle_feature_dim", -1)
    )
    ray_feature_dim = int(
        checkpoint.get("ray_feature_dim", -1)
    )
    del checkpoint

    meta = {
        "global_step": global_step,
        "num_gaussians": int(model.num_gaussians),
        "checkpoint_num_gaussians": checkpoint_gaussians,
        "feature_type": feature_type,
        "particle_feature_dim": particle_feature_dim,
        "ray_feature_dim": ray_feature_dim,
        "decoder_state": decoder_state,
        "post_processing": str(conf.post_processing.method),
        "renderer": "native_3dgut",
        "tensor_devices": tensor_report,
    }
    return model, decoder, post_processing, conf, meta


# =============================================================================
# IMAGE I/O AND METRICS
# =============================================================================


def output_path_for(output_dir: Path, image_name: str) -> Path:
    relative = Path(image_name)
    if relative.is_absolute() or ".." in relative.parts:
        relative = Path(relative.name)
    if FORCE_OUTPUT_EXTENSION:
        extension = FORCE_OUTPUT_EXTENSION if FORCE_OUTPUT_EXTENSION.startswith(".") else "." + FORCE_OUTPUT_EXTENSION
        relative = relative.with_suffix(extension)
    elif relative.suffix == "":
        relative = relative.with_suffix(".png")
    return output_dir / relative


def save_rgb(rgb: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(rgb, mode="RGB")
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        image.save(path, format="JPEG", quality=JPEG_QUALITY, subsampling=0, optimize=False)
        return path
    if suffix == ".png":
        image.save(path, format="PNG", compress_level=PNG_COMPRESS_LEVEL)
        return path
    fallback = path.with_suffix(".png")
    image.save(fallback, format="PNG", compress_level=PNG_COMPRESS_LEVEL)
    return fallback


def find_gt_path(image_name: str) -> Path | None:
    direct = GT_DIR / image_name
    if direct.is_file():
        return direct
    stem = Path(image_name).stem.casefold()
    matches = [
        p for p in GT_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and p.stem.casefold() == stem
    ]
    return sorted(matches)[0] if matches else None


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def evaluate_pair(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    device: torch.device,
    ssim_metric: StructuralSimilarityIndexMeasure,
    lpips_metric: LearnedPerceptualImagePatchSimilarity,
) -> dict[str, float]:
    pred = prediction.astype(np.float32) / 255.0
    gt = ground_truth.astype(np.float32) / 255.0
    diff = pred - gt
    mse = float(np.mean(diff * diff))
    mae = float(np.mean(np.abs(diff)))
    psnr = float("inf") if mse <= 0 else -10.0 * math.log10(mse)
    psnr_norm = min(max(psnr / PSNR_MAX, 0.0), 1.0)
    pred_t = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).to(device)
    gt_t = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).to(device)
    ssim_metric.reset()
    lpips_metric.reset()
    with torch.inference_mode():
        ssim = float(ssim_metric(pred_t, gt_t).item())
        lpips = float(lpips_metric(pred_t, gt_t).item())
    score = 0.4 * (1.0 - lpips) + 0.3 * ssim + 0.3 * psnr_norm
    return {
        "psnr": psnr, "psnr_norm": psnr_norm, "ssim": ssim,
        "lpips": lpips, "score": score, "mse": mse, "mae": mae,
    }


def finite_mean(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return statistics.fmean(finite) if finite else float("nan")


def write_metrics(
    rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    metrics_dir: Path,
    checkpoint_path: Path,
    meta: dict[str, Any],
    distortion: dict[str, Any],
) -> None:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_name", "status", "message", "render_seconds",
        "psnr", "psnr_norm", "ssim", "lpips", "score", "mse", "mae",
    ]
    with (metrics_dir / "metrics_per_image.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    ok = [row for row in rows if row.get("status") == "ok"]
    numeric = {
        key: finite_mean([float(row[key]) for row in ok])
        for key in ("psnr", "psnr_norm", "ssim", "lpips", "score", "mse", "mae")
    }
    missing = sum(row.get("status") == "missing_gt" for row in rows)
    mismatch = sum(row.get("status") == "size_mismatch" for row in rows)
    errors = sum(row.get("status") == "error" for row in rows)
    score_from_means = (
        0.4 * (1.0 - numeric["lpips"])
        + 0.3 * numeric["ssim"]
        + 0.3 * min(max(numeric["psnr"] / PSNR_MAX, 0.0), 1.0)
    )
    summary = {
        "scene": SCENE_NAME,
        "run": RUN_NAME,
        "checkpoint": str(checkpoint_path),
        "expected_images": len(records),
        "evaluated_images": len(ok),
        "missing_gt": missing,
        "size_mismatch": mismatch,
        "evaluation_errors": errors,
        "metrics": numeric,
        "score_from_mean_metrics": score_from_means,
        "psnr_max": PSNR_MAX,
        "lpips_network": LPIPS_NET,
        "native_distortion": {
            "enabled": USE_NATIVE_DISTORTION,
            "colmap_model": distortion["model"],
            "radial": np.asarray(distortion["radial"]).tolist(),
            "tangential": np.asarray(distortion["tangential"]).tolist(),
        },
        "model": meta,
    }
    (metrics_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    text = f"""==========================================
3DGRUT V2 NHT RENDER + EVALUATION SUMMARY
==========================================
Scene                  : {SCENE_NAME}
Run                    : {RUN_NAME}
Checkpoint             : {checkpoint_path}
Renderer               : native 3DGUT
COLMAP camera model     : {distortion['model']}
Native distortion       : {USE_NATIVE_DISTORTION}
Radial coefficients     : {np.asarray(distortion['radial']).tolist()}
NHT decoder state       : {meta['decoder_state']}
Post processing         : {meta['post_processing']}
Number of Gaussians     : {meta['num_gaussians']}
Expected images         : {len(records)}
Evaluated images        : {len(ok)}
Missing GT              : {missing}
Size mismatch           : {mismatch}
Evaluation errors       : {errors}
------------------------------------------
PSNR mean               : {numeric['psnr']:.8f}
PSNR norm mean          : {numeric['psnr_norm']:.8f}
SSIM mean               : {numeric['ssim']:.8f}
LPIPS mean              : {numeric['lpips']:.8f}
Official score mean     : {numeric['score']:.8f}
Score from mean metrics : {score_from_means:.8f}
MSE mean                : {numeric['mse']:.10f}
MAE mean                : {numeric['mae']:.10f}
------------------------------------------
Formula:
score = 0.4*(1-LPIPS) + 0.3*SSIM + 0.3*clamp(PSNR/PSNR_MAX, 0, 1)
PSNR_MAX                : {PSNR_MAX}
LPIPS network           : {LPIPS_NET}
==========================================
"""
    (metrics_dir / "metrics_summary.txt").write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"Per-image metrics: {metrics_dir / 'metrics_per_image.csv'}")


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    locate_3dgrut_root()
    if not torch.cuda.is_available():
        fail("CUDA is not available.")
    if LPIPS_NET not in {"vgg", "alex", "squeeze"}:
        fail("LPIPS_NET must be 'vgg', 'alex', or 'squeeze'.")
    for path in (TEST_CSV_PATH, CAMERAS_BIN_PATH):
        if not path.exists():
            fail(f"Required test component is missing: {path}")

    if ENABLE_EVALUATION and not GT_DIR.is_dir():
        fail(
            f"ENABLE_EVALUATION is active but the GT directory is missing: {GT_DIR}"
        )

    checkpoint_path, run_dir = resolve_checkpoint()
    device = torch.device(f"cuda:{GPU_ID}")
    torch.cuda.set_device(device)
    records = read_test_poses(TEST_CSV_PATH)
    colmap_camera = select_colmap_camera()
    distortion = distortion_from_colmap(colmap_camera)
    model, decoder, post_processing, conf, meta = load_model(checkpoint_path, device)

    effective_step = int(meta["global_step"])
    result_root = run_dir / f"custom_test_step{effective_step}_native_3dgut"
    output_dir = result_root / "renders"
    metrics_dir = result_root / f"_metrics_lpips_{LPIPS_NET}"
    output_dir.mkdir(parents=True, exist_ok=True)

    ssim_metric = None
    lpips_metric = None
    if ENABLE_EVALUATION:
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        lpips_metric = LearnedPerceptualImagePatchSimilarity(
            net_type=LPIPS_NET, normalize=True
        ).to(device)

    print("========== 3DGRUT V2 CUSTOM TEST RENDER ==========")
    print(f"Checkpoint          : {checkpoint_path}")
    print(f"Run directory       : {run_dir}")
    print(f"Test CSV            : {TEST_CSV_PATH}")
    print(f"Ground Truth        : {GT_DIR}")
    print(f"Test poses          : {len(records)}")
    print(f"Renderer             : native 3DGUT")
    print(f"Gaussians            : {meta['num_gaussians']:,}")
    print(f"Gaussian device      : {model.positions.device}")
    print(f"NHT decoder          : {meta['decoder_state']}")
    print(f"Post processing     : {meta['post_processing']}")
    print(f"COLMAP camera model : {distortion['model']}")
    print(f"Radial coefficients : {np.asarray(distortion['radial']).tolist()}")
    print(f"Physical GPU        : {os.environ.get('PHYSICAL_GPU_ID', 'unknown')}")
    print(f"PyTorch device      : {device}")
    print(f"Checkpoint request  : {_checkpoint_step_raw}")
    print(f"Evaluation enabled  : {ENABLE_EVALUATION}")
    print(f"LPIPS network       : {LPIPS_NET if ENABLE_EVALUATION else 'disabled'}")
    print(f"Output              : {output_dir}")

    camera_cache: dict[tuple[Any, ...], tuple[torch.Tensor, torch.Tensor, dict[str, Any], torch.Tensor]] = {}
    rows: list[dict[str, Any]] = []

    # CUDA/plugin warmup on the first pose.
    first_batch = make_batch(records[0], distortion, camera_cache, device)
    if not model.positions.is_cuda:
        fail(
            f"Gaussian positions unexpectedly moved to {model.positions.device} "
            "before warmup."
        )
    if not first_batch.rays_dir.is_cuda:
        fail(
            f"Camera rays unexpectedly on {first_batch.rays_dir.device}."
        )
    print("[Warmup] Gaussian device :", model.positions.device)
    print("[Warmup] Rays device     :", first_batch.rays_dir.device)
    cuda_memory_line(device, "before warmup")

    with torch.inference_mode():
        for _ in range(2):
            warm = model(first_batch, train=False, frame_id=effective_step)
            if decoder is not None:
                warm = apply_feature_decoder(
                    decoder,
                    warm,
                    first_batch,
                    training=False,
                    center_ray_encoding=bool(getattr(conf.model.nht_decoder, "center_ray_encoding", False)),
                )
            warm = apply_background(model.background, warm, first_batch, training=False)
            if post_processing is not None:
                warm = apply_post_processing(post_processing, warm, first_batch, training=False)
        torch.cuda.synchronize(device)

    for record in tqdm(records, desc="Rendering native 3DGUT test poses"):
        output_path = output_path_for(output_dir, record["image_name"])
        row: dict[str, Any] = {
            "image_name": record["image_name"],
            "status": "not_evaluated",
            "message": "",
            "render_seconds": "",
        }
        try:
            if output_path.exists() and not OVERWRITE:
                actual_output = output_path
                render_seconds = 0.0
            else:
                batch = make_batch(record, distortion, camera_cache, device)
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                with torch.inference_mode():
                    outputs = model(batch, train=False, frame_id=effective_step)
                    if decoder is not None:
                        outputs = apply_feature_decoder(
                            decoder,
                            outputs,
                            batch,
                            training=False,
                            center_ray_encoding=bool(
                                getattr(conf.model.nht_decoder, "center_ray_encoding", False)
                            ),
                        )
                    outputs = apply_background(model.background, outputs, batch, training=False)
                    if post_processing is not None:
                        outputs = apply_post_processing(post_processing, outputs, batch, training=False)
                    rgb = outputs["pred_features"].clamp(0.0, 1.0)
                torch.cuda.synchronize(device)
                render_seconds = time.perf_counter() - started
                rgb_u8 = rgb[0].mul(255.0).round().to(torch.uint8).cpu().numpy()
                actual_output = save_rgb(rgb_u8, output_path)
                if SAVE_ALPHA and "pred_opacity" in outputs:
                    alpha = outputs["pred_opacity"][0]
                    if alpha.ndim == 3:
                        alpha = alpha[..., 0]
                    alpha_u8 = alpha.clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy()
                    Image.fromarray(alpha_u8, mode="L").save(
                        actual_output.with_name(actual_output.stem + "_alpha.png")
                    )

            row["render_seconds"] = f"{render_seconds:.8f}"
            if not ENABLE_EVALUATION:
                row["status"] = "rendered"
                row["message"] = "Evaluation disabled."
                rows.append(row)
                continue

            gt_path = find_gt_path(record["image_name"])
            if gt_path is None:
                row["status"] = "missing_gt"
                row["message"] = "Ground Truth not found."
                rows.append(row)
                continue

            prediction = load_rgb(actual_output)
            ground_truth = load_rgb(gt_path)
            pred_size = (prediction.shape[1], prediction.shape[0])
            gt_size = (ground_truth.shape[1], ground_truth.shape[0])
            if pred_size != gt_size:
                if RESIZE_PREDICTION_TO_GT:
                    prediction = np.asarray(
                        Image.fromarray(prediction).resize(gt_size, Image.Resampling.LANCZOS),
                        dtype=np.uint8,
                    )
                else:
                    row["status"] = "size_mismatch"
                    row["message"] = f"GT={gt_size}, prediction={pred_size}"
                    rows.append(row)
                    continue

            assert ssim_metric is not None
            assert lpips_metric is not None
            values = evaluate_pair(
                prediction, ground_truth, device, ssim_metric, lpips_metric
            )
            row.update({key: f"{value:.10f}" for key, value in values.items()})
            row["status"] = "ok"
        except Exception as exc:
            row["status"] = "error"
            row["message"] = f"{type(exc).__name__}: {exc}"
            if not CONTINUE_ON_EVAL_ERROR:
                raise
        rows.append(row)

    write_metrics(rows, records, metrics_dir, checkpoint_path, meta, distortion)


if __name__ == "__main__":
    main()

__RENDER_3DGRUT_V2_PY__

    chmod +x "$RENDER_LAUNCHER"
    "$VENV_DIR/bin/python" -m py_compile "$RENDER_LAUNCHER"
    echo "[Code] Embedded renderer created: $RENDER_LAUNCHER"
}

run_renderer() {
    export SCENE_NAME
    export APPEARANCE_MODE
    export MAX_STEPS
    export CAP_MAX
    export NHT_FEATURE_DIM
    export RUN_NAME
    export EXPERIMENT_DIR="$OUT_ROOT/$RUN_NAME"
    export CHECKPOINT_STEP
    export CHECKPOINT_PATH
    export LPIPS_NET
    export ENABLE_EVALUATION
    export USE_NATIVE_DISTORTION
    export USE_FEATURE_DECODER_EMA
    export OVERWRITE_RENDER
    export SAVE_ALPHA
    export MAX_IMAGES
    export FORCE_OUTPUT_EXTENSION
    export RESIZE_PREDICTION_TO_GT
    export CONTINUE_ON_EVAL_ERROR
    export RENDER_LOGICAL_GPU_ID=0

    echo
    echo "============================================================"
    echo "3DGRUT RENDER CONFIGURATION"
    echo "============================================================"
    echo "Scene                : $SCENE_NAME"
    echo "Run name             : $RUN_NAME"
    echo "Experiment directory : $EXPERIMENT_DIR"
    echo "Checkpoint request   : $CHECKPOINT_STEP"
    echo "Checkpoint path      : ${CHECKPOINT_PATH:-auto}"
    echo "Physical GPU         : 1 (hard-locked)"
    echo "PyTorch device       : cuda:0 (physical GPU 1)"
    echo "Venv                  : $VENV_DIR"
    echo "Slang compiler        : $SLANGC_BIN"
    echo "Native distortion    : $USE_NATIVE_DISTORTION"
    echo "NHT EMA              : $USE_FEATURE_DECODER_EMA"
    echo "Renderer             : native 3DGUT from checkpoint config"
    echo "Torch extension dir  : $TORCH_EXTENSIONS_DIR"
    echo "CUDA debug sync      : $CUDA_DEBUG_SYNC"
    echo "Evaluation           : $ENABLE_EVALUATION"
    echo "LPIPS                : $LPIPS_NET"
    echo "Overwrite renders    : $OVERWRITE_RENDER"
    echo "Max images           : ${MAX_IMAGES:-all}"
    echo "Log                  : $LOG_FILE"
    echo "============================================================"
    echo

    cd "$REPO_DIR"
    env \
        CUDA_DEVICE_ORDER=PCI_BUS_ID \
        CUDA_VISIBLE_DEVICES=1 \
        PHYSICAL_GPU_ID=1 \
        TORCH_EXTENSIONS_DIR="$TORCH_EXTENSIONS_DIR" \
        "$VENV_DIR/bin/python" "$RENDER_LAUNCHER"
}

# =============================================================================
# EXECUTION
# =============================================================================

echo "============================================================"
echo "3DGRUT ONE-FILE RENDER"
echo "============================================================"
echo "Script directory : $SCRIPT_DIR"
echo "AI Tuyen root    : $AI_TUYEN_ROOT"
echo "Repository       : $REPO_DIR"
echo "Venv             : $VENV_DIR"
echo "Scene            : $SCENE_NAME"
echo "Physical GPU     : 1 (hard-locked)"
echo "Slang compiler   : $SLANGC_BIN"
echo "============================================================"

check_runtime
ensure_slangc
prepare_dataset_view
patch_tcnn_jit_control
write_embedded_renderer
run_renderer

echo
echo "============================================================"
echo "RENDER FINISHED"
echo "============================================================"
echo "Run      : $OUT_ROOT/$RUN_NAME"
echo "Log      : $LOG_FILE"
echo "============================================================"
