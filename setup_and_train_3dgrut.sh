#!/usr/bin/env bash
#
# One-file installer + trainer for nv-tlabs/3dgrut v2.
#
# This script:
#   1. Checks NVIDIA/CUDA prerequisites.
#   2. Optionally installs Ubuntu build packages.
#   3. Clones nv-tlabs/3dgrut and checks out the pinned tested commit.
#   4. Runs NVIDIA's official UV environment installer.
#   5. Links a COLMAP dataset into the repository.
#   6. Generates the complete Python training launcher embedded below.
#   7. Starts 3DGUT + MCMC + NHT training with live progress.
#   8. Saves on Ctrl+C and auto-resumes on the next run.
#
# The training Python file does NOT need to be uploaded separately.
#
# Expected project layout (automatically detected):
#
#   ~/ai-tuyen/
#   ├── bts_ai_race/
#   │   └── setup_and_train_3dgrut.sh
#   └── dataset/
#       └── phase1/
#           └── public_set/
#               └── HCM0181/
#                   ├── train/
#                   │   ├── images/
#                   │   └── sparse/0/
#                   └── test/
#
# Default run:
#
#   cd ~/ai-tuyen/bts_ai_race
#   GPU_ID=1 bash setup_and_train_3dgrut.sh
#
# Select another scene:
#
#   GPU_ID=1 bash setup_and_train_3dgrut.sh HCM0204
#
# RTX 4060 8 GB safer override:
#
#   CAP_MAX=1500000 \
#   NHT_FEATURE_DIM=32 \
#   GPU_ID=1 bash setup_and_train_3dgrut.sh HCM0181
#
# DATASET_SOURCE can still be supplied explicitly for a custom location.
#
set -Eeuo pipefail
IFS=$'\n\t'

on_error() {
    local exit_code=$?
    echo
    echo "============================================================" >&2
    echo "FAILED at line ${BASH_LINENO[0]} with exit code ${exit_code}" >&2
    echo "Log file: ${LOG_FILE:-not-created}" >&2
    echo "============================================================" >&2
    exit "${exit_code}"
}
trap on_error ERR

on_interrupt() {
    echo
    echo "============================================================" >&2
    echo "INTERRUPT REQUESTED" >&2
    echo "The patched trainer will save ckpt_last.pt before exiting." >&2
    echo "Run this same command again; AUTO_RESUME=true will continue it." >&2
    echo "============================================================" >&2
}
trap on_interrupt INT

# =============================================================================
# USER CONFIGURATION
# Override any value by exporting it before running this script.
# =============================================================================

# Resolve paths from this shell file, not from the current working directory.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
AI_TUYEN_ROOT="$(cd "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd)"

# The first positional argument may select another scene:
#   bash setup_and_train_3dgrut.sh HCM0204
SCENE_NAME="${1:-${SCENE_NAME:-HCM0181}}"

# Default dataset location for the current project layout.
PUBLIC_SET_ROOT="${PUBLIC_SET_ROOT:-$AI_TUYEN_ROOT/dataset/phase1/public_set}"
DATASET_SOURCE="${DATASET_SOURCE:-}"

# Keep the large cloned repository, .venv and run outputs outside bts_ai_race.
WORK_ROOT="${WORK_ROOT:-$AI_TUYEN_ROOT/3dgrut_workspace}"
REPO_DIR="${REPO_DIR:-$WORK_ROOT/3dgrut}"
REPO_URL="${REPO_URL:-https://github.com/nv-tlabs/3dgrut.git}"
REPO_COMMIT="${REPO_COMMIT:-a37ef721012dea0f29c0fcfff2d525023b4e854a}"

# Physical NVIDIA GPU index. This server should train on GPU 1 by default.
# Override when needed, for example: GPU_ID=0 bash setup_and_train_3dgrut.sh
GPU_ID="${GPU_ID:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"

APPEARANCE_MODE="${APPEARANCE_MODE:-native_distortion}"
MAX_STEPS="${MAX_STEPS:-35000}"
GEOMETRY_STEPS="${GEOMETRY_STEPS:-25000}"
COLOR_REFINE_STEPS="${COLOR_REFINE_STEPS:-10000}"
SAVE_STEPS="${SAVE_STEPS:-30000,32500,35000}"

CAP_MAX="${CAP_MAX:-2000000}"
NHT_FEATURE_DIM="${NHT_FEATURE_DIM:-48}"
DATA_FACTOR="${DATA_FACTOR:-1}"
TEST_SPLIT_INTERVAL="${TEST_SPLIT_INTERVAL:-0}"

NORMALIZE_WORLD_SPACE="${NORMALIZE_WORLD_SPACE:-false}"
GSPLAT_IMAGE_DOWNSCALE="${GSPLAT_IMAGE_DOWNSCALE:-false}"
FORCE_REBUILD_PHOTOMETRIC="${FORCE_REBUILD_PHOTOMETRIC:-false}"
OVERWRITE_EXPERIMENT="${OVERWRITE_EXPERIMENT:-false}"
DRY_RUN="${DRY_RUN:-false}"

# Resume the newest checkpoint found under the same experiment name.
AUTO_RESUME="${AUTO_RESUME:-true}"

# Patch train.py so Ctrl+C first writes ckpt_last.pt, then exits.
SAVE_ON_INTERRUPT="${SAVE_ON_INTERRUPT:-true}"

# Keep Rich's live progress bar while also recording a terminal log.
USE_PTY_LOGGING="${USE_PTY_LOGGING:-true}"

# tiny-cuda-nn JIT currently fails on this server's runtime include discovery.
# 0 avoids the warning and uses the stable non-JIT path. Set 1 to retry JIT.
TCNN_JIT_FUSION="${TCNN_JIT_FUSION:-0}"

RUN_NAME="${RUN_NAME:-}"
OUT_ROOT="${OUT_ROOT:-$REPO_DIR/runs}"

# 1: try apt installation when possible; 0: never call apt.
INSTALL_SYSTEM_PACKAGES="${INSTALL_SYSTEM_PACKAGES:-1}"

# 1: rerun NVIDIA's installer even when the environment marker exists.
FORCE_ENV_SETUP="${FORCE_ENV_SETUP:-0}"

# Limit parallel native compilation to avoid host RAM exhaustion.
MAX_JOBS="${MAX_JOBS:-8}"

# Warn before training when the selected physical GPU has less free VRAM.
# For the default 2M Gaussian + NHT feature-dim 48 configuration, 30 GiB is a
# conservative preflight threshold on this 46 GiB L40S server.
MIN_FREE_GPU_MEMORY_MIB="${MIN_FREE_GPU_MEMORY_MIB:-30000}"

# Set STRICT_GPU_MEMORY_CHECK=1 to stop instead of only warning.
STRICT_GPU_MEMORY_CHECK="${STRICT_GPU_MEMORY_CHECK:-0}"

# Modern PyTorch allocator variable.
PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

# Apply physical-GPU selection before environment validation and training.
# Inside PyTorch this selected physical GPU becomes logical cuda:0.
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PHYSICAL_GPU_ID="$GPU_ID"
export PYTORCH_ALLOC_CONF

# =============================================================================
# LOGGING
# =============================================================================

mkdir -p "$WORK_ROOT/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-$WORK_ROOT/logs/3dgrut_${SCENE_NAME}_${TIMESTAMP}.log}"

# Rich progress bars require a TTY. Re-run this same script once inside
# util-linux `script`, which provides a pseudo-terminal and records the output.
if [[ "$USE_PTY_LOGGING" == "true" \
   && -z "${_3DGRUT_PTY_ACTIVE:-}" \
   && -t 1 \
   && -x "$(command -v script 2>/dev/null || true)" ]]; then
    export _3DGRUT_PTY_ACTIVE=1
    export LOG_FILE
    printf -v _3DGRUT_REEXEC_COMMAND '%q ' bash "$0" "$@"
    exec script -q -f -e -c "$_3DGRUT_REEXEC_COMMAND" "$LOG_FILE"
fi

# Non-interactive fallback (nohup, CI, redirected output). This records logs,
# but a live Rich progress bar cannot be rendered without a terminal.
if [[ -z "${_3DGRUT_PTY_ACTIVE:-}" ]]; then
    exec > >(tee -a "$LOG_FILE") 2>&1
fi

echo "============================================================"
echo "3DGRUT ONE-FILE SETUP + TRAIN"
echo "============================================================"
echo "Date             : $(date --iso-8601=seconds)"
echo "Script directory : $SCRIPT_DIR"
echo "AI Tuyen root    : $AI_TUYEN_ROOT"
echo "Public set root  : $PUBLIC_SET_ROOT"
echo "Scene            : $SCENE_NAME"
echo "Work root        : $WORK_ROOT"
echo "Repository       : $REPO_DIR"
echo "Pinned commit    : $REPO_COMMIT"
echo "Physical GPU ID  : $GPU_ID"
echo "CUDA visible     : $CUDA_VISIBLE_DEVICES"
echo "Min free VRAM    : ${MIN_FREE_GPU_MEMORY_MIB} MiB"
echo "Auto resume      : $AUTO_RESUME"
echo "Save on Ctrl+C   : $SAVE_ON_INTERRUPT"
echo "PTY progress     : $USE_PTY_LOGGING"
echo "TCNN JIT fusion  : $TCNN_JIT_FUSION"
echo "Log              : $LOG_FILE"
echo "============================================================"

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

install_ubuntu_packages() {
    [[ "$INSTALL_SYSTEM_PACKAGES" == "1" ]] || {
        echo "[System] Skipping apt packages because INSTALL_SYSTEM_PACKAGES=0."
        return 0
    }

    have_command apt-get || {
        echo "[System] apt-get is unavailable; skipping system package installation."
        return 0
    }

    local -a prefix=()
    if [[ "$(id -u)" -eq 0 ]]; then
        prefix=()
    elif have_command sudo && sudo -n true >/dev/null 2>&1; then
        prefix=(sudo)
    else
        echo "[System] No root/non-interactive sudo access; skipping apt packages."
        echo "[System] Install build-essential, cmake, ninja, git, curl and OpenGL headers manually if setup fails."
        return 0
    fi

    echo "[System] Installing Ubuntu build dependencies..."
    "${prefix[@]}" apt-get update
    DEBIAN_FRONTEND=noninteractive "${prefix[@]}" apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        curl \
        wget \
        build-essential \
        cmake \
        ninja-build \
        pkg-config \
        ffmpeg \
        unzip \
        libgl1-mesa-dev \
        libegl1-mesa-dev \
        libx11-dev \
        libxi-dev \
        libxrandr-dev \
        libxinerama-dev \
        libxcursor-dev \
        libglvnd-dev
}

detect_cuda() {
    have_command nvidia-smi || die "nvidia-smi was not found. Install the NVIDIA driver first."

    [[ "$GPU_ID" =~ ^[0-9]+$ ]] || die \
        "GPU_ID must be a non-negative integer, got: $GPU_ID"

    echo "[CUDA] All physical GPUs:"
    nvidia-smi \
        --query-gpu=index,name,memory.total,memory.used,memory.free,driver_version \
        --format=csv,noheader || true

    if ! nvidia-smi -i "$GPU_ID" --query-gpu=index --format=csv,noheader,nounits \
        >/dev/null 2>&1; then
        die "Physical GPU $GPU_ID does not exist or is not accessible."
    fi

    local gpu_name=""
    local total_mib=""
    local used_mib=""
    local free_mib=""
    local driver_version=""

    gpu_name="$(
        nvidia-smi -i "$GPU_ID" \
            --query-gpu=name \
            --format=csv,noheader \
        | head -n 1 | xargs
    )"
    total_mib="$(
        nvidia-smi -i "$GPU_ID" \
            --query-gpu=memory.total \
            --format=csv,noheader,nounits \
        | head -n 1 | xargs
    )"
    used_mib="$(
        nvidia-smi -i "$GPU_ID" \
            --query-gpu=memory.used \
            --format=csv,noheader,nounits \
        | head -n 1 | xargs
    )"
    free_mib="$(
        nvidia-smi -i "$GPU_ID" \
            --query-gpu=memory.free \
            --format=csv,noheader,nounits \
        | head -n 1 | xargs
    )"
    driver_version="$(
        nvidia-smi -i "$GPU_ID" \
            --query-gpu=driver_version \
            --format=csv,noheader \
        | head -n 1 | xargs
    )"

    echo "[CUDA] Selected physical GPU : $GPU_ID"
    echo "[CUDA] GPU name              : $gpu_name"
    echo "[CUDA] VRAM total            : ${total_mib} MiB"
    echo "[CUDA] VRAM used             : ${used_mib} MiB"
    echo "[CUDA] VRAM free             : ${free_mib} MiB"
    echo "[CUDA] Driver                : $driver_version"
    echo "[CUDA] PyTorch logical GPU   : cuda:0"
    echo "[CUDA] CUDA_VISIBLE_DEVICES  : $CUDA_VISIBLE_DEVICES"

    if [[ "$free_mib" =~ ^[0-9]+$ ]] \
        && (( free_mib < MIN_FREE_GPU_MEMORY_MIB )); then
        local message
        message="Physical GPU $GPU_ID has only ${free_mib} MiB free; recommended minimum is ${MIN_FREE_GPU_MEMORY_MIB} MiB."

        if [[ "$STRICT_GPU_MEMORY_CHECK" == "1" ]]; then
            die "$message"
        fi

        echo "WARNING: $message"
        echo "WARNING: Check running processes with: nvidia-smi -i $GPU_ID"
    fi

    if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
        CUDA_HOME="$(readlink -f "$CUDA_HOME")"
    elif have_command nvcc; then
        local nvcc_real
        nvcc_real="$(readlink -f "$(command -v nvcc)")"
        CUDA_HOME="$(dirname "$(dirname "$nvcc_real")")"
    elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
        CUDA_HOME="$(readlink -f /usr/local/cuda)"
    else
        local candidate=""
        for candidate in /usr/local/cuda-*; do
            if [[ -x "$candidate/bin/nvcc" ]]; then
                CUDA_HOME="$candidate"
            fi
        done
    fi

    [[ -n "${CUDA_HOME:-}" && -x "$CUDA_HOME/bin/nvcc" ]] || die \
        "CUDA toolkit/nvcc was not found. Set CUDA_HOME to a supported CUDA toolkit."

    export CUDA_HOME
    export PATH="$CUDA_HOME/bin:$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

    local cuda_include_dir=""
    local include_candidate=""
    for include_candidate in \
        "$CUDA_HOME/include" \
        "$CUDA_HOME/targets/x86_64-linux/include" \
        "$CUDA_HOME/targets/$(uname -m)-linux/include" \
        "/usr/local/cuda/include" \
        "/usr/include"
    do
        if [[ -f "$include_candidate/vector_types.h" ]]; then
            cuda_include_dir="$include_candidate"
            break
        fi
    done

    if [[ -n "$cuda_include_dir" ]]; then
        export CUDA_INCLUDE_DIR="$cuda_include_dir"
        export CPATH="$cuda_include_dir${CPATH:+:$CPATH}"
        export C_INCLUDE_PATH="$cuda_include_dir${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
        export CPLUS_INCLUDE_PATH="$cuda_include_dir${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
        echo "[CUDA] Runtime include=$cuda_include_dir"
        echo "[CUDA] vector_types.h found"
    else
        echo "WARNING: vector_types.h was not found under CUDA_HOME or standard include paths."
        echo "WARNING: TCNN JIT will remain disabled unless TCNN_JIT_FUSION=1 is requested."
    fi

    echo "[CUDA] CUDA_HOME=$CUDA_HOME"
    echo "[CUDA] nvcc real path=$(readlink -f "$CUDA_HOME/bin/nvcc")"
    "$CUDA_HOME/bin/nvcc" --version
}

install_uv() {
    if have_command uv; then
        echo "[UV] Reusing $(uv --version)"
        return 0
    fi

    have_command curl || die "curl is required to install uv."
    echo "[UV] Installing uv..."
    curl -LsSf --retry 5 --retry-delay 2 https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    have_command uv || die "uv installation completed but the uv command is still unavailable."
    echo "[UV] Installed $(uv --version)"
}

clone_or_update_repo() {
    mkdir -p "$(dirname "$REPO_DIR")"

    if [[ ! -d "$REPO_DIR/.git" ]]; then
        echo "[Git] Cloning $REPO_URL ..."
        git clone --filter=blob:none "$REPO_URL" "$REPO_DIR"
    else
        echo "[Git] Existing repository found: $REPO_DIR"
    fi

    cd "$REPO_DIR"
    git remote set-url origin "$REPO_URL"
    git reset --hard
    git fetch --force origin "$REPO_COMMIT"
    git checkout --detach "$REPO_COMMIT"
    git reset --hard "$REPO_COMMIT"
    git submodule sync --recursive
    git submodule update --init --recursive

    local actual
    actual="$(git rev-parse HEAD)"
    [[ "$actual" == "$REPO_COMMIT" ]] || die \
        "Repository revision mismatch: expected $REPO_COMMIT, got $actual"

    echo "[Git] Checked out: $actual"
}

setup_python_environment() {
    cd "$REPO_DIR"

    [[ -f install_env_uv.sh ]] || die \
        "The pinned checkout does not contain install_env_uv.sh."

    local marker="$REPO_DIR/.env_setup_${REPO_COMMIT}"
    if [[ "$FORCE_ENV_SETUP" == "1" || ! -x "$REPO_DIR/.venv/bin/python" || ! -f "$marker" ]]; then
        echo "[Env] Running NVIDIA's official install_env_uv.sh ..."
        export MAX_JOBS
        bash install_env_uv.sh 3dgrut
        touch "$marker"
    else
        echo "[Env] Existing completed environment detected; skipping reinstall."
        echo "[Env] Set FORCE_ENV_SETUP=1 to rebuild it."
    fi

    [[ -x "$REPO_DIR/.venv/bin/python" ]] || die \
        "Python virtual environment was not created at $REPO_DIR/.venv"

    # shellcheck disable=SC1091
    source "$REPO_DIR/.venv/bin/activate"

    echo "[Env] Python: $(python --version)"
    CUDA_VISIBLE_DEVICES="$GPU_ID" PHYSICAL_GPU_ID="$GPU_ID" python - <<'PYVERIFY'
import os
import torch

print("PyTorch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Selected physical GPU:", os.environ.get("PHYSICAL_GPU_ID"))
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access the selected CUDA GPU.")

if torch.cuda.device_count() != 1:
    raise SystemExit(
        f"Expected exactly one visible CUDA GPU, got {torch.cuda.device_count()}."
    )

free_bytes, total_bytes = torch.cuda.mem_get_info(0)
print("PyTorch logical GPU:", 0)
print("GPU name:", torch.cuda.get_device_name(0))
print("Visible VRAM free MiB:", free_bytes // (1024 * 1024))
print("Visible VRAM total MiB:", total_bytes // (1024 * 1024))
PYVERIFY
}

auto_find_dataset() {
    # Respect an explicitly supplied valid path.
    if [[ -n "$DATASET_SOURCE" && -d "$DATASET_SOURCE" ]]; then
        return 0
    fi

    local candidate=""
    local root=""

    # Search the project's public dataset first. Scene matching is
    # case-insensitive because some folders are lowercase (for example hcm0031).
    for root in \
        "$PUBLIC_SET_ROOT" \
        "$AI_TUYEN_ROOT/dataset/phase1/private_set1" \
        "$AI_TUYEN_ROOT/dataset/phase1" \
        "$WORK_ROOT/data" \
        "$WORK_ROOT/datasets"
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

    # Compatibility with the older directory layout.
    for candidate in \
        "$SCRIPT_DIR/data/$SCENE_NAME" \
        "$REPO_DIR/data/$SCENE_NAME" \
        "$HOME/ai_race_2026/data/$SCENE_NAME"
    do
        if [[ -d "$candidate" ]]; then
            DATASET_SOURCE="$candidate"
            return 0
        fi
    done
}

prepare_dataset_link() {
    auto_find_dataset
    [[ -n "$DATASET_SOURCE" ]] || die \
        "Could not auto-detect scene '$SCENE_NAME'. Expected: $PUBLIC_SET_ROOT/$SCENE_NAME. Set DATASET_SOURCE explicitly for another location."

    DATASET_SOURCE="$(readlink -f "$DATASET_SOURCE")"
    [[ -d "$DATASET_SOURCE" ]] || die "Dataset directory does not exist: $DATASET_SOURCE"

    local ready_root
    if [[ -d "$DATASET_SOURCE/images" && -d "$DATASET_SOURCE/sparse/0" ]]; then
        ready_root="$DATASET_SOURCE"
    elif [[ -d "$DATASET_SOURCE/train/images" && -d "$DATASET_SOURCE/train/sparse/0" ]]; then
        local view_root="$WORK_ROOT/dataset_views/$SCENE_NAME"
        echo "[Data] Competition train/ layout detected; creating a non-destructive dataset view."
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
            "Unsupported dataset layout. Expected images+sparse/0 or train/images+train/sparse/0 under: $DATASET_SOURCE"
    fi

    local required
    for required in \
        "$ready_root/images" \
        "$ready_root/sparse/0/cameras.bin" \
        "$ready_root/sparse/0/images.bin" \
        "$ready_root/sparse/0/points3D.bin"
    do
        [[ -e "$required" ]] || die "Missing dataset component: $required"
    done

    mkdir -p "$REPO_DIR/data"
    local target="$REPO_DIR/data/$SCENE_NAME"

    if [[ -L "$target" ]]; then
        rm -f "$target"
    elif [[ -e "$target" ]]; then
        if [[ "$(readlink -f "$target")" != "$(readlink -f "$ready_root")" ]]; then
            die \
                "A real, different dataset path already exists at $target. Move it or set REPO_DIR to another location."
        fi
    fi

    if [[ ! -e "$target" ]]; then
        ln -s "$ready_root" "$target"
    fi

    echo "[Data] Scene           : $SCENE_NAME"
    echo "[Data] Public set root : $PUBLIC_SET_ROOT"
    echo "[Data] Source          : $DATASET_SOURCE"
    echo "[Data] Training root   : $ready_root"
    echo "[Data] Repository link : $target -> $(readlink -f "$target")"
}

write_embedded_trainer() {
    TRAIN_LAUNCHER="$REPO_DIR/train_3dgrut_v2_embedded.py"
    cat > "$TRAIN_LAUNCHER" <<'__TRAIN_3DGRUT_V2_PY__'
#!/usr/bin/env python3
"""
Native 3DGRUT v2 training launcher.

Pipeline:
- Native 3DGUT rasterizer and native COLMAP distortion (SIMPLE_RADIAL supported).
- MCMC capped at 2,000,000 Gaussians.
- Neural Harmonic Textures with feature dimension 48.
- 25k geometry/appearance + 10k NHT color-only refinement.
- Photometric v2-mild preprocessing for the recommended native-distortion run.
- Horizon-aware weighted L1 and Sobel edge loss.
- Optional raw-data + PPISP ablation.

Place this file in the root of nv-tlabs/3dgrut next to train.py.
"""

from __future__ import annotations

import csv
import json
import math
import os
import py_compile
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, NoReturn

import numpy as np


# =============================================================================
# USER CONFIGURATION
# =============================================================================

SCENE_NAME = os.environ.get("SCENE_NAME", "HCM0181")
SCRIPT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_ROOT / "data"
SOURCE_SCENE_ROOT = DATA_ROOT / SCENE_NAME
PHOTO_DATASET_ROOT = DATA_ROOT / f"{SCENE_NAME}_photo_v4"

# Recommended first run:
#   native_distortion -> photo_v4 + native SIMPLE_RADIAL + no PPISP
# Optional ablation:
#   ppisp_raw         -> raw RGB + native SIMPLE_RADIAL + PPISP
APPEARANCE_MODE = os.environ.get("APPEARANCE_MODE", "native_distortion")

GPU_ID = int(os.environ.get("GPU_ID", "0"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "35000"))
GEOMETRY_STEPS = int(os.environ.get("GEOMETRY_STEPS", "25000"))
COLOR_REFINE_STEPS = int(os.environ.get("COLOR_REFINE_STEPS", "10000"))
SAVE_STEPS = [int(value) for value in os.environ.get("SAVE_STEPS", "30000,32500,35000").split(",") if value.strip()]
CAP_MAX = int(os.environ.get("CAP_MAX", "2000000"))
NHT_FEATURE_DIM = int(os.environ.get("NHT_FEATURE_DIM", "48"))

DATA_FACTOR = int(os.environ.get("DATA_FACTOR", "1"))
NORMALIZE_WORLD_SPACE = os.environ.get("NORMALIZE_WORLD_SPACE", "false").strip().lower() in {"1", "true", "yes", "y", "on"}
GSPLAT_IMAGE_DOWNSCALE = os.environ.get("GSPLAT_IMAGE_DOWNSCALE", "false").strip().lower() in {"1", "true", "yes", "y", "on"}
TEST_SPLIT_INTERVAL = int(os.environ.get("TEST_SPLIT_INTERVAL", "0"))  # 0 = use all available RGB frames

# Native 3DGUT/NHT parameters.
UT_ALPHA = 0.1
PARTICLE_FEATURE_HALF = True
FEATURE_OUTPUT_HALF = True
PARTICLE_KERNEL_MAX_ALPHA = 0.99
NHT_DECODER_HIDDEN_DIM = 128
NHT_DECODER_NUM_LAYERS = 3
NHT_DECODER_LR = 0.00068
NHT_DECODER_EMA_DECAY = 0.95

# Regularization from the official 3DGUT-MCMC-NHT app config.
USE_OPACITY_REG = True
LAMBDA_OPACITY = 0.02
USE_SCALE_REG = True
LAMBDA_SCALE = 0.005

# Horizon-aware loss.
HORIZON_AWARE_LOSS = True
HORIZON_REPORT_FILENAME = "_photometric_report.csv"
HORIZON_TOP_REGION_END = 0.58
HORIZON_SKY_INTERIOR_END = 0.24
HORIZON_SKY_LUMA_MIN = 0.55
HORIZON_SKY_EDGE_MAX = 0.025
HORIZON_EDGE_THRESHOLD = 0.040
HORIZON_SKY_WEIGHT = 0.25
HORIZON_BAND_WEIGHT = 1.25
HORIZON_TOP_FOREGROUND_WEIGHT = 1.10
HORIZON_EDGE_LOSS_WEIGHT = 0.02

# PPISP options used only for APPEARANCE_MODE="ppisp_raw".
# Distillation is disabled so it does not replace the final NHT refinement phase.
PPISP_USE_CONTROLLER = True
PPISP_DISTILLATION_STEPS = 0

# Photometric v2-mild preprocessing.
FORCE_REBUILD_PHOTOMETRIC = os.environ.get("FORCE_REBUILD_PHOTOMETRIC", "false").strip().lower() in {"1", "true", "yes", "y", "on"}
HORIZONTAL_TOP_BOTTOM_RATIO_THRESHOLD = 1.15
MIN_REFERENCE_IMAGES = 6
PROFILE_WIDTH = 256
PROFILE_HEIGHT = 160
PROFILE_SMOOTH_SIGMA = 5.0
PROFILE_VALID_LUMA_MIN = 0.015
PROFILE_VALID_LUMA_MAX = 0.985
TOP_PROFILE_RANGE = (0.04, 0.34)
LOWER_PROFILE_RANGE = (0.58, 0.90)
TOP_FULL_CORRECTION_END = 0.30
BOTTOM_ZERO_CORRECTION_START = 0.58
GAIN_MIN = 0.90
GAIN_MAX = 1.12
PHOTOMETRIC_STRENGTH = 0.55
JPEG_QUALITY = 98
PNG_COMPRESS_LEVEL = 3
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

RUN_NAME = os.environ.get(
    "RUN_NAME",
    (
        f"{SCENE_NAME}_3dgrut_v2_3dgut_mcmc_nht_"
        f"{APPEARANCE_MODE}_{MAX_STEPS // 1000}k_"
        f"{CAP_MAX // 1_000_000}m_fd{NHT_FEATURE_DIM}"
    ),
)
OUT_ROOT = Path(os.environ.get("OUT_ROOT", str(SCRIPT_ROOT / "runs"))).expanduser().resolve()
OVERWRITE_EXPERIMENT = os.environ.get("OVERWRITE_EXPERIMENT", "false").strip().lower() in {"1", "true", "yes", "y", "on"}
DRY_RUN = os.environ.get("DRY_RUN", "false").strip().lower() in {"1", "true", "yes", "y", "on"}
AUTO_RESUME = os.environ.get("AUTO_RESUME", "true").strip().lower() in {"1", "true", "yes", "y", "on"}
SAVE_ON_INTERRUPT = os.environ.get("SAVE_ON_INTERRUPT", "true").strip().lower() in {"1", "true", "yes", "y", "on"}
TCNN_JIT_FUSION = os.environ.get("TCNN_JIT_FUSION", "0").strip().lower() in {"1", "true", "yes", "y", "on"}

# Tested against this public main commit. The script checks source anchors before patching.
EXPECTED_3DGRUT_COMMIT = "a37ef721012dea0f29c0fcfff2d525023b4e854a"
DATASET_PATCH_MARKER = "# BEGIN dataset MISSING RGB FILTER PATCH"
DATASET_NO_SPLIT_MARKER = "# BEGIN dataset NO-SPLIT INDEX PATCH"
HORIZON_PATCH_MARKER = "# BEGIN dataset HORIZON LOSS PATCH"
INTERRUPT_PATCH_MARKER = "# BEGIN SAVE CHECKPOINT ON INTERRUPT PATCH"
TCNN_JIT_PATCH_MARKER = "# BEGIN TCNN JIT CONTROL PATCH"


# =============================================================================
# COMMON HELPERS
# =============================================================================


def fail(message: str) -> NoReturn:
    raise RuntimeError(message)


def list_images(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def ensure_directory_symlink(link_path: Path, target_path: Path) -> None:
    target_path = target_path.resolve()
    if link_path.is_symlink():
        if link_path.resolve() == target_path:
            return
        link_path.unlink()
    elif link_path.exists():
        fail(
            "Cannot create symlink because a real path exists:\n"
            f"  {link_path}\nRename or remove it first."
        )
    link_path.parent.mkdir(parents=True, exist_ok=True)
    link_path.symlink_to(target_path, target_is_directory=True)


def fraction_slice(length: int, interval: tuple[float, float]) -> slice:
    start = max(0, min(int(round(length * interval[0])), length - 1))
    end = max(start + 1, min(int(round(length * interval[1])), length))
    return slice(start, end)


def gaussian_kernel_1d(sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.ones(1, dtype=np.float64)
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(x * x) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


def smooth_profile(profile: np.ndarray, sigma: float) -> np.ndarray:
    kernel = gaussian_kernel_1d(sigma)
    radius = len(kernel) // 2
    padded = np.pad(profile.astype(np.float64), (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float64)


def smoothstep(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    return values * values * (3.0 - 2.0 * values)


def srgb_to_linear(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    return np.where(
        image <= 0.04045,
        image / 12.92,
        ((image + 0.055) / 1.055) ** 2.4,
    )


def linear_to_srgb(image: np.ndarray) -> np.ndarray:
    image = np.clip(image, 0.0, 1.0)
    return np.where(
        image <= 0.0031308,
        12.92 * image,
        1.055 * np.power(image, 1.0 / 2.4) - 0.055,
    )


def luminance_linear(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


# =============================================================================
# PHOTOMETRIC DATASET
# =============================================================================


def load_profile_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        image = image.convert("RGB").resize(
            (PROFILE_WIDTH, PROFILE_HEIGHT),
            Image.Resampling.BILINEAR,
        )
        return np.asarray(image, dtype=np.float32) / 255.0


def analyze_vertical_profile(path: Path) -> dict[str, Any]:
    luma = luminance_linear(srgb_to_linear(load_profile_rgb(path)))
    rows: list[float] = []
    for row in luma:
        valid = row[(row >= PROFILE_VALID_LUMA_MIN) & (row <= PROFILE_VALID_LUMA_MAX)]
        if valid.size < max(8, row.size // 20):
            valid = row
        rows.append(float(np.median(valid)))

    profile = smooth_profile(np.asarray(rows), PROFILE_SMOOTH_SIGMA)
    top = max(float(np.median(profile[fraction_slice(len(profile), TOP_PROFILE_RANGE)])), 1e-6)
    lower = max(float(np.median(profile[fraction_slice(len(profile), LOWER_PROFILE_RANGE)])), 1e-6)
    ratio = top / lower
    return {
        "normalized_profile": profile / lower,
        "top_anchor": top,
        "lower_anchor": lower,
        "top_bottom_ratio": ratio,
        "upper_bright": ratio >= HORIZONTAL_TOP_BOTTOM_RATIO_THRESHOLD,
    }


def save_rgb_uint8(rgb: np.ndarray, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(rgb, mode="RGB")
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        image.save(path, format="JPEG", quality=JPEG_QUALITY, subsampling=0, optimize=False)
    elif suffix == ".png":
        image.save(path, format="PNG", compress_level=PNG_COMPRESS_LEVEL)
    elif suffix == ".webp":
        image.save(path, format="WEBP", quality=JPEG_QUALITY, method=4)
    else:
        image.save(path)


def correct_image(
    source: Path,
    destination: Path,
    normalized_profile: np.ndarray,
    reference_profile: np.ndarray,
) -> dict[str, float]:
    from PIL import Image

    with Image.open(source) as image:
        rgb_srgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0

    height = rgb_srgb.shape[0]
    gain = reference_profile / np.maximum(normalized_profile, 1e-6)
    gain = np.clip(smooth_profile(gain, PROFILE_SMOOTH_SIGMA), GAIN_MIN, GAIN_MAX)
    source_y = np.linspace(0.0, 1.0, len(gain), dtype=np.float64)
    target_y = np.linspace(0.0, 1.0, height, dtype=np.float64)
    gain_y = np.interp(target_y, source_y, gain)
    fade = (target_y - TOP_FULL_CORRECTION_END) / max(
        BOTTOM_ZERO_CORRECTION_START - TOP_FULL_CORRECTION_END, 1e-6
    )
    top_weight = 1.0 - smoothstep(fade)
    effective_gain = 1.0 + PHOTOMETRIC_STRENGTH * top_weight * (gain_y - 1.0)
    effective_gain = np.clip(effective_gain, GAIN_MIN, GAIN_MAX)

    corrected = srgb_to_linear(rgb_srgb) * effective_gain[:, None, None].astype(np.float32)
    corrected = linear_to_srgb(corrected)
    rgb_u8 = (np.clip(corrected, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    save_rgb_uint8(rgb_u8, destination)

    return {
        "gain_min": float(effective_gain.min()),
        "gain_max": float(effective_gain.max()),
        "gain_top_median": float(np.median(effective_gain[fraction_slice(height, TOP_PROFILE_RANGE)])),
        "gain_lower_median": float(np.median(effective_gain[fraction_slice(height, LOWER_PROFILE_RANGE)])),
    }


def photometric_config() -> dict[str, Any]:
    return {
        "version": 2,
        "source_scene": str(SOURCE_SCENE_ROOT.resolve()),
        "threshold": HORIZONTAL_TOP_BOTTOM_RATIO_THRESHOLD,
        "profile_width": PROFILE_WIDTH,
        "profile_height": PROFILE_HEIGHT,
        "profile_smooth_sigma": PROFILE_SMOOTH_SIGMA,
        "top_profile_range": list(TOP_PROFILE_RANGE),
        "lower_profile_range": list(LOWER_PROFILE_RANGE),
        "top_full_correction_end": TOP_FULL_CORRECTION_END,
        "bottom_zero_correction_start": BOTTOM_ZERO_CORRECTION_START,
        "gain_min": GAIN_MIN,
        "gain_max": GAIN_MAX,
        "strength": PHOTOMETRIC_STRENGTH,
        "jpeg_quality": JPEG_QUALITY,
    }


def photometric_dataset_current(source_files: list[Path]) -> bool:
    images_dir = PHOTO_DATASET_ROOT / "images"
    config_path = PHOTO_DATASET_ROOT / "_photometric_config.json"
    report_path = PHOTO_DATASET_ROOT / HORIZON_REPORT_FILENAME
    if FORCE_REBUILD_PHOTOMETRIC or not config_path.is_file() or not report_path.is_file():
        return False
    try:
        if json.loads(config_path.read_text(encoding="utf-8")) != photometric_config():
            return False
    except Exception:
        return False
    generated = list_images(images_dir)
    if len(generated) != len(source_files):
        return False
    source_root = SOURCE_SCENE_ROOT / "images"
    for source in source_files:
        destination = images_dir / source.relative_to(source_root)
        if not destination.is_file() or destination.stat().st_mtime < source.stat().st_mtime:
            return False
    return True


def prepare_photometric_dataset() -> dict[str, Any]:
    from tqdm import tqdm

    source_images_root = SOURCE_SCENE_ROOT / "images"
    source_sparse = SOURCE_SCENE_ROOT / "sparse"
    source_files = list_images(source_images_root)
    if not source_files:
        fail(f"No training images found in: {source_images_root}")
    if not source_sparse.is_dir():
        fail(f"Sparse COLMAP directory is missing: {source_sparse}")

    PHOTO_DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    ensure_directory_symlink(PHOTO_DATASET_ROOT / "sparse", source_sparse)
    report_path = PHOTO_DATASET_ROOT / HORIZON_REPORT_FILENAME

    if photometric_dataset_current(source_files):
        corrected = 0
        upper_bright = 0
        with report_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                upper_bright += row.get("upper_bright") == "True"
                corrected += row.get("corrected") == "True"
        print("[Photometric] Existing photo_v4 dataset is current; reusing it.")
        return {
            "source_images": len(source_files),
            "upper_bright_images": upper_bright,
            "corrected_images": corrected,
            "report_path": str(report_path),
        }

    destination_root = PHOTO_DATASET_ROOT / "images"
    if destination_root.exists() or destination_root.is_symlink():
        remove_path(destination_root)
    destination_root.mkdir(parents=True, exist_ok=True)

    analyses: dict[str, dict[str, Any]] = {}
    print("========== PHOTOMETRIC ANALYSIS ==========")
    for path in tqdm(source_files, desc="Analyzing vertical luminance"):
        name = path.relative_to(source_images_root).as_posix()
        analyses[name] = analyze_vertical_profile(path)

    reference_names = [name for name, item in analyses.items() if item["upper_bright"]]
    correction_enabled = len(reference_names) >= MIN_REFERENCE_IMAGES
    if correction_enabled:
        stack = np.stack([analyses[name]["normalized_profile"] for name in reference_names])
        reference = smooth_profile(np.median(stack, axis=0), PROFILE_SMOOTH_SIGMA)
        reference /= max(
            float(np.median(reference[fraction_slice(len(reference), LOWER_PROFILE_RANGE)])),
            1e-6,
        )
    else:
        reference = np.ones(PROFILE_HEIGHT, dtype=np.float64)
        print("WARNING: too few upper-bright images; copying all images unchanged.")

    rows: list[dict[str, Any]] = []
    for source in tqdm(source_files, desc="Writing photo_v4 dataset"):
        relative = source.relative_to(source_images_root)
        name = relative.as_posix()
        destination = destination_root / relative
        analysis = analyses[name]
        should_correct = correction_enabled and bool(analysis["upper_bright"])
        if should_correct:
            gain_stats = correct_image(
                source,
                destination,
                analysis["normalized_profile"],
                reference,
            )
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            gain_stats = {
                "gain_min": 1.0,
                "gain_max": 1.0,
                "gain_top_median": 1.0,
                "gain_lower_median": 1.0,
            }
        rows.append({
            "image_name": name,
            "top_anchor": f"{analysis['top_anchor']:.10f}",
            "lower_anchor": f"{analysis['lower_anchor']:.10f}",
            "top_bottom_ratio": f"{analysis['top_bottom_ratio']:.10f}",
            "upper_bright": bool(analysis["upper_bright"]),
            "corrected": should_correct,
            **{key: f"{value:.10f}" for key, value in gain_stats.items()},
        })

    with report_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (PHOTO_DATASET_ROOT / "_photometric_config.json").write_text(
        json.dumps(photometric_config(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    corrected_count = sum(bool(row["corrected"]) for row in rows)
    print(f"Source images        : {len(source_files)}")
    print(f"Upper-bright images  : {len(reference_names)}")
    print(f"Corrected images     : {corrected_count}")
    print(f"Derived dataset      : {PHOTO_DATASET_ROOT}")
    print(f"Horizon report       : {report_path}")
    print("==========================================")
    return {
        "source_images": len(source_files),
        "upper_bright_images": len(reference_names),
        "corrected_images": corrected_count,
        "report_path": str(report_path),
    }


# =============================================================================
# 3DGRUT SOURCE PATCHES
# =============================================================================


def locate_3dgrut_root() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in (here, here.parent):
        if (
            (candidate / "train.py").is_file()
            and (candidate / "threedgrut" / "trainer.py").is_file()
            and (candidate / "configs" / "apps" / "colmap_3dgut_mcmc_nht.yaml").is_file()
        ):
            return candidate
    fail("Place this launcher in the root of the 3DGRUT repository, next to train.py.")


def backup_once(path: Path, suffix: str) -> Path:
    backup = path.with_name(path.name + suffix)
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"[Patch] Backup created: {backup}")
    return backup


def replace_once(text: str, old: str, new: str, description: str) -> str:
    count = text.count(old)
    if count != 1:
        fail(
            f"Could not apply {description}: expected one source anchor, found {count}.\n"
            "The checked-out 3DGRUT source may differ from the tested v2 commit."
        )
    return text.replace(old, new, 1)


def patch_colmap_missing_rgb_filter(path: Path) -> None:
    """Apply missing-RGB filtering and fix the upstream no-split index bug."""
    backup = path.with_name(path.name + ".pre_hcm0204_rgb_filter.bak")
    text = path.read_text(encoding="utf-8")

    # The first launcher version already patched this checkout. Rebuild from
    # the pristine backup so the v2 patch is deterministic and idempotent.
    if DATASET_PATCH_MARKER in text:
        if not backup.is_file():
            fail(
                "The old COLMAP patch is present but its pristine backup is "
                f"missing:\n  {backup}\n"
                "Run 'git checkout -- threedgrut/datasets/dataset_colmap.py' "
                "and rerun this launcher."
            )
        print(f"[Patch] Restoring pristine COLMAP loader from: {backup}")
        text = backup.read_text(encoding="utf-8")

    if DATASET_NO_SPLIT_MARKER in text:
        print("[Patch] COLMAP loader v2 is already present.")
        return

    reload_anchor = """        self.load_intrinsics_and_extrinsics()
        frame_indices_before_split = self._filter_cameras()
"""
    reload_replacement = """        self.load_intrinsics_and_extrinsics()

        # BEGIN HCM0204 MISSING RGB FILTER PATCH
        _rgb_frame_indices = self._filter_missing_rgb_frames()
        _camera_frame_indices = self._filter_cameras()
        frame_indices_before_split = [
            _rgb_frame_indices[index]
            for index in _camera_frame_indices
        ]
        # END HCM0204 MISSING RGB FILTER PATCH
"""
    text = replace_once(
        text,
        reload_anchor,
        reload_replacement,
        "COLMAP reload filter v2",
    )

    helper_anchor = "    def _filter_cameras(self) -> list[int]:\n"
    helper_code = r"""    # BEGIN HCM0204 MISSING RGB FILTER PATCH
    def _filter_missing_rgb_frames(self) -> list[int]:
        # Keep only registered COLMAP frames with an existing RGB file.
        image_root = os.path.join(self.path, "images")
        if not os.path.isdir(image_root):
            raise ValueError(f"Image folder does not exist: {image_root}")

        kept_extrinsics = []
        kept_original_indices = []
        dropped_names = []

        for original_index, extr in enumerate(self.cam_extrinsics):
            image_path = os.path.join(image_root, str(extr.name))
            if os.path.isfile(image_path):
                kept_extrinsics.append(extr)
                kept_original_indices.append(original_index)
            else:
                dropped_names.append(str(extr.name))

        original_count = len(self.cam_extrinsics)
        self.cam_extrinsics = kept_extrinsics

        logger.info(
            f"[COLMAP] Train RGB filter: keeping {len(kept_extrinsics)}/"
            f"{original_count} registered frames; dropping "
            f"{len(dropped_names)} frames without RGB files."
        )
        if dropped_names:
            logger.info(
                "[COLMAP] Example dropped frames: "
                + ", ".join(dropped_names[:5])
            )
        if not kept_extrinsics:
            raise ValueError(
                f"No registered COLMAP frame has a matching RGB file in "
                f"{image_root}."
            )

        return kept_original_indices

    # END HCM0204 MISSING RGB FILTER PATCH

"""
    text = replace_once(
        text,
        helper_anchor,
        helper_code + helper_anchor,
        "COLMAP missing-RGB helper",
    )

    # Upstream 3DGRUT uses np.where(indices)[0] for cam_extrinsics but direct
    # indexing for poses/image_paths. With test_split_interval <= 0, indices is
    # [0, 1, 2, ...], so np.where removes frame 0 and creates a 239-vs-240
    # mismatch. Convert the no-split case to an all-True boolean mask.
    split_anchor = """        indices = np.arange(self.n_frames)

        # If test_split_interval is set, every test_split_interval frame will be excluded from the training set
        # If test_split_interval is non-positive, all images will be used for training and testing
        if self.test_split_interval > 0:
            if self.split == "train":
                indices = np.mod(indices, self.test_split_interval) != 0
            else:
                indices = np.mod(indices, self.test_split_interval) == 0
"""
    split_replacement = """        # BEGIN HCM0204 NO-SPLIT INDEX PATCH
        frame_numbers = np.arange(self.n_frames)

        # If test_split_interval is positive, build the normal boolean split
        # mask. If it is non-positive, use an all-True boolean mask so every
        # per-frame container is indexed identically.
        if self.test_split_interval > 0:
            if self.split == "train":
                indices = np.mod(frame_numbers, self.test_split_interval) != 0
            else:
                indices = np.mod(frame_numbers, self.test_split_interval) == 0
        else:
            indices = np.ones(self.n_frames, dtype=bool)
        # END HCM0204 NO-SPLIT INDEX PATCH
"""
    text = replace_once(
        text,
        split_anchor,
        split_replacement,
        "COLMAP no-split index fix",
    )

    final_anchor = """        # Update the number of frames to only include the samples from the split
        self.n_frames = self.poses.shape[0]

        # Clear existing worker caches to force recreation with new intrinsics
"""
    final_replacement = """        # Update the number of frames to only include the samples from the split
        self.n_frames = self.poses.shape[0]

        # BEGIN HCM0204 FRAME STATE ASSERT PATCH
        _frame_lengths = {
            "n_frames": int(self.n_frames),
            "cam_extrinsics": len(self.cam_extrinsics),
            "poses": len(self.poses),
            "image_paths": len(self.image_paths),
            "mask_paths": len(self.mask_paths),
            "camera_centers": len(self.camera_centers),
        }
        if len(set(_frame_lengths.values())) != 1:
            raise RuntimeError(
                "COLMAP per-frame state is inconsistent after filtering: "
                f"{_frame_lengths}"
            )
        logger.info(
            "[COLMAP] Final frame state synchronized: "
            f"{_frame_lengths}"
        )
        # END HCM0204 FRAME STATE ASSERT PATCH

        # Clear existing worker caches to force recreation with new intrinsics
"""
    text = replace_once(
        text,
        final_anchor,
        final_replacement,
        "COLMAP frame-state assertion",
    )

    path.write_text(text, encoding="utf-8")
    py_compile.compile(str(path), doraise=True)
    print(f"[Patch] Updated COLMAP loader to RGB/no-split patch v2: {path}")


def patch_horizon_aware_loss(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if HORIZON_PATCH_MARKER in text:
        print("[Patch] Horizon-aware loss is already present.")
        return

    text = replace_once(text, "import os\n", "import csv\nimport os\n", "csv import")
    text = replace_once(
        text,
        "import torch.nn as nn\n",
        "import torch.nn as nn\nimport torch.nn.functional as F\n",
        "torch functional import",
    )

    module_helpers = r'''

# BEGIN HCM0204 HORIZON LOSS PATCH HELPERS

def _horizon_normalize_name(name: str) -> str:
    return str(name).replace("\\", "/").lstrip("./")


def _horizon_luminance(rgb: torch.Tensor) -> torch.Tensor:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def _horizon_sobel_magnitude(rgb: torch.Tensor) -> torch.Tensor:
    luma = _horizon_luminance(rgb).unsqueeze(1)
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=rgb.device,
        dtype=rgb.dtype,
    ).view(1, 1, 3, 3) / 8.0
    kernel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=rgb.device,
        dtype=rgb.dtype,
    ).view(1, 1, 3, 3) / 8.0
    gx = F.conv2d(luma, kernel_x, padding=1)
    gy = F.conv2d(luma, kernel_y, padding=1)
    return torch.sqrt(gx.square() + gy.square() + 1e-12).squeeze(1)

# END HCM0204 HORIZON LOSS PATCH HELPERS
'''
    text = replace_once(
        text,
        "\nclass Trainer3DGRUT:\n",
        module_helpers + "\nclass Trainer3DGRUT:\n",
        "horizon module helpers",
    )
    text = replace_once(
        text,
        """        self.init_dataloaders(conf)\n        self.init_scene_extents(self.train_dataset)\n""",
        """        self.init_dataloaders(conf)\n        self._init_horizon_aware_loss(conf)\n        self.init_scene_extents(self.train_dataset)\n""",
        "horizon initialization",
    )

    trainer_methods = r'''    # BEGIN HCM0204 HORIZON LOSS PATCH
    def _init_horizon_aware_loss(self, conf: DictConfig) -> None:
        self._horizon_enabled = bool(
            OmegaConf.select(conf, "loss.horizon_aware.enabled", default=False)
        )
        self._horizon_train_flags = None
        if not self._horizon_enabled:
            return

        report_raw = str(
            OmegaConf.select(conf, "loss.horizon_aware.report_path", default="") or ""
        )
        report_name = str(
            OmegaConf.select(
                conf,
                "loss.horizon_aware.report_filename",
                default="_photometric_report.csv",
            )
        )
        report_path = Path(report_raw) if report_raw else Path(conf.path) / report_name
        report_path = report_path.expanduser().resolve()
        if not report_path.is_file():
            raise FileNotFoundError(f"Horizon report does not exist: {report_path}")

        flags_by_name = {}
        with report_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                name = _horizon_normalize_name(row.get("image_name", ""))
                raw = str(row.get("upper_bright", "")).strip().lower()
                flags_by_name[name] = raw in {"1", "true", "yes", "y"}

        image_root = (Path(conf.path) / self.train_dataset.get_images_folder()).resolve()
        flags = []
        missing = 0
        for image_path_raw in self.train_dataset.image_paths:
            image_path = Path(str(image_path_raw)).resolve()
            try:
                name = image_path.relative_to(image_root).as_posix()
            except ValueError:
                name = image_path.name
            name = _horizon_normalize_name(name)
            missing += name not in flags_by_name
            flags.append(bool(flags_by_name.get(name, False)))

        self._horizon_train_flags = torch.tensor(
            flags, dtype=torch.bool, device=self.device
        )
        logger.info(
            f"[HorizonLoss] Active train frames: {sum(flags)}/{len(flags)}; "
            f"missing report rows: {missing}; report={report_path}"
        )

    def _horizon_aware_rgb_loss(
        self,
        rgb_pred: torch.Tensor,
        rgb_gt: torch.Tensor,
        mask: Optional[torch.Tensor],
        active: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if not active:
            zero = rgb_pred.sum() * 0.0
            return torch.abs(rgb_pred - rgb_gt).mean(), zero, {}

        cfg = self.conf.loss.horizon_aware
        batch, height, width, _ = rgb_gt.shape
        gt_edges = _horizon_sobel_magnitude(rgb_gt).detach()
        gt_luma = _horizon_luminance(rgb_gt).detach()
        y = torch.linspace(
            0.0, 1.0, height, device=rgb_gt.device, dtype=rgb_gt.dtype
        ).view(1, height, 1)

        top = y <= float(cfg.top_region_end)
        sky_region = y <= float(cfg.sky_interior_end)
        sky = sky_region & (gt_luma >= float(cfg.sky_luma_min)) & (
            gt_edges <= float(cfg.sky_edge_max)
        )
        band = top & (gt_edges >= float(cfg.edge_threshold))
        foreground = top & ~sky & ~band

        weights = torch.ones(
            (batch, height, width), device=rgb_gt.device, dtype=rgb_gt.dtype
        )
        weights = torch.where(
            foreground,
            torch.as_tensor(float(cfg.top_foreground_weight), device=rgb_gt.device, dtype=rgb_gt.dtype),
            weights,
        )
        weights = torch.where(
            sky,
            torch.as_tensor(float(cfg.sky_weight), device=rgb_gt.device, dtype=rgb_gt.dtype),
            weights,
        )
        weights = torch.where(
            band,
            torch.as_tensor(float(cfg.band_weight), device=rgb_gt.device, dtype=rgb_gt.dtype),
            weights,
        )

        valid = torch.ones_like(weights, dtype=torch.bool)
        if mask is not None:
            valid = mask[..., 0] > 0.5
            weights = weights * valid.to(weights.dtype)

        pixel_l1 = torch.abs(rgb_pred - rgb_gt).mean(dim=-1)
        weighted_l1 = (pixel_l1 * weights).sum() / weights.sum().clamp_min(1.0)

        pred_edges = _horizon_sobel_magnitude(rgb_pred)
        edge_mask = band & valid
        denominator = edge_mask.to(rgb_gt.dtype).sum()
        if bool(denominator.detach().item() > 0):
            edge_loss = (
                torch.abs(pred_edges - gt_edges) * edge_mask.to(rgb_gt.dtype)
            ).sum() / denominator.clamp_min(1.0)
        else:
            edge_loss = rgb_pred.sum() * 0.0

        total_pixels = torch.as_tensor(
            batch * height * width, device=rgb_gt.device, dtype=rgb_gt.dtype
        )
        stats = {
            "sky_ratio": sky.to(rgb_gt.dtype).sum() / total_pixels,
            "band_ratio": band.to(rgb_gt.dtype).sum() / total_pixels,
            "top_foreground_ratio": foreground.to(rgb_gt.dtype).sum() / total_pixels,
            "mean_weight": weights.sum() / valid.to(rgb_gt.dtype).sum().clamp_min(1.0),
        }
        return weighted_l1, edge_loss, stats

    # END HCM0204 HORIZON LOSS PATCH

'''
    text = replace_once(
        text,
        "    def init_dataloaders(self, conf: DictConfig):\n",
        trainer_methods + "    def init_dataloaders(self, conf: DictConfig):\n",
        "horizon trainer methods",
    )

    original_l1 = """        # L1 loss\n        loss_l1 = torch.zeros(1, device=self.device)\n        lambda_l1 = 0.0\n        if self.conf.loss.use_l1:\n            with torch.cuda.nvtx.range(f\"loss-l1\"):\n                loss_l1 = torch.abs(rgb_pred - rgb_gt).mean()\n                lambda_l1 = self.conf.loss.lambda_l1\n"""
    replacement_l1 = """        # L1 loss\n        loss_l1 = torch.zeros(1, device=self.device)\n        loss_horizon_edge = torch.zeros(1, device=self.device)\n        lambda_l1 = 0.0\n        lambda_horizon_edge = 0.0\n        horizon_stats = {}\n        if self.conf.loss.use_l1:\n            with torch.cuda.nvtx.range(f\"loss-l1\"):\n                use_horizon = (\n                    self._horizon_enabled\n                    and torch.is_grad_enabled()\n                    and self._horizon_train_flags is not None\n                )\n                active_horizon = False\n                if use_horizon:\n                    frame_idx = int(gpu_batch.frame_idx)\n                    if 0 <= frame_idx < len(self._horizon_train_flags):\n                        active_horizon = bool(self._horizon_train_flags[frame_idx].item())\n                    loss_l1, loss_horizon_edge, horizon_stats = self._horizon_aware_rgb_loss(\n                        rgb_pred=rgb_pred,\n                        rgb_gt=rgb_gt,\n                        mask=mask,\n                        active=active_horizon,\n                    )\n                    if active_horizon:\n                        lambda_horizon_edge = float(\n                            self.conf.loss.horizon_aware.edge_loss_weight\n                        )\n                else:\n                    loss_l1 = torch.abs(rgb_pred - rgb_gt).mean()\n                lambda_l1 = self.conf.loss.lambda_l1\n"""
    text = replace_once(text, original_l1, replacement_l1, "horizon weighted L1")

    original_total = """        # Total loss\n        loss = lambda_l1 * loss_l1 + lambda_ssim * loss_ssim + lambda_opacity * loss_opacity + lambda_scale * loss_scale\n        return dict(\n            total_loss=loss,\n            l1_loss=lambda_l1 * loss_l1,\n            l2_loss=lambda_l2 * loss_l2,\n            ssim_loss=lambda_ssim * loss_ssim,\n            opacity_loss=lambda_opacity * loss_opacity,\n            scale_loss=lambda_scale * loss_scale,\n        )\n"""
    replacement_total = """        # Total loss\n        loss = (\n            lambda_l1 * loss_l1\n            + lambda_ssim * loss_ssim\n            + lambda_opacity * loss_opacity\n            + lambda_scale * loss_scale\n            + lambda_horizon_edge * loss_horizon_edge\n        )\n        result = dict(\n            total_loss=loss,\n            l1_loss=lambda_l1 * loss_l1,\n            l2_loss=lambda_l2 * loss_l2,\n            ssim_loss=lambda_ssim * loss_ssim,\n            opacity_loss=lambda_opacity * loss_opacity,\n            scale_loss=lambda_scale * loss_scale,\n            horizon_edge_loss=lambda_horizon_edge * loss_horizon_edge,\n        )\n        for name, value in horizon_stats.items():\n            result[f\"horizon_{name}\"] = value\n        return result\n"""
    text = replace_once(text, original_total, replacement_total, "horizon total loss")

    log_anchor = """            if self.post_processing is not None and \"post_processing_reg_loss\" in batch_metrics[\"losses\"]:\n"""
    log_replacement = """            if \"horizon_edge_loss\" in batch_metrics[\"losses\"]:\n                writer.add_scalar(\n                    \"loss/horizon_edge/train\",\n                    batch_metrics[\"losses\"][\"horizon_edge_loss\"],\n                    global_step,\n                )\n                for horizon_name in (\n                    \"horizon_sky_ratio\",\n                    \"horizon_band_ratio\",\n                    \"horizon_top_foreground_ratio\",\n                    \"horizon_mean_weight\",\n                ):\n                    if horizon_name in batch_metrics[\"losses\"]:\n                        writer.add_scalar(\n                            f\"loss/{horizon_name}/train\",\n                            batch_metrics[\"losses\"][horizon_name],\n                            global_step,\n                        )\n            if self.post_processing is not None and \"post_processing_reg_loss\" in batch_metrics[\"losses\"]:\n"""
    text = replace_once(text, log_anchor, log_replacement, "horizon TensorBoard logging")

    backup_once(path, ".pre_hcm0204_horizon.bak")
    path.write_text(text, encoding="utf-8")
    py_compile.compile(str(path), doraise=True)
    print(f"[Patch] Updated: {path}")



def patch_save_checkpoint_on_interrupt(path: Path) -> None:
    """Make Ctrl+C save ckpt_last.pt and return a non-zero exit code."""
    text = path.read_text(encoding="utf-8")
    if INTERRUPT_PATCH_MARKER in text:
        print("[Patch] Save-on-interrupt support is already present.")
        return

    anchor = """    trainer = Trainer3DGRUT(conf)
    try:
        trainer.run_training()
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
"""
    replacement = """    trainer = Trainer3DGRUT(conf)
    try:
        trainer.run_training()
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
        # BEGIN SAVE CHECKPOINT ON INTERRUPT PATCH
        try:
            trainer.save_checkpoint(last_checkpoint=True)
            logger.warning(
                f"Interrupt checkpoint saved at global step {trainer.global_step}. "
                "Run the launcher again to resume automatically."
            )
        except Exception as checkpoint_error:
            logger.error(
                f"Could not save interrupt checkpoint: {checkpoint_error}"
            )
        raise
        # END SAVE CHECKPOINT ON INTERRUPT PATCH
"""
    text = replace_once(
        text,
        anchor,
        replacement,
        "save checkpoint on KeyboardInterrupt",
    )
    backup_once(path, ".pre_interrupt_checkpoint.bak")
    path.write_text(text, encoding="utf-8")
    py_compile.compile(str(path), doraise=True)
    print(f"[Patch] Ctrl+C checkpoint saving enabled: {path}")


def patch_tcnn_jit_control(path: Path) -> None:
    """Allow TCNN_JIT_FUSION=0 to skip a known-failing RTC JIT attempt."""
    text = path.read_text(encoding="utf-8")
    if TCNN_JIT_PATCH_MARKER in text:
        print("[Patch] tiny-cuda-nn JIT control is already present.")
        return

    text = replace_once(
        text,
        "import tinycudann as tcnn\nimport torch\n",
        "import os\n\nimport tinycudann as tcnn\nimport torch\n",
        "feature decoder os import",
    )

    anchor = """        if hasattr(tcnn, "supports_jit_fusion"):
            self.network.jit_fusion = tcnn.supports_jit_fusion()
"""
    replacement = """        # BEGIN TCNN JIT CONTROL PATCH
        if hasattr(tcnn, "supports_jit_fusion"):
            raw_jit = os.environ.get("TCNN_JIT_FUSION", "0").strip().lower()
            requested_jit = raw_jit in {"1", "true", "yes", "y", "on"}
            self.network.jit_fusion = bool(
                requested_jit and tcnn.supports_jit_fusion()
            )
        # END TCNN JIT CONTROL PATCH
"""
    text = replace_once(
        text,
        anchor,
        replacement,
        "tiny-cuda-nn JIT environment control",
    )
    backup_once(path, ".pre_tcnn_jit_control.bak")
    path.write_text(text, encoding="utf-8")
    py_compile.compile(str(path), doraise=True)
    print(
        f"[Patch] tiny-cuda-nn JIT controlled by TCNN_JIT_FUSION="
        f"{int(TCNN_JIT_FUSION)}: {path}"
    )


# =============================================================================
# LAUNCH
# =============================================================================


def git_revision(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def newest_checkpoint(experiment_dir: Path) -> Path | None:
    if not experiment_dir.is_dir():
        return None

    candidates = {
        *experiment_dir.rglob("ckpt_last.pt"),
        *experiment_dir.rglob("ckpt_*.pt"),
    }
    candidates = {
        path for path in candidates
        if path.is_file() and path.stat().st_size > 0
    }
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def main() -> None:
    if APPEARANCE_MODE not in {"native_distortion", "ppisp_raw"}:
        fail("APPEARANCE_MODE must be 'native_distortion' or 'ppisp_raw'.")
    if GEOMETRY_STEPS + COLOR_REFINE_STEPS != MAX_STEPS:
        fail("GEOMETRY_STEPS + COLOR_REFINE_STEPS must equal MAX_STEPS.")
    if MAX_STEPS not in SAVE_STEPS:
        fail("SAVE_STEPS must contain MAX_STEPS.")
    if any(step <= 0 or step > MAX_STEPS for step in SAVE_STEPS):
        fail(f"Every save step must satisfy 0 < step <= {MAX_STEPS}.")

    root = locate_3dgrut_root()
    dataset_file = root / "threedgrut" / "datasets" / "dataset_colmap.py"
    trainer_file = root / "threedgrut" / "trainer.py"
    feature_decoder_file = root / "threedgrut" / "model" / "feature_decoder.py"
    train_entry = root / "train.py"

    required = [
        SOURCE_SCENE_ROOT / "images",
        SOURCE_SCENE_ROOT / "sparse" / "0" / "cameras.bin",
        SOURCE_SCENE_ROOT / "sparse" / "0" / "images.bin",
        SOURCE_SCENE_ROOT / "sparse" / "0" / "points3D.bin",
    ]
    for path in required:
        if not path.exists():
            fail(f"Dataset component is missing: {path}")

    print("========== 3DGRUT V2 PREPARATION ==========")
    print(f"Repository root       : {root}")
    print(f"Git revision          : {git_revision(root)}")
    print(f"Tested commit         : {EXPECTED_3DGRUT_COMMIT}")

    photo_summary = prepare_photometric_dataset()
    horizon_report = (PHOTO_DATASET_ROOT / HORIZON_REPORT_FILENAME).resolve()

    if APPEARANCE_MODE == "native_distortion":
        train_scene = PHOTO_DATASET_ROOT.resolve()
        pp_method = "null"
        load_exif = "false"
    else:
        train_scene = SOURCE_SCENE_ROOT.resolve()
        pp_method = "ppisp"
        load_exif = "true"

    patch_colmap_missing_rgb_filter(dataset_file)
    patch_horizon_aware_loss(trainer_file)
    patch_tcnn_jit_control(feature_decoder_file)
    if SAVE_ON_INTERRUPT:
        patch_save_checkpoint_on_interrupt(train_entry)

    experiment_dir = OUT_ROOT / RUN_NAME
    resume_checkpoint = newest_checkpoint(experiment_dir)

    if resume_checkpoint is not None:
        if OVERWRITE_EXPERIMENT:
            print(
                f"[Resume] OVERWRITE_EXPERIMENT=True; deleting old run: "
                f"{experiment_dir}"
            )
            shutil.rmtree(experiment_dir)
            resume_checkpoint = None
        elif AUTO_RESUME:
            print(f"[Resume] Found checkpoint: {resume_checkpoint}")
            print("[Resume] Training will continue from its stored global_step.")
        else:
            fail(
                f"A checkpoint already exists under: {experiment_dir}\n"
                "Set AUTO_RESUME=true to continue, OVERWRITE_EXPERIMENT=true "
                "to restart, or change RUN_NAME."
            )

    command = [
        sys.executable,
        str(train_entry),
        "--config-name", "apps/colmap_3dgut_mcmc_nht.yaml",
        f"path={train_scene}",
        f"out_dir={OUT_ROOT.resolve()}",
        f"experiment_name={RUN_NAME}",
        f"n_iterations={MAX_STEPS}",
        f"num_workers={NUM_WORKERS}",
        "val_frequency=999999",
        "test_last=false",
        "compute_extra_metrics=false",
        f"dataset.downsample_factor={DATA_FACTOR}",
        f"dataset.test_split_interval={TEST_SPLIT_INTERVAL}",
        f"dataset.normalize_world_space={str(NORMALIZE_WORLD_SPACE).lower()}",
        f"dataset.gsplat_image_downscale={str(GSPLAT_IMAGE_DOWNSCALE).lower()}",
        f"dataset.load_exif={load_exif}",
        f"strategy.add.max_n_gaussians={CAP_MAX}",
        f"strategy.add.end_iteration={GEOMETRY_STEPS}",
        f"strategy.relocate.end_iteration={GEOMETRY_STEPS}",
        f"strategy.perturb.end_iteration={GEOMETRY_STEPS}",
        f"model.nht_features.dim={NHT_FEATURE_DIM}",
        f"model.nht_decoder.hidden_dim={NHT_DECODER_HIDDEN_DIM}",
        f"model.nht_decoder.num_layers={NHT_DECODER_NUM_LAYERS}",
        f"model.nht_decoder.learning_rate={NHT_DECODER_LR}",
        f"model.nht_decoder.ema_decay={NHT_DECODER_EMA_DECAY}",
        f"model.nht_decoder.color_refine_steps={COLOR_REFINE_STEPS}",
        f"model.nht_decoder.scheduler.max_steps={MAX_STEPS}",
        f"scheduler.features.max_steps={MAX_STEPS}",
        f"scheduler.positions.max_steps={GEOMETRY_STEPS}",
        f"render.particle_feature_half={str(PARTICLE_FEATURE_HALF).lower()}",
        f"render.feature_output_half={str(FEATURE_OUTPUT_HALF).lower()}",
        f"render.particle_kernel_max_alpha={PARTICLE_KERNEL_MAX_ALPHA}",
        f"render.splat.ut_alpha={UT_ALPHA}",
        f"loss.use_opacity={str(USE_OPACITY_REG).lower()}",
        f"loss.lambda_opacity={LAMBDA_OPACITY}",
        f"loss.use_scale={str(USE_SCALE_REG).lower()}",
        f"loss.lambda_scale={LAMBDA_SCALE}",
        "checkpoint.iterations=[" + ",".join(str(step) for step in SAVE_STEPS) + "]",
        f"post_processing.method={pp_method}",
        f"+loss.horizon_aware.enabled={str(HORIZON_AWARE_LOSS).lower()}",
        f"+loss.horizon_aware.report_path={horizon_report}",
        f"+loss.horizon_aware.report_filename={HORIZON_REPORT_FILENAME}",
        f"+loss.horizon_aware.top_region_end={HORIZON_TOP_REGION_END}",
        f"+loss.horizon_aware.sky_interior_end={HORIZON_SKY_INTERIOR_END}",
        f"+loss.horizon_aware.sky_luma_min={HORIZON_SKY_LUMA_MIN}",
        f"+loss.horizon_aware.sky_edge_max={HORIZON_SKY_EDGE_MAX}",
        f"+loss.horizon_aware.edge_threshold={HORIZON_EDGE_THRESHOLD}",
        f"+loss.horizon_aware.sky_weight={HORIZON_SKY_WEIGHT}",
        f"+loss.horizon_aware.band_weight={HORIZON_BAND_WEIGHT}",
        f"+loss.horizon_aware.top_foreground_weight={HORIZON_TOP_FOREGROUND_WEIGHT}",
        f"+loss.horizon_aware.edge_loss_weight={HORIZON_EDGE_LOSS_WEIGHT}",
    ]
    if APPEARANCE_MODE == "ppisp_raw":
        command.extend([
            f"post_processing.use_controller={str(PPISP_USE_CONTROLLER).lower()}",
            f"post_processing.n_distillation_steps={PPISP_DISTILLATION_STEPS}",
        ])

    if resume_checkpoint is not None:
        command.append(f"resume={resume_checkpoint.resolve()}")

    print("\n========== 3DGRUT TRAIN CONFIG ==========")
    print(f"Mode                  : {APPEARANCE_MODE}")
    print(f"Dataset               : {train_scene}")
    print(f"Photo images          : {photo_summary['source_images']}")
    print(f"Photo corrected       : {photo_summary['corrected_images']}")
    print(f"Upper-bright images   : {photo_summary['upper_bright_images']}")
    print("Camera projection     : native 3DGUT distortion")
    print(f"Steps                 : {MAX_STEPS:,}")
    print(f"Geometry steps        : {GEOMETRY_STEPS:,}")
    print(f"Color refine steps    : {COLOR_REFINE_STEPS:,}")
    print(f"Save steps            : {SAVE_STEPS}")
    print(f"MCMC cap              : {CAP_MAX:,}")
    print(f"NHT feature dim       : {NHT_FEATURE_DIM}")
    print(f"PPISP                 : {pp_method}")
    print(f"Horizon report        : {horizon_report}")
    print(f"TCNN JIT fusion       : {TCNN_JIT_FUSION}")
    print(f"Auto resume           : {AUTO_RESUME}")
    print(f"Resume checkpoint     : {resume_checkpoint or 'none'}")
    print(f"Output experiment     : {experiment_dir}")
    print("\nCommand:")
    print(" \\\n  ".join(command))

    if DRY_RUN:
        print("\nDRY_RUN=True: patches were checked/applied; training was not started.")
        return

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["TCNN_JIT_FUSION"] = "1" if TCNN_JIT_FUSION else "0"
    environment.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    subprocess.run(command, cwd=str(root), env=environment, check=True)

    checkpoint = newest_checkpoint(experiment_dir)
    if checkpoint is None:
        fail(f"Training finished but ckpt_last.pt was not found under: {experiment_dir}")

    manifest = {
        "scene": SCENE_NAME,
        "run_name": RUN_NAME,
        "appearance_mode": APPEARANCE_MODE,
        "checkpoint": str(checkpoint.resolve()),
        "experiment_dir": str(experiment_dir.resolve()),
        "source_scene": str(SOURCE_SCENE_ROOT.resolve()),
        "train_scene": str(train_scene),
        "horizon_report": str(horizon_report),
        "git_revision": git_revision(root),
        "max_steps": MAX_STEPS,
        "geometry_steps": GEOMETRY_STEPS,
        "color_refine_steps": COLOR_REFINE_STEPS,
        "cap_max": CAP_MAX,
        "nht_feature_dim": NHT_FEATURE_DIM,
        "resumed_from": str(resume_checkpoint.resolve()) if resume_checkpoint else None,
        "tcnn_jit_fusion": TCNN_JIT_FUSION,
    }
    manifest_path = experiment_dir / "latest_run.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("\n========== 3DGRUT TRAIN FINISHED ==========")
    print(f"Checkpoint : {checkpoint}")
    print(f"Manifest   : {manifest_path}")


if __name__ == "__main__":
    main()

__TRAIN_3DGRUT_V2_PY__
    chmod +x "$TRAIN_LAUNCHER"
    "$REPO_DIR/.venv/bin/python" -m py_compile "$TRAIN_LAUNCHER"
    echo "[Code] Embedded trainer created and syntax-checked: $TRAIN_LAUNCHER"
}

run_training() {
    cd "$REPO_DIR"

    export SCENE_NAME
    export GPU_ID
    export NUM_WORKERS
    export APPEARANCE_MODE
    export MAX_STEPS
    export GEOMETRY_STEPS
    export COLOR_REFINE_STEPS
    export SAVE_STEPS
    export CAP_MAX
    export NHT_FEATURE_DIM
    export DATA_FACTOR
    export TEST_SPLIT_INTERVAL
    export NORMALIZE_WORLD_SPACE
    export GSPLAT_IMAGE_DOWNSCALE
    export FORCE_REBUILD_PHOTOMETRIC
    export OVERWRITE_EXPERIMENT
    export DRY_RUN
    export AUTO_RESUME
    export SAVE_ON_INTERRUPT
    export TCNN_JIT_FUSION
    export OUT_ROOT
    export PYTORCH_ALLOC_CONF
    export CUDA_DEVICE_ORDER
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    export PHYSICAL_GPU_ID="$GPU_ID"
    export PYTHONUNBUFFERED=1
    export MAX_JOBS

    if [[ -n "$RUN_NAME" ]]; then
        export RUN_NAME
    else
        unset RUN_NAME || true
    fi

    echo
    echo "============================================================"
    echo "TRAINING CONFIGURATION"
    echo "============================================================"
    echo "Dataset              : $REPO_DIR/data/$SCENE_NAME"
    echo "Physical GPU         : $GPU_ID"
    echo "PyTorch GPU          : cuda:0 (mapped from physical GPU $GPU_ID)"
    echo "CUDA visible devices : $CUDA_VISIBLE_DEVICES"
    echo "Appearance mode      : $APPEARANCE_MODE"
    echo "Max steps            : $MAX_STEPS"
    echo "Geometry steps       : $GEOMETRY_STEPS"
    echo "Color refine steps   : $COLOR_REFINE_STEPS"
    echo "Save steps           : $SAVE_STEPS"
    echo "Gaussian cap         : $CAP_MAX"
    echo "NHT feature dim      : $NHT_FEATURE_DIM"
    echo "Auto resume          : $AUTO_RESUME"
    echo "Save on Ctrl+C       : $SAVE_ON_INTERRUPT"
    echo "TCNN JIT fusion      : $TCNN_JIT_FUSION"
    echo "Output root          : $OUT_ROOT"
    echo "============================================================"
    echo

    "$REPO_DIR/.venv/bin/python" "$TRAIN_LAUNCHER"
}

# =============================================================================
# EXECUTION
# =============================================================================

install_ubuntu_packages
detect_cuda
install_uv
clone_or_update_repo
setup_python_environment
prepare_dataset_link
write_embedded_trainer
run_training

echo
echo "============================================================"
echo "ALL DONE"
echo "============================================================"
echo "Repository : $REPO_DIR"
echo "Runs       : $OUT_ROOT"
echo "Log        : $LOG_FILE"
echo "============================================================"
