#!/usr/bin/env bash
#
# ONE-FILE GSPLAT INSTALL + ROUND-2 SETUP + FULL PIPELINE
#
# Pipeline source:
#   Embedded run_sub_r2.sh
#
# Dataset path behavior:
#   ~/ai-tuyen/dataset/phase1/public_set/<scene>
#
# Supported scene layouts:
#
#   <scene>/
#   ├── images/
#   ├── sparse/0/
#   └── test/test_poses.csv
#
# or:
#
#   <scene>/
#   ├── train/
#   │   ├── images/
#   │   └── sparse/0/
#   └── test/test_poses.csv
#
# Full sequence:
#   system prerequisites → clone gsplat → create venv → install PyTorch
#   → build/install gsplat CUDA extension + example dependencies
#   → workspace links → train each seed → selfcheck first seed
#   → render each seed → mean ensemble → enhancer train/apply
#
# Packaging is disabled by default:
#   ENABLE_PACKAGING=0
#
# Enable the original JPEG/ZIP stage:
#   ENABLE_PACKAGING=1 bash setup_and_run_sub_r2_full_gpu0.sh
#
set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_VERSION="2026-07-24-v2-gsplat-bootstrap"

on_error() {
    local exit_code=$?
    echo
    echo "============================================================" >&2
    echo "FAILED at line ${BASH_LINENO[0]} with exit code ${exit_code}" >&2
    echo "Setup log: ${SETUP_LOG:-not-created}" >&2
    echo "============================================================" >&2
    exit "$exit_code"
}
trap on_error ERR

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

if [[ -f "$SCRIPT_DIR/gsplat/examples/simple_trainer.py" ]]; then
    DETECTED_PROJECT_DIR="$SCRIPT_DIR"
elif [[ -f "$SCRIPT_DIR/../gsplat/examples/simple_trainer.py" ]]; then
    DETECTED_PROJECT_DIR="$(cd "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd)"
else
    DETECTED_PROJECT_DIR="$SCRIPT_DIR"
fi

PROJECT_DIR="${PROJECT_DIR:-$DETECTED_PROJECT_DIR}"
PROJECT_DIR="$(readlink -f "$PROJECT_DIR")"

AI_TUYEN_ROOT="${AI_TUYEN_ROOT:-$(cd "$PROJECT_DIR/.." >/dev/null 2>&1 && pwd)}"
PUBLIC_SET_ROOT="${PUBLIC_SET_ROOT:-$AI_TUYEN_ROOT/dataset/phase1/public_set}"
DATASET_ROOT="${DATASET_ROOT:-$PUBLIC_SET_ROOT}"

GPU_ID="${GPU_ID:-0}"

# gsplat source setup. The official repository is used by default.
GSPLAT_DIR="${GSPLAT_DIR:-$PROJECT_DIR/gsplat}"
GSPLAT_REPO_URL="${GSPLAT_REPO_URL:-https://github.com/nerfstudio-project/gsplat.git}"
GSPLAT_REF="${GSPLAT_REF:-main}"

# Existing repositories are reused by default. Set GSPLAT_UPDATE=1 to fetch and
# checkout GSPLAT_REF. Set GSPLAT_FORCE_RESET=1 only when local modifications
# may be discarded.
GSPLAT_UPDATE="${GSPLAT_UPDATE:-0}"
GSPLAT_FORCE_RESET="${GSPLAT_FORCE_RESET:-0}"

# auto: install only when imports/markers are missing
# 1   : always run gsplat/example dependency installation
# 0   : never install Python packages
INSTALL_GSPLAT="${INSTALL_GSPLAT:-auto}"

# PyTorch must exist before gsplat is built. When torch is absent, these package
# specs are installed. Override TORCH_INDEX_URL for a particular CUDA wheel
# channel, for example:
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
TORCH_PACKAGES="${TORCH_PACKAGES:-torch==2.9.1 torchvision==0.24.1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"

MAX_JOBS="${MAX_JOBS:-8}"
FORCE_GSPLAT_REBUILD="${FORCE_GSPLAT_REBUILD:-0}"
PATCH_GSPLAT_RUN_SUB_COMPAT="${PATCH_GSPLAT_RUN_SUB_COMPAT:-1}"
GSPLAT_BUILD_MARKER="${GSPLAT_BUILD_MARKER:-}"

# Preserve explicit empty values for two-machine splitting.
SCENES_HCM="${SCENES_HCM-"HCM0421 HCM0539 HCM0540 HCM0644 HCM0674"}"
SCENES_OBJ="${SCENES_OBJ-"bonsai chair"}"

SEEDS="${SEEDS:-"42 7"}"
CAP_HCM="${CAP_HCM:-6000000}"
CAP_OBJ="${CAP_OBJ:-3000000}"
OBJ_SH_DEGREE="${OBJ_SH_DEGREE:-3}"
SUBTAG="${SUBTAG:-1}"
ENH_ARCH="${ENH_ARCH:-vgg}"

ENABLE_PACKAGING="${ENABLE_PACKAGING:-0}"
SETUP_ONLY="${SETUP_ONLY:-0}"

INSTALL_SYSTEM_PACKAGES="${INSTALL_SYSTEM_PACKAGES:-1}"
SETUP_PYTHON_ENV="${SETUP_PYTHON_ENV:-auto}"
FORCE_PIP_SETUP="${FORCE_PIP_SETUP:-0}"
INSTALL_PROJECT_EDITABLE="${INSTALL_PROJECT_EDITABLE:-auto}"
EXTRA_PIP_REQUIREMENTS="${EXTRA_PIP_REQUIREMENTS:-}"

MIN_FREE_DISK_GB="${MIN_FREE_DISK_GB:-20}"
MIN_FREE_GPU_MEMORY_MIB="${MIN_FREE_GPU_MEMORY_MIB:-2000}"
STRICT_GPU_MEMORY_CHECK="${STRICT_GPU_MEMORY_CHECK:-0}"

VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PY="$VENV_DIR/bin/python"
GSPLAT_BUILD_MARKER="${GSPLAT_BUILD_MARKER:-$VENV_DIR/.gsplat_source_installed}"

WORKSPACE_R2_ROOT="${WORKSPACE_R2_ROOT:-$PROJECT_DIR/workspace_r2}"
TOOLS_DIR="$PROJECT_DIR/tools"
GENERATED_WORKER="${GENERATED_WORKER:-$TOOLS_DIR/.run_sub_r2_embedded_worker.sh}"

# run_sub_r2.sh searches these names in this exact order. We create the first
# path as a non-destructive alias to DATASET_ROOT.
R2_ALIAS_PARENT="$PROJECT_DIR/VAI_NVS_DATA_ROUND_2"
R2_ALIAS="$R2_ALIAS_PARENT/VAI_NVS_DATA_ROUND2"

mkdir -p "$PROJECT_DIR/logs/r2_setup"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SETUP_LOG="${SETUP_LOG:-$PROJECT_DIR/logs/r2_setup/setup_full_r2_${TIMESTAMP}.log}"

if [[ -z "${_R2_FULL_SETUP_LOG_ACTIVE:-}" ]]; then
    export _R2_FULL_SETUP_LOG_ACTIVE=1
    exec > >(tee -a "$SETUP_LOG") 2>&1
fi

say() {
    echo
    echo "[$(date +%F' '%T)] ============================================================"
    echo "[$(date +%F' '%T)] $*"
    echo "[$(date +%F' '%T)] ============================================================"
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

have_command() {
    command -v "$1" >/dev/null 2>&1
}

disk_guard() {
    local free_gb
    free_gb="$(
        df --output=avail -BG "$PROJECT_DIR" \
        | tail -n 1 \
        | tr -dc '0-9'
    )"

    [[ "$free_gb" =~ ^[0-9]+$ ]] || die \
        "Could not determine free disk space."

    if (( free_gb < MIN_FREE_DISK_GB )); then
        die "Only ${free_gb} GiB free; require ${MIN_FREE_DISK_GB} GiB."
    fi

    echo "[Disk] Free=${free_gb} GiB; guard=${MIN_FREE_DISK_GB} GiB"
}

check_gpu() {
    have_command nvidia-smi || die "nvidia-smi was not found."
    [[ "$GPU_ID" =~ ^[0-9]+$ ]] || die \
        "GPU_ID must be a non-negative integer."

    nvidia-smi -i "$GPU_ID" \
        --query-gpu=index \
        --format=csv,noheader,nounits \
        >/dev/null 2>&1 || die \
        "Physical GPU $GPU_ID does not exist or is unavailable."

    local free_mib
    free_mib="$(
        nvidia-smi -i "$GPU_ID" \
            --query-gpu=memory.free \
            --format=csv,noheader,nounits \
        | head -n 1 \
        | xargs
    )"

    nvidia-smi -i "$GPU_ID" \
        --query-gpu=index,name,memory.total,memory.used,memory.free,driver_version \
        --format=csv,noheader \
        | sed 's/^/[GPU] /'

    if [[ "$free_mib" =~ ^[0-9]+$ ]] \
       && (( free_mib < MIN_FREE_GPU_MEMORY_MIB ))
    then
        local message
        message="GPU $GPU_ID has ${free_mib} MiB free; threshold=${MIN_FREE_GPU_MEMORY_MIB} MiB."

        if [[ "$STRICT_GPU_MEMORY_CHECK" == "1" ]]; then
            die "$message"
        fi

        echo "WARNING: $message"
    fi
}

install_system_packages() {
    [[ "$INSTALL_SYSTEM_PACKAGES" == "1" ]] || {
        echo "[System] Skipping apt packages."
        return
    }

    have_command apt-get || {
        echo "[System] apt-get unavailable; skipping."
        return
    }

    local -a prefix=()

    if [[ "$(id -u)" -eq 0 ]]; then
        prefix=()
    elif have_command sudo && sudo -n true >/dev/null 2>&1; then
        prefix=(sudo)
    else
        echo "[System] No root/non-interactive sudo; skipping apt."
        return
    fi

    echo "[System] Installing common dependencies..."
    "${prefix[@]}" apt-get update
    DEBIAN_FRONTEND=noninteractive "${prefix[@]}" apt-get install -y \
        --no-install-recommends \
        python3 \
        python3-venv \
        python3-dev \
        build-essential \
        cmake \
        ninja-build \
        git \
        pkg-config \
        ffmpeg \
        libgl1 \
        libglib2.0-0
}

setup_cuda_toolkit() {
    local nvcc_path=""
    local nvcc_real=""

    if [[ -n "${CUDA_HOME:-}" && -x "$CUDA_HOME/bin/nvcc" ]]; then
        CUDA_HOME="$(readlink -f "$CUDA_HOME")"
    elif have_command nvcc; then
        nvcc_path="$(command -v nvcc)"
        nvcc_real="$(readlink -f "$nvcc_path")"
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
        "CUDA toolkit/nvcc was not found. Install a CUDA toolkit compatible " \
        "with the selected PyTorch build or set CUDA_HOME."

    export CUDA_HOME
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
    export MAX_JOBS

    echo "[CUDA] CUDA_HOME=$CUDA_HOME"
    echo "[CUDA] nvcc=$(readlink -f "$CUDA_HOME/bin/nvcc")"
    "$CUDA_HOME/bin/nvcc" --version
}

clone_or_prepare_gsplat() {
    mkdir -p "$(dirname "$GSPLAT_DIR")"

    if [[ ! -d "$GSPLAT_DIR/.git" ]]; then
        [[ ! -e "$GSPLAT_DIR" ]] || die \
            "GSPLAT_DIR exists but is not a Git repository: $GSPLAT_DIR"

        echo "[Git] Cloning gsplat..."
        echo "[Git] Repository : $GSPLAT_REPO_URL"
        echo "[Git] Destination: $GSPLAT_DIR"

        git clone --recursive "$GSPLAT_REPO_URL" "$GSPLAT_DIR"
        GSPLAT_UPDATE=1
    else
        echo "[Git] Reusing existing gsplat source: $GSPLAT_DIR"
    fi

    (
        cd "$GSPLAT_DIR"
        git remote set-url origin "$GSPLAT_REPO_URL"

        if [[ "$GSPLAT_UPDATE" == "1" ]]; then
            echo "[Git] Fetching gsplat ref: $GSPLAT_REF"
            git fetch --tags --force origin

            if [[ "$GSPLAT_FORCE_RESET" == "1" ]]; then
                git reset --hard
                git clean -fd
            elif [[ -n "$(git status --porcelain)" ]]; then
                die \
                    "gsplat has local changes. Commit/stash them, or set " \
                    "GSPLAT_FORCE_RESET=1 to discard them."
            fi

            if git rev-parse --verify --quiet "$GSPLAT_REF^{commit}" >/dev/null; then
                git checkout --detach "$GSPLAT_REF"
            elif git rev-parse --verify --quiet "origin/$GSPLAT_REF^{commit}" >/dev/null; then
                git checkout --detach "origin/$GSPLAT_REF"
            else
                # Supports explicit commit hashes that are not local yet.
                git fetch --force origin "$GSPLAT_REF"
                git checkout --detach FETCH_HEAD
            fi
        fi

        git submodule sync --recursive
        git submodule update --init --recursive

        echo "[Git] gsplat revision: $(git rev-parse HEAD)"
        echo "[Git] gsplat status:"
        git status --short
    )

    [[ -f "$GSPLAT_DIR/examples/simple_trainer.py" ]] || die \
        "Cloned gsplat source does not contain examples/simple_trainer.py"
    [[ -f "$GSPLAT_DIR/examples/requirements.txt" ]] || die \
        "Cloned gsplat source does not contain examples/requirements.txt"
}

patch_gsplat_run_sub_compat() {
    [[ "$PATCH_GSPLAT_RUN_SUB_COMPAT" == "1" ]] || {
        echo "[Patch] Skipping run_sub_r2 compatibility patch."
        return
    }

    "$PYTHON_BIN" - "$GSPLAT_DIR/examples/simple_trainer.py" <<'PY_PATCH_GSPLAT'
from pathlib import Path
import py_compile
import shutil
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
marker = "# BEGIN RUN_SUB_R2 COMPAT PATCH"

if marker in text:
    print(f"[Patch] run_sub_r2 compatibility already present: {path}")
    raise SystemExit(0)

backup = path.with_name(path.name + ".pre_run_sub_r2_compat.bak")
if not backup.exists():
    shutil.copy2(path, backup)


def replace_once(old: str, new: str, label: str) -> None:
    global text
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label}: expected one source anchor, found {count}. "
            "Set GSPLAT_REF to a compatible revision or disable "
            "PATCH_GSPLAT_RUN_SUB_COMPAT."
        )
    text = text.replace(old, new, 1)


# Tyro exposes dataclass fields as CLI flags with underscores converted to
# hyphens. These fields provide --global-seed, --raw-distortion,
# --dist-k1-override and --dist-k2-override.
config_anchor = """    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
"""
config_replacement = """    # BEGIN RUN_SUB_R2 COMPAT PATCH
    # Reproducible seed used by tools/run_sub_r2.sh.
    global_seed: int = 42

    # Compatibility flag for the production command. Current gsplat's COLMAP
    # parser already keeps camera distortion and passes it to 3DGUT, so this
    # flag intentionally does not undistort or alter the input images.
    raw_distortion: bool = False

    # Optional measured radial-distortion overrides used by USE_TRUE_DIST=1.
    dist_k1_override: Optional[float] = None
    dist_k2_override: Optional[float] = None
    # END RUN_SUB_R2 COMPAT PATCH

    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
"""
replace_once(config_anchor, config_replacement, "Config compatibility fields")

replace_once(
    "        set_random_seed(42 + local_rank)\n",
    "        set_random_seed(cfg.global_seed + local_rank)\n",
    "configurable global seed",
)

parser_anchor = """            self.parser = Parser(
                data_dir=cfg.data_dir,
                factor=cfg.data_factor,
                normalize=cfg.normalize_world_space,
                test_every=cfg.test_every,
                load_exposure=cfg.load_exposure,
            )
            self.trainset = Dataset(
"""
parser_replacement = """            self.parser = Parser(
                data_dir=cfg.data_dir,
                factor=cfg.data_factor,
                normalize=cfg.normalize_world_space,
                test_every=cfg.test_every,
                load_exposure=cfg.load_exposure,
            )

            # BEGIN RUN_SUB_R2 DISTORTION OVERRIDE PATCH
            if (
                cfg.dist_k1_override is not None
                or cfg.dist_k2_override is not None
            ):
                import numpy as _np

                for _camera_id, _params in self.parser.params_dict.items():
                    _updated = _np.zeros(4, dtype=_np.float32)
                    _params_array = _np.asarray(_params, dtype=_np.float32)
                    _updated[: min(len(_params_array), 4)] = _params_array[:4]

                    if cfg.dist_k1_override is not None:
                        _updated[0] = float(cfg.dist_k1_override)
                    if cfg.dist_k2_override is not None:
                        _updated[1] = float(cfg.dist_k2_override)

                    self.parser.params_dict[_camera_id] = _updated

                print(
                    "[Parser] radial distortion override: "
                    f"k1={cfg.dist_k1_override}, "
                    f"k2={cfg.dist_k2_override}"
                )
            # END RUN_SUB_R2 DISTORTION OVERRIDE PATCH

            self.trainset = Dataset(
"""
replace_once(parser_anchor, parser_replacement, "distortion override")

path.write_text(text, encoding="utf-8")
py_compile.compile(str(path), doraise=True)
print(f"[Patch] Added run_sub_r2 compatibility: {path}")
print(f"[Patch] Backup: {backup}")
PY_PATCH_GSPLAT
}

install_torch_if_missing() {
    if "$PY" -c "import torch, torchvision" >/dev/null 2>&1; then
        echo "[Python] Existing PyTorch installation detected."
        return
    fi

    local -a torch_packages=()
    read -r -a torch_packages <<< "$TORCH_PACKAGES"
    (( ${#torch_packages[@]} > 0 )) || die "TORCH_PACKAGES is empty."

    echo "[Python] Installing PyTorch before building gsplat:"
    printf '  %q' "${torch_packages[@]}"
    echo

    if [[ -n "$TORCH_INDEX_URL" ]]; then
        "$PY" -m pip install \
            --index-url "$TORCH_INDEX_URL" \
            "${torch_packages[@]}"
    else
        "$PY" -m pip install "${torch_packages[@]}"
    fi
}

install_gsplat_source() {
    local should_install=0
    local current_revision=""
    local marker_revision=""

    current_revision="$(git -C "$GSPLAT_DIR" rev-parse HEAD)"

    case "$INSTALL_GSPLAT" in
        1|true|yes|on)
            should_install=1
            ;;
        0|false|no|off)
            should_install=0
            ;;
        auto)
            if [[ "$FORCE_GSPLAT_REBUILD" == "1" ]]; then
                should_install=1
            elif [[ ! -f "$GSPLAT_BUILD_MARKER" ]]; then
                should_install=1
            else
                marker_revision="$(cat "$GSPLAT_BUILD_MARKER" 2>/dev/null || true)"
                if [[ "$marker_revision" != "$current_revision" ]]; then
                    should_install=1
                elif ! (
                    cd "$PROJECT_DIR"
                    "$PY" -c \
                        "import torch, torchvision, gsplat, cv2, pycolmap; assert hasattr(pycolmap, 'Reconstruction') or hasattr(pycolmap, 'SceneManager')"
                ) >/dev/null 2>&1
                then
                    should_install=1
                fi
            fi
            ;;
        *)
            die "INSTALL_GSPLAT must be auto, true, or false."
            ;;
    esac

    if (( should_install == 0 )); then
        echo "[Python] gsplat source environment is already ready."
        return
    fi

    echo "[Python] Installing gsplat source and example dependencies..."
    "$PY" -m pip install --upgrade pip setuptools wheel ninja packaging

    install_torch_if_missing

    # Build/install the CUDA extension from the cloned source.
    (
        cd "$GSPLAT_DIR"
        MAX_JOBS="$MAX_JOBS" \
        "$PY" -m pip install \
            --no-build-isolation \
            --config-settings editable_mode=compat \
            -e .
    )

    # Install the exact dependencies required by simple_trainer.py, render and
    # PPISP/bilateral-grid support. PyTorch is installed first intentionally.
    "$PY" -m pip install \
        --no-build-isolation \
        -r "$GSPLAT_DIR/examples/requirements.txt"

    # Reinstall editable source last in case dependency resolution changed it.
    (
        cd "$GSPLAT_DIR"
        MAX_JOBS="$MAX_JOBS" \
        "$PY" -m pip install \
            --no-build-isolation \
            --config-settings editable_mode=compat \
            -e .
    )

    printf '%s\n' "$current_revision" > "$GSPLAT_BUILD_MARKER"
    echo "[Python] gsplat install marker: $GSPLAT_BUILD_MARKER"
}

verify_gsplat_pipeline() {
    (
        cd "$PROJECT_DIR"
        "$PY" - "$GSPLAT_DIR/examples/simple_trainer.py" <<'PY_VERIFY_GSPLAT'
import importlib
import subprocess
import sys
from pathlib import Path

trainer = Path(sys.argv[1])
required_modules = [
    "torch",
    "torchvision",
    "gsplat",
    "numpy",
    "cv2",
    "tyro",
    "tensorboard",
    "torchmetrics",
]

missing = []
for name in required_modules:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f"{name}: {exc}")

try:
    import pycolmap
    if not (
        hasattr(pycolmap, "Reconstruction")
        or hasattr(pycolmap, "SceneManager")
    ):
        raise AttributeError(
            "neither Reconstruction nor SceneManager is available"
        )
except Exception as exc:
    missing.append(f"pycolmap parser API: {exc}")

if missing:
    raise SystemExit(
        "Missing or broken Python dependencies:\n  - " + "\n  - ".join(missing)
    )

import torch
import gsplat

print("[Verify] Python:", sys.version.split()[0])
print("[Verify] torch:", torch.__version__)
print("[Verify] torch CUDA:", torch.version.cuda)
print("[Verify] gsplat:", getattr(gsplat, "__version__", "source/editable"))
print("[Verify] CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access CUDA.")
if torch.cuda.device_count() != 1:
    raise SystemExit(
        f"Expected one visible CUDA GPU, got {torch.cuda.device_count()}."
    )

print("[Verify] logical cuda:0:", torch.cuda.get_device_name(0))

proc = subprocess.run(
    [sys.executable, str(trainer), "mcmc", "--help"],
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    check=False,
)
help_text = proc.stdout

if proc.returncode != 0:
    raise SystemExit(
        "simple_trainer.py mcmc --help failed:\n" + help_text[-4000:]
    )

required_flags = [
    "--with-ut",
    "--with-eval3d",
    "--raw-distortion",
    "--strategy.cap-max",
    "--global-seed",
    "--save-steps",
    "--eval-steps",
]
missing_flags = [flag for flag in required_flags if flag not in help_text]

if missing_flags:
    raise SystemExit(
        "The selected gsplat revision does not expose the flags required by "
        "run_sub_r2.sh: "
        + ", ".join(missing_flags)
        + "\nUse GSPLAT_REPO_URL/GSPLAT_REF to select the compatible fork or commit."
    )

print("[Verify] Required run_sub_r2 trainer flags: OK")
PY_VERIFY_GSPLAT
    )
}

setup_python_environment() {
    if [[ ! -x "$PY" ]]; then
        echo "[Python] Creating venv: $VENV_DIR"
        "$PYTHON_BIN" -m venv "$VENV_DIR"
    else
        echo "[Python] Reusing venv: $VENV_DIR"
    fi

    "$PY" -m pip install --upgrade pip setuptools wheel ninja packaging

    install_gsplat_source

    if [[ -n "$EXTRA_PIP_REQUIREMENTS" ]]; then
        local -a extra_packages=()
        read -r -a extra_packages <<< "$EXTRA_PIP_REQUIREMENTS"
        if (( ${#extra_packages[@]} > 0 )); then
            "$PY" -m pip install "${extra_packages[@]}"
        fi
    fi

    verify_gsplat_pipeline
}

find_scene_source() {
    local scene="$1"
    local candidate=""

    if [[ -d "$DATASET_ROOT/$scene" ]]; then
        readlink -f "$DATASET_ROOT/$scene"
        return
    fi

    candidate="$(
        find "$DATASET_ROOT" \
            -mindepth 1 \
            -maxdepth 1 \
            -type d \
            -iname "$scene" \
            -print \
            -quit 2>/dev/null || true
    )"

    [[ -n "$candidate" ]] || return 1
    readlink -f "$candidate"
}

validate_scene() {
    local scene="$1"
    local source="$2"
    local train_root=""

    if [[ -d "$source/images" && -d "$source/sparse/0" ]]; then
        train_root="$source"
    elif [[ -d "$source/train/images" && -d "$source/train/sparse/0" ]]; then
        train_root="$source/train"
    else
        die \
            "Unsupported layout for $scene. Expected images+sparse/0 or " \
            "train/images+train/sparse/0 under: $source"
    fi

    local required
    for required in \
        "$train_root/images" \
        "$train_root/sparse/0/cameras.bin" \
        "$train_root/sparse/0/images.bin" \
        "$train_root/sparse/0/points3D.bin" \
        "$source/test/test_poses.csv"
    do
        [[ -e "$required" ]] || die "Missing scene component: $required"
    done

    printf '%s\n' "$train_root"
}

prepare_workspace_scene() {
    local scene="$1"
    local train_root="$2"
    local target="$WORKSPACE_R2_ROOT/$scene"

    mkdir -p "$WORKSPACE_R2_ROOT"

    if [[ -L "$target" ]]; then
        rm -f "$target"
    elif [[ -e "$target" ]]; then
        if [[ -d "$target/images" && -d "$target/sparse/0" ]]; then
            echo "[Data] Reusing prepared workspace: $target"
            return
        fi

        die "Incompatible real path already exists: $target"
    fi

    ln -s "$train_root" "$target"
    echo "[Data] Workspace: $target -> $(readlink -f "$target")"
}

prepare_project_gsplat_link() {
    local expected="$PROJECT_DIR/gsplat"
    local actual
    actual="$(readlink -f "$GSPLAT_DIR")"

    if [[ "$expected" == "$GSPLAT_DIR" ]]; then
        return
    fi

    if [[ -L "$expected" ]]; then
        rm -f "$expected"
    elif [[ -e "$expected" ]]; then
        local existing
        existing="$(readlink -f "$expected")"
        [[ "$existing" == "$actual" ]] || die \
            "PROJECT_DIR/gsplat already exists and points elsewhere: $expected"
        return
    fi

    ln -s "$actual" "$expected"
    echo "[Data] gsplat link: $expected -> $actual"
}

prepare_r2_alias() {
    DATASET_ROOT="$(readlink -f "$DATASET_ROOT")"
    mkdir -p "$R2_ALIAS_PARENT"

    if [[ -L "$R2_ALIAS" ]]; then
        local existing_target
        existing_target="$(readlink -f "$R2_ALIAS")"

        if [[ "$existing_target" != "$DATASET_ROOT" ]]; then
            echo "[Data] Updating R2 alias:"
            echo "  old: $R2_ALIAS -> $existing_target"
            echo "  new: $R2_ALIAS -> $DATASET_ROOT"
            rm -f "$R2_ALIAS"
            ln -s "$DATASET_ROOT" "$R2_ALIAS"
        fi
    elif [[ -e "$R2_ALIAS" ]]; then
        local existing_real
        existing_real="$(readlink -f "$R2_ALIAS")"

        [[ "$existing_real" == "$DATASET_ROOT" ]] || die \
            "Cannot create R2 alias because a different real path exists: $R2_ALIAS"
    else
        ln -s "$DATASET_ROOT" "$R2_ALIAS"
    fi

    echo "[Data] R2 alias: $R2_ALIAS -> $(readlink -f "$R2_ALIAS")"
}

write_embedded_worker() {
    mkdir -p "$TOOLS_DIR"

    cat > "$GENERATED_WORKER" <<'__RUN_SUB_R2_PAYLOAD__'
#!/bin/bash
# ============================================================================
# ROUND 2 — production: train → selfcheck → render → ensemble → L16-XL → zip.
#
# 2 nhánh chiến thuật (bắt buộc, không phải tuỳ chọn — xem prepare_r2.sh):
#   HCM*        (SIMPLE_RADIAL k1≈+0.009): 3DGUT --with-ut --with-eval3d --raw-distortion
#   bonsai/chair(SIMPLE_PINHOLE, video)  : CLASSIC — parser assert nếu bật raw-distortion
#
# ROUND 2 KHÔNG có GT test → sau seed đầu mỗi scene chạy SELFCHECK (render 3 pose
# train qua đúng đường CSV, PSNR>=20) — chặn bug transform trước khi phí GPU.
#
# Config qua env (mặc định an toàn, chỉnh sau khi có gate round 2):
#   SUBTAG=2 SEEDS="42 7 123"  CAP_HCM=...  CAP_OBJ=...  ENH_ARCH=vgg|unet
#   SUBTAG tách output giữa các lần production (renders __sub$SUBTAG, zip R2_SUB$SUBTAG)
# Chạy: tmux new -s r2 && bash tools/run_sub_r2.sh 2>&1 | tee /tmp/run_r2.txt
# ============================================================================
set -uo pipefail
PROJ="$(cd "$(dirname "$0")/.." && pwd)"; cd "$PROJ"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset BTS_TMASK   # chống lây nhiễm transient-mask (DOC3): production KHÔNG mask trừ khi gate bảo bật
PY=.venv/bin/python
# TỰ DÒ root data R2 (layout local có tầng lồng, server phẳng — xem prepare_r2.sh)
R2=""
for c in VAI_NVS_DATA_ROUND_2/VAI_NVS_DATA_ROUND2 VAI_NVS_DATA_ROUND_2 VAI_NVS_DATA_ROUND2; do
  [ -s "$c/bonsai/test/test_poses.csv" ] && { R2="$c"; break; }
done
[ -n "$R2" ] || { echo "❌ không tìm thấy data round 2"; exit 1; }
# Override được qua env để CHIA VIỆC 2 máy (server: SCENES_OBJ="" · 4060: SCENES_HCM="")
SCENES_HCM=${SCENES_HCM-"HCM0421 HCM0539 HCM0540 HCM0644 HCM0674"}
SCENES_OBJ=${SCENES_OBJ-"bonsai chair"}
OBJ_SH_DEGREE=${OBJ_SH_DEGREE:-3}   # O4 gate quyết: 4 nếu SH4 thắng trên holdout bonsai/chair
SEEDS=${SEEDS:-"42 7"}
CAP_HCM=${CAP_HCM:-6000000}     # 1320×989 = 1/4 pixel của round 1 → knee thấp hơn 12M nhiều
CAP_OBJ=${CAP_OBJ:-3000000}     # scene object-centric nhỏ (54-80k điểm SfM)
SUBTAG=${SUBTAG:-1}             # đổi mỗi lần production để không đè/skip nhầm output cũ
ENH_ARCH=${ENH_ARCH:-vgg}       # GATE_B1 17/07: vgg prior thắng L16-unet +0.0052 → mặc định vgg
ZIP=${ZIP:-submission_R2_SUB${SUBTAG}.zip}
say(){ echo; echo "[$(date +%H:%M)] ═════ $* ═════"; }
die(){ echo "❌ $*"; exit 1; }
disk_guard(){ FREE=$(df --output=avail -BG . | tail -1 | tr -dc 0-9); [ "$FREE" -lt 8 ] && die "đĩa ${FREE}GB<8"; }

k1_of(){ $PY - "$1" <<'EOF'
import struct, sys
with open(f"workspace_r2/{sys.argv[1]}/sparse/0/cameras.bin", "rb") as f:
    f.read(8); cid, model, w, h = struct.unpack("<iiQQ", f.read(24))
    if model == 2:
        print(repr(struct.unpack("<dddd", f.read(32))[3]))
    elif model in (0, 1):
        print("0.0")
    else:
        sys.exit(f"model {model} chưa hỗ trợ")
EOF
}
# S1: méo ĐO THẬT per-scene (audit_distortion). USE_TRUE_DIST=1 → dùng thay k1 lưu.
# Đọc distortion_r2.tsv → echo "k1 k2". Chỉ bật SAU khi GATE_S1 xác nhận thắng.
dist_true_of(){ awk -v s="$1" '$1==s{print $2, $3}' tools/distortion_r2.tsv; }

say "0. tiên quyết"
[ -x "$PY" ] || die "thiếu .venv"
$PY -c "import pycolmap; assert hasattr(pycolmap, 'Reconstruction') or hasattr(pycolmap, 'SceneManager')" 2>/dev/null \
  || die ".venv hỏng pycolmap: cần Reconstruction hoặc SceneManager"
for s in $SCENES_HCM $SCENES_OBJ; do
  [ -s "workspace_r2/$s/sparse/0/images.bin" ] || die "thiếu workspace_r2/$s → bash tools/prepare_r2.sh trước"
done
disk_guard
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | sed 's/^/  GPU /'
echo "  ✅ SEEDS=[$SEEDS] CAP_HCM=$CAP_HCM CAP_OBJ=$CAP_OBJ · đĩa ${FREE}GB · GPU $CUDA_VISIBLE_DEVICES"

# ---------- train + render 1 scene × 1 seed ----------
train_render_seed(){   # $1=scene $2=seed $3=cap $4=branch(gut|classic)
  local s=$1 seed=$2 cap=$3 branch=$4
  local res="results/r2_${s}__s${seed}" rend="renders_r2/${s}__s${seed}"
  local UT_TRAIN="" UT_REND="" K1=""
  if [ "$branch" = gut ]; then
    K1=$(k1_of "$s") || die "k1_of $s"
    UT_TRAIN="--with-ut --with-eval3d --raw-distortion"
    UT_REND="--with_ut --radial_k1 $K1"
    if [ "${USE_TRUE_DIST:-0}" = "1" ]; then   # S1: méo đo thật (sau khi GATE_S1 thắng)
      read TK1 TK2 <<<"$(dist_true_of "$s")"
      [ -n "$TK1" ] || die "USE_TRUE_DIST=1 nhưng thiếu $s trong distortion_r2.tsv"
      UT_TRAIN="$UT_TRAIN --dist-k1-override $TK1 --dist-k2-override $TK2"
      UT_REND="--with_ut --radial_k1 $TK1 --radial_k2 $TK2"
      echo "  [S1] $s méo thật k1=$TK1 k2=$TK2"
    fi
  else
    UT_TRAIN="--sh-degree $OBJ_SH_DEGREE"   # O4: SH4 cho scene vật liệu khó nếu gate thắng
  fi
  if ! [ -s "$res/ckpts/ckpt_29999_rank0.pt" ]; then
    disk_guard
    echo "[$(date +%H:%M)] train $s seed=$seed cap=$cap branch=$branch"
    $PY gsplat/examples/simple_trainer.py mcmc --data-dir "workspace_r2/$s" --data-factor 1 \
      --result-dir "$PWD/$res" --max-steps 30000 --test-every 999999 \
      --disable-viewer --antialiased $UT_TRAIN \
      --strategy.cap-max "$cap" --eval-steps 30000 --save-steps 30000 --global-seed "$seed" \
      2>&1 | tee "/tmp/r2_${s}_s${seed}.log" || die "train $s seed$seed — xem /tmp/r2_${s}_s${seed}.log"
    rm -f "$res/ckpts/ckpt_14999_rank0.pt"; rm -rf "$res/videos"
  else echo "  ⏩ $s seed$seed ckpt có"; fi

  # SELFCHECK sau seed ĐẦU TIÊN của scene (round 2 không có GT — đây là lưới an toàn duy nhất)
  if [ "$seed" = "${SEEDS%% *}" ] && ! [ -f "$res/SELFCHECK_OK" ]; then
    $PY tools/r2_selfcheck.py gen --ws "workspace_r2/$s" --n 5 --out "/tmp/r2_sc_${s}.csv" || die "selfcheck gen $s"
    rm -rf "/tmp/r2_sc_${s}_render"
    $PY tools/render_test_poses.py --ckpt "$res/ckpts/ckpt_29999_rank0.pt" \
      --csv "/tmp/r2_sc_${s}.csv" --out "/tmp/r2_sc_${s}_render" --data_dir "workspace_r2/$s" \
      --antialiased $UT_REND 2>&1 | grep -av "render " || die "selfcheck render $s"
    $PY tools/r2_selfcheck.py score --render_dir "/tmp/r2_sc_${s}_render" --ws "workspace_r2/$s" \
      || die "SELFCHECK FAIL $s — transform/nhánh camera SAI, dừng scene này, DÁN LOG CHO CLAUDE"
    touch "$res/SELFCHECK_OK"
  fi

  if [ "$(ls "$rend" 2>/dev/null | wc -l)" -lt 5 ]; then
    $PY tools/render_test_poses.py --ckpt "$res/ckpts/ckpt_29999_rank0.pt" \
      --csv "$R2/$s/test/test_poses.csv" --out "$rend" --data_dir "workspace_r2/$s" \
      --antialiased $UT_REND 2>&1 | tee -a /tmp/r2_render.log || die "render $s seed$seed"
  fi
  N_EXPECT=$(($(wc -l < "$R2/$s/test/test_poses.csv") - 1))
  [ "$(ls "$rend" | wc -l)" -eq "$N_EXPECT" ] || die "$s seed$seed render $(ls "$rend" | wc -l)/$N_EXPECT ảnh"

  # PRUNE_CKPT=1 (opt-in, tiết kiệm đĩa): sau khi render ĐỦ ảnh đã verify ở trên,
  # ckpt của seed KHÔNG-phải-đầu chỉ còn dùng để re-render → xoá được (~1-2GB/seed).
  # Seed đầu GIỮ NGUYÊN (enhancer train từ nó). In rõ từng ckpt xoá.
  if [ "${PRUNE_CKPT:-0}" = "1" ] && [ "$seed" != "${SEEDS%% *}" ]; then
    echo "  🧹 PRUNE_CKPT: xoá $res/ckpts ($(du -sh "$res/ckpts" 2>/dev/null | cut -f1)) — renders đã đủ $N_EXPECT ảnh"
    rm -rf "$res/ckpts"
  fi
}

# ---------- vòng chính ----------
if [ "${PACK_ONLY:-0}" = "1" ]; then
  echo "PACK_ONLY: bỏ train, đóng gói thẳng từ renders_r2/*__sub${SUBTAG}"
  SCENES_HCM="HCM0421 HCM0539 HCM0540 HCM0644 HCM0674"; SCENES_OBJ="bonsai chair"
  for s in $SCENES_OBJ $SCENES_HCM; do
    [ "$(ls "renders_r2/${s}__sub${SUBTAG}" 2>/dev/null | wc -l)" -ge 5 ] \
      || die "PACK_ONLY nhưng thiếu renders_r2/${s}__sub${SUBTAG} — chưa gộp đủ 7 scene (thiếu = LOẠI BÀI)"
  done
fi
[ "${PACK_ONLY:-0}" = "1" ] || for s in $SCENES_OBJ $SCENES_HCM; do
  case " $SCENES_OBJ " in *" $s "*) branch=classic; cap=$CAP_OBJ;; *) branch=gut; cap=$CAP_HCM;; esac
  # cap riêng per-scene qua env CAP_<scene> (vd CAP_bonsai=6000000 — profiler/O-gate quyết)
  eval "override=\${CAP_${s}:-}"; [ -n "$override" ] && cap=$override
  say "SCENE $s ($branch, cap=$cap)"
  [ -d "renders_r2/${s}__sub${SUBTAG}" ] && { echo "  ⏩ $s hoàn tất"; continue; }

  DIRS=""
  for seed in $SEEDS; do
    train_render_seed "$s" "$seed" "$cap" "$branch"
    DIRS="$DIRS renders_r2/${s}__s${seed}"
  done

  N_SEED=$(echo $SEEDS | wc -w)
  ENS="renders_r2/${s}__ens${SUBTAG}"
  if [ "$N_SEED" -gt 1 ]; then
    $PY tools/ensemble.py --dirs $DIRS --out "$ENS" --mode mean >/dev/null || die "ensemble $s"
  else
    rm -rf "$ENS"; cp -r "renders_r2/${s}__s${SEEDS}" "$ENS"
  fi

  # Enhancer per-scene (ENH_ARCH=vgg: B1 restoration prior thắng L16-unet +0.0052, GATE_B 17/07)
  FIRST_SEED=${SEEDS%% *}
  L16_FLAGS=""
  if [ "$branch" = gut ]; then
    if [ "${USE_TRUE_DIST:-0}" = "1" ]; then
      read TK1 TK2 <<<"$(dist_true_of "$s")"
      L16_FLAGS="--with_ut --radial_k1 $TK1 --radial_k2 $TK2"
    else
      L16_FLAGS="--with_ut --radial_k1 $(k1_of "$s")"
    fi
  fi
  NET="results/r2_${s}__enh_${ENH_ARCH}/net.pt"
  if ! [ -s "$NET" ]; then
    $PY tools/enhance_net.py train --workspace "workspace_r2/$s" \
      --ckpt "results/r2_${s}__s${FIRST_SEED}/ckpts/ckpt_29999_rank0.pt" \
      --out "$NET" $L16_FLAGS --arch "$ENH_ARCH" \
      --steps 8000 --ch_mult 2 --patch 320 2>&1 | tee "/tmp/r2_enh_${s}.log" || die "enh train $s"
  fi
  $PY tools/enhance_net.py apply --net "$NET" \
    --in_dir "$ENS" --out_dir "renders_r2/${s}__sub${SUBTAG}" >/dev/null || die "enh apply $s"
  echo "[$(date +%H:%M)] SCENE-OK $s"
done

# chạy chia máy (SCENES_OBJ="" hoặc SCENES_HCM="") → BỎ đóng gói ở máy này,
# gộp renders về 1 máy rồi PACK_ONLY=1 để đóng gói đủ 7 scene (thiếu scene = LOẠI BÀI)
if [ -z "$SCENES_HCM" ] || [ -z "$SCENES_OBJ" ]; then
  echo; echo "⏸ CHẠY MỘT PHẦN (SCENES_HCM='$SCENES_HCM' · SCENES_OBJ='$SCENES_OBJ')"
  echo "  → KHÔNG đóng gói ở đây. Gộp renders_r2/*__sub${SUBTAG} về 1 máy rồi chạy:"
  echo "     PACK_ONLY=1 SUBTAG=$SUBTAG bash tools/run_sub_r2.sh"
  exit 0
fi

if [ "${ENABLE_PACKAGING:-0}" != "1" ]; then
  echo
  echo "✅ TRAIN + SELFCHECK + RENDER + ENSEMBLE + POSTPROCESS hoàn tất."
  echo "⏭ ENABLE_PACKAGING=0 → bỏ qua bước tạo ZIP."
  echo "   Chạy lại với ENABLE_PACKAGING=1 PACK_ONLY=1 để chỉ đóng gói."
  exit 0
fi

say "ĐÓNG GÓI (PNG → JPEG q96 đúng tên CSV → zip)"
$PY - <<EOF || die "repackage"
import csv, cv2, sys
from pathlib import Path
flags = [cv2.IMWRITE_JPEG_QUALITY, 96]; miss = 0
for sd in sorted(Path("$R2").iterdir()):
    if not (sd / "test/test_poses.csv").exists(): continue
    s = sd.name; out = Path("renders_sub_r2") / s; out.mkdir(parents=True, exist_ok=True)
    for r in csv.DictReader(open(sd / "test/test_poses.csv")):
        n = r["image_name"]; p = Path(f"renders_r2/{s}__sub${SUBTAG}") / (Path(n).stem + ".png")
        if not p.exists(): print("THIẾU", s, n); miss += 1; continue
        cv2.imwrite(str(out / n), cv2.imread(str(p)), flags)
print(f"repackage: thiếu {miss}")
sys.exit(1 if miss else 0)
EOF
$PY tools/make_submission.py --data_root "$R2" --renders_root renders_sub_r2 \
  --ext .same --out "$ZIP" || die "make_submission"
ls -la "$ZIP"
echo "[$(date +%d/%m' '%H:%M)] R2-ALL-DONE → $ZIP (nhớ kiểm <350MB và đủ 7 scene ở output trên)"

__RUN_SUB_R2_PAYLOAD__

    chmod +x "$GENERATED_WORKER"
    bash -n "$GENERATED_WORKER"
    echo "[Code] Embedded full pipeline worker: $GENERATED_WORKER"
}

# Positional arguments select a subset of scenes while retaining the original
# HCM/object branch logic.
if (( $# > 0 )); then
    selected_hcm=()
    selected_obj=()

    for scene in "$@"; do
        if [[ "$scene" == HCM* ]]; then
            selected_hcm+=("$scene")
        else
            selected_obj+=("$scene")
        fi
    done

    SCENES_HCM="${selected_hcm[*]}"
    SCENES_OBJ="${selected_obj[*]}"
fi

read -r -a HCM_ARRAY <<< "$SCENES_HCM"
read -r -a OBJ_ARRAY <<< "$SCENES_OBJ"
ALL_SCENES=("${OBJ_ARRAY[@]}" "${HCM_ARRAY[@]}")

(( ${#ALL_SCENES[@]} > 0 )) || die "No scenes selected."

[[ -d "$DATASET_ROOT" ]] || die "Dataset root does not exist: $DATASET_ROOT"

for required_tool in \
    "$PROJECT_DIR/tools/render_test_poses.py" \
    "$PROJECT_DIR/tools/r2_selfcheck.py" \
    "$PROJECT_DIR/tools/ensemble.py" \
    "$PROJECT_DIR/tools/enhance_net.py"
do
    [[ -f "$required_tool" ]] || die "Missing pipeline tool: $required_tool"
done

say "ROUND-2 SETUP"
echo "Script version        : $SCRIPT_VERSION"
echo "Project directory     : $PROJECT_DIR"
echo "AI Tuyen root         : $AI_TUYEN_ROOT"
echo "Dataset root          : $DATASET_ROOT"
echo "gsplat repository     : $GSPLAT_REPO_URL"
echo "gsplat ref            : $GSPLAT_REF"
echo "gsplat directory      : $GSPLAT_DIR"
echo "Install gsplat        : $INSTALL_GSPLAT"
echo "Patch run_sub compat  : $PATCH_GSPLAT_RUN_SUB_COMPAT"
echo "Torch packages        : $TORCH_PACKAGES"
echo "Torch index URL       : ${TORCH_INDEX_URL:-default PyPI}"
echo "Workspace root        : $WORKSPACE_R2_ROOT"
echo "GPU                    : $GPU_ID"
echo "HCM scenes             : $SCENES_HCM"
echo "Object scenes          : $SCENES_OBJ"
echo "Seeds                  : $SEEDS"
echo "HCM cap                : $CAP_HCM"
echo "Object cap             : $CAP_OBJ"
echo "Enhancer               : $ENH_ARCH"
echo "Subtag                 : $SUBTAG"
echo "Packaging              : $ENABLE_PACKAGING"
echo "Setup log              : $SETUP_LOG"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PHYSICAL_GPU_ID="$GPU_ID"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

echo "[Setup order] GPU → disk → packages → CUDA → clone gsplat → build/install → pipeline"
check_gpu
disk_guard
install_system_packages
setup_cuda_toolkit
clone_or_prepare_gsplat
prepare_project_gsplat_link
patch_gsplat_run_sub_compat
setup_python_environment
prepare_r2_alias

for scene in "${ALL_SCENES[@]}"; do
    source="$(find_scene_source "$scene" || true)"
    [[ -n "$source" ]] || die \
        "Scene '$scene' was not found under DATASET_ROOT=$DATASET_ROOT"

    train_root="$(validate_scene "$scene" "$source")"

    echo
    echo "[Data] Scene      : $scene"
    echo "[Data] Source     : $source"
    echo "[Data] Train root : $train_root"
    echo "[Data] Test CSV   : $source/test/test_poses.csv"

    prepare_workspace_scene "$scene" "$train_root"
done

write_embedded_worker

if [[ "$SETUP_ONLY" == "1" ]]; then
    echo
    echo "SETUP_ONLY=1: setup completed; pipeline was not started."
    exit 0
fi

say "START FULL RUN_SUB_R2 PIPELINE"

cd "$PROJECT_DIR"

export SCENES_HCM
export SCENES_OBJ
export SEEDS
export CAP_HCM
export CAP_OBJ
export OBJ_SH_DEGREE
export SUBTAG
export ENH_ARCH
export ENABLE_PACKAGING
export CUDA_VISIBLE_DEVICES="$GPU_ID"

exec bash "$GENERATED_WORKER"
