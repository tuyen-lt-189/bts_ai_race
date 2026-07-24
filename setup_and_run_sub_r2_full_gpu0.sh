#!/usr/bin/env bash
#
# ONE-FILE ROUND-2 SETUP + FULL PIPELINE
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
#   setup → workspace links → train each seed → selfcheck first seed
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

setup_python_environment() {
    if [[ ! -x "$PY" ]]; then
        echo "[Python] Creating venv: $VENV_DIR"
        "$PYTHON_BIN" -m venv "$VENV_DIR"
        FORCE_PIP_SETUP=1
    else
        echo "[Python] Reusing venv: $VENV_DIR"
    fi

    local should_setup=0

    case "$SETUP_PYTHON_ENV" in
        1|true|yes|on)
            should_setup=1
            ;;
        0|false|no|off)
            should_setup=0
            ;;
        auto)
            if [[ "$FORCE_PIP_SETUP" == "1" ]]; then
                should_setup=1
            elif ! (
                cd "$PROJECT_DIR"
                "$PY" -c \
                    "import torch, numpy, cv2, gsplat; from pycolmap import SceneManager"
            ) >/dev/null 2>&1
            then
                should_setup=1
            fi
            ;;
        *)
            die "SETUP_PYTHON_ENV must be auto, true, or false."
            ;;
    esac

    if (( should_setup == 1 )); then
        "$PY" -m pip install --upgrade pip setuptools wheel ninja

        if [[ -f "$PROJECT_DIR/requirements.txt" ]]; then
            echo "[Python] Installing requirements.txt"
            "$PY" -m pip install -r "$PROJECT_DIR/requirements.txt"
        fi

        case "$INSTALL_PROJECT_EDITABLE" in
            1|true|yes|on)
                "$PY" -m pip install -e "$PROJECT_DIR"
                ;;
            auto)
                if [[ -f "$PROJECT_DIR/pyproject.toml" \
                   || -f "$PROJECT_DIR/setup.py" ]]
                then
                    echo "[Python] Installing project in editable mode."
                    "$PY" -m pip install -e "$PROJECT_DIR"
                fi
                ;;
            0|false|no|off)
                ;;
            *)
                die "INSTALL_PROJECT_EDITABLE must be auto, true, or false."
                ;;
        esac

        if [[ -n "$EXTRA_PIP_REQUIREMENTS" ]]; then
            # Intentional splitting: this variable contains pip package specs.
            # shellcheck disable=SC2086
            "$PY" -m pip install $EXTRA_PIP_REQUIREMENTS
        fi
    fi

    (
        cd "$PROJECT_DIR"
        "$PY" - <<'PY_VERIFY'
import cv2
import gsplat
import numpy
import torch
from pycolmap import SceneManager

print("[Python] Python:", __import__("sys").version.split()[0])
print("[Python] torch:", torch.__version__)
print("[Python] torch CUDA:", torch.version.cuda)
print("[Python] CUDA available:", torch.cuda.is_available())
print("[Python] gsplat import: OK")
print("[Python] pycolmap.SceneManager import: OK")
print("[Python] cv2:", cv2.__version__)

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access CUDA.")

if torch.cuda.device_count() != 1:
    raise SystemExit(
        f"Expected one visible CUDA GPU, got {torch.cuda.device_count()}."
    )

print("[Python] logical cuda:0:", torch.cuda.get_device_name(0))
PY_VERIFY
    )
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
$PY -c "from pycolmap import SceneManager" 2>/dev/null || die ".venv hỏng pycolmap rmbrualla (DOC3 §3.7) — chạy GATE_B.sh B2.0 hoặc pip install lại"
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
[[ -f "$PROJECT_DIR/gsplat/examples/simple_trainer.py" ]] || die \
    "Missing trainer: $PROJECT_DIR/gsplat/examples/simple_trainer.py"

for required_tool in \
    "$PROJECT_DIR/tools/render_test_poses.py" \
    "$PROJECT_DIR/tools/r2_selfcheck.py" \
    "$PROJECT_DIR/tools/ensemble.py" \
    "$PROJECT_DIR/tools/enhance_net.py"
do
    [[ -f "$required_tool" ]] || die "Missing pipeline tool: $required_tool"
done

say "ROUND-2 SETUP"
echo "Project directory     : $PROJECT_DIR"
echo "AI Tuyen root         : $AI_TUYEN_ROOT"
echo "Dataset root          : $DATASET_ROOT"
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

check_gpu
disk_guard
install_system_packages
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
