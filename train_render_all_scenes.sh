#!/usr/bin/env bash
# VERSION: PTY_FIX_V2_2026-07-24
# Fix: preserve Rich training progress through util-linux script pseudo-TTY.
#
# Train + render tuần tự toàn bộ scene của VAI_NVS_DATA_ROUND_2.
#
# Mỗi scene được xử lý theo thứ tự:
#   1. Train bằng setup_and_train_3dgrut_gpu0.sh
#   2. Đọc checkpoint từ latest_run.json
#   3. Render test/test_poses.csv bằng render_3dgrut_gpu0.sh
#   4. Kiểm tra số ảnh render so với số pose trong CSV
#   5. Chuyển sang scene tiếp theo
#
# Cách chạy toàn bộ:
#   chmod +x train_render_all_scenes.sh
#   bash train_render_all_scenes.sh
#
# Chỉ chạy một vài scene:
#   bash train_render_all_scenes.sh bonsai chair
#
# Chạy lại sau khi bị ngắt:
#   bash train_render_all_scenes.sh
# Script sẽ bỏ qua scene đã train/render hoàn chỉnh và resume scene dở dang.

set -Eeuo pipefail
IFS=$'\n\t'

# =============================================================================
# ĐƯỜNG DẪN
# =============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
AI_TUYEN_ROOT="${AI_TUYEN_ROOT:-$HOME/ai-tuyen}"

# Đặt 3 file .sh trong cùng thư mục, hoặc override bằng đường dẫn tuyệt đối.
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$SCRIPT_DIR/setup_and_train_3dgrut_gpu0.sh}"
RENDER_SCRIPT="${RENDER_SCRIPT:-$SCRIPT_DIR/render_3dgrut_gpu0.sh}"

DATASET_ROOT="${DATASET_ROOT:-$AI_TUYEN_ROOT/dataset/VAI_NVS_DATA_ROUND_2}"
WORK_ROOT="${WORK_ROOT:-$AI_TUYEN_ROOT/3dgrut_workspace}"
REPO_DIR="${REPO_DIR:-$WORK_ROOT/3dgrut}"
OUT_ROOT="${OUT_ROOT:-$REPO_DIR/runs}"
LOG_ROOT="${LOG_ROOT:-$WORK_ROOT/logs/round2_batch}"

# =============================================================================
# DANH SÁCH SCENE
# Ưu tiên: positional arguments > biến SCENES > danh sách mặc định.
# =============================================================================

if (( $# > 0 )); then
    SCENE_LIST=("$@")
elif [[ -n "${SCENES:-}" ]]; then
    IFS=' ' read -r -a SCENE_LIST <<< "$SCENES"
else
    SCENE_LIST=(
        bonsai
        chair
        HCM0421
        HCM0539
        HCM0540
        HCM0644
        HCM0674
    )
fi

# =============================================================================
# CẤU HÌNH TRAIN ĐÃ CHỐT
# =============================================================================

GPU_ID="${GPU_ID:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"

TRAIN_PRESET="${TRAIN_PRESET:-ab03_ppisp_nohorizon}"
APPEARANCE_MODE="${APPEARANCE_MODE:-ppisp_raw}"

NHT_FEATURE_DIM="${NHT_FEATURE_DIM:-80}"
NHT_DECODER_HIDDEN_DIM="${NHT_DECODER_HIDDEN_DIM:-128}"
NHT_DECODER_NUM_LAYERS="${NHT_DECODER_NUM_LAYERS:-3}"
NHT_DECODER_LR="${NHT_DECODER_LR:-0.00068}"
NHT_DECODER_EMA_DECAY="${NHT_DECODER_EMA_DECAY:-0.95}"

PARTICLE_FEATURE_HALF="${PARTICLE_FEATURE_HALF:-true}"
FEATURE_OUTPUT_HALF="${FEATURE_OUTPUT_HALF:-true}"

CAP_MAX="${CAP_MAX:-5000000}"
MAX_STEPS="${MAX_STEPS:-35000}"
GEOMETRY_STEPS="${GEOMETRY_STEPS:-25000}"
COLOR_REFINE_STEPS="${COLOR_REFINE_STEPS:-10000}"
SAVE_STEPS="${SAVE_STEPS:-30000,32500,35000}"

# RUN_NAME của từng scene = <scene>_<RUN_TAG>.
# Ví dụ: bonsai_ab12_ppisp_fd80_dec128x3_35k_5m
RUN_TAG="${RUN_TAG:-ab12_ppisp_fd80_dec128x3_35k_5m}"

# true giúp scene đang train dở tiếp tục từ checkpoint gần nhất.
# Nó không thay đổi hyperparameter hay kết quả của run hoàn chỉnh.
AUTO_RESUME="${AUTO_RESUME:-true}"
OVERWRITE_EXPERIMENT="${OVERWRITE_EXPERIMENT:-false}"
INSTALL_SYSTEM_PACKAGES="${INSTALL_SYSTEM_PACKAGES:-0}"
FORCE_ENV_SETUP="${FORCE_ENV_SETUP:-0}"

# =============================================================================
# CẤU HÌNH RENDER TEST POSE
# =============================================================================

CHECKPOINT_STEP="${CHECKPOINT_STEP:-last}"
ENABLE_EVALUATION="${ENABLE_EVALUATION:-false}"
USE_NATIVE_DISTORTION="${USE_NATIVE_DISTORTION:-true}"
USE_FEATURE_DECODER_EMA="${USE_FEATURE_DECODER_EMA:-true}"
OVERWRITE_RENDER="${OVERWRITE_RENDER:-false}"
SAVE_ALPHA="${SAVE_ALPHA:-false}"
MAX_IMAGES="${MAX_IMAGES:-}"

# =============================================================================
# HÀNH VI BATCH
# =============================================================================

SKIP_COMPLETED_TRAIN="${SKIP_COMPLETED_TRAIN:-true}"
SKIP_COMPLETED_RENDER="${SKIP_COMPLETED_RENDER:-true}"
STOP_ON_ERROR="${STOP_ON_ERROR:-true}"

# Script train dùng Rich progress bar, vì vậy stdout phải là một TTY thật.
# Wrapper sẽ tự chạy lại bên trong util-linux `script` để vừa giữ progress
# trực tiếp trên terminal, vừa ghi toàn bộ phiên làm việc vào BATCH_LOG.
BATCH_USE_PTY_LOGGING="${BATCH_USE_PTY_LOGGING:-true}"

# Không cần tạo PTY lồng thêm trong từng script con vì wrapper đã cung cấp TTY.
CHILD_USE_PTY_LOGGING="${CHILD_USE_PTY_LOGGING:-false}"

# =============================================================================
# LOGGING
# =============================================================================

mkdir -p "$OUT_ROOT" "$LOG_ROOT"
TIMESTAMP="${BATCH_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
BATCH_LOG="$LOG_ROOT/train_render_all_${TIMESTAMP}.log"
SUMMARY_FILE="$LOG_ROOT/train_render_all_${TIMESTAMP}.tsv"
CURRENT_SCENE="not-started"

# Preserve Rich/tqdm live progress while still recording a complete terminal log.
# `script` allocates a pseudo-terminal, unlike `tee`, which converts stdout to
# a pipe and causes Rich to disable its live progress display.
if [[ "$BATCH_USE_PTY_LOGGING" == "true" \
   && -z "${_3DGRUT_BATCH_PTY_ACTIVE:-}" \
   && -t 1 \
   && -x "$(command -v script 2>/dev/null || true)" ]]; then
    export _3DGRUT_BATCH_PTY_ACTIVE=1
    export BATCH_TIMESTAMP="$TIMESTAMP"
    printf -v _BATCH_REEXEC_COMMAND '%q ' bash "$0" "$@"
    exec script -q -f -e -c "$_BATCH_REEXEC_COMMAND" "$BATCH_LOG"
fi

# Non-interactive fallback for nohup/CI/redirection. Logs are still written,
# but a dynamically updating progress bar cannot be displayed without a TTY.
if [[ -z "${_3DGRUT_BATCH_PTY_ACTIVE:-}" ]]; then
    exec > >(tee -a "$BATCH_LOG") 2>&1
fi

printf 'scene\ttrain_status\trender_status\tcheckpoint\trender_dir\n' > "$SUMMARY_FILE"

on_error() {
    local exit_code=$?
    echo
    echo "============================================================" >&2
    echo "BATCH FAILED" >&2
    echo "Scene     : $CURRENT_SCENE" >&2
    echo "Exit code : $exit_code" >&2
    echo "Log       : $BATCH_LOG" >&2
    echo "Summary   : $SUMMARY_FILE" >&2
    echo "============================================================" >&2
    exit "$exit_code"
}
trap on_error ERR

on_interrupt() {
    echo
    echo "============================================================" >&2
    echo "BATCH INTERRUPTED AT SCENE: $CURRENT_SCENE" >&2
    echo "Chạy lại cùng lệnh để tiếp tục." >&2
    echo "Scene hoàn chỉnh sẽ được bỏ qua; scene đang dở sẽ AUTO_RESUME." >&2
    echo "============================================================" >&2
}
trap on_interrupt INT TERM

# =============================================================================
# HELPERS
# =============================================================================

is_true() {
    case "${1,,}" in
        1|true|yes|y|on) return 0 ;;
        *) return 1 ;;
    esac
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

section() {
    echo
    echo "============================================================"
    echo "$1"
    echo "============================================================"
}

python_bin() {
    if [[ -x "$REPO_DIR/.venv/bin/python" ]]; then
        printf '%s\n' "$REPO_DIR/.venv/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        command -v python3
    else
        return 1
    fi
}

checkpoint_from_manifest() {
    local manifest="$1"
    local py=""
    py="$(python_bin)" || return 1

    "$py" - "$manifest" <<'PY'
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
try:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    checkpoint = Path(data["checkpoint"]).expanduser().resolve()
except Exception:
    raise SystemExit(1)

if checkpoint.is_file() and checkpoint.stat().st_size > 0:
    print(checkpoint)
else:
    raise SystemExit(1)
PY
}

newest_checkpoint() {
    local run_dir="$1"
    [[ -d "$run_dir" ]] || return 1

    find "$run_dir" -type f \
        \( -name 'ckpt_last.pt' -o -name 'ckpt_*.pt' \) \
        -size +0c -printf '%T@\t%p\n' 2>/dev/null \
        | sort -nr \
        | head -n 1 \
        | cut -f2-
}

resolve_checkpoint() {
    local run_dir="$1"
    local manifest="$run_dir/latest_run.json"
    local checkpoint=""

    if [[ -f "$manifest" ]]; then
        checkpoint="$(checkpoint_from_manifest "$manifest" 2>/dev/null || true)"
    fi

    if [[ -z "$checkpoint" ]]; then
        checkpoint="$(newest_checkpoint "$run_dir" || true)"
    fi

    [[ -n "$checkpoint" && -f "$checkpoint" ]] || return 1
    printf '%s\n' "$checkpoint"
}

csv_pose_count() {
    local csv_path="$1"
    local py=""
    py="$(python_bin)" || return 1

    "$py" - "$csv_path" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open("r", encoding="utf-8-sig", newline="") as handle:
    print(sum(1 for _ in csv.DictReader(handle)))
PY
}

latest_render_dir() {
    local run_dir="$1"
    [[ -d "$run_dir" ]] || return 1

    find "$run_dir" -type d \
        -path '*/custom_test_step*_native_3dgut/renders' \
        -printf '%T@\t%p\n' 2>/dev/null \
        | sort -nr \
        | head -n 1 \
        | cut -f2-
}

rendered_image_count() {
    local render_dir="$1"
    [[ -d "$render_dir" ]] || {
        printf '0\n'
        return 0
    }

    find "$render_dir" -type f \
        \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \
           -o -iname '*.webp' -o -iname '*.bmp' -o -iname '*.tif' \
           -o -iname '*.tiff' \) \
        | wc -l \
        | tr -d '[:space:]'
}

render_is_complete() {
    local run_dir="$1"
    local test_csv="$2"
    local render_dir=""
    local expected="0"
    local actual="0"

    render_dir="$(latest_render_dir "$run_dir" || true)"
    [[ -n "$render_dir" ]] || return 1

    expected="$(csv_pose_count "$test_csv")"
    actual="$(rendered_image_count "$render_dir")"

    [[ "$expected" =~ ^[0-9]+$ && "$actual" =~ ^[0-9]+$ ]] || return 1
    (( expected > 0 && actual >= expected )) || return 1

    printf '%s\n' "$render_dir"
}

append_summary() {
    local scene="$1"
    local train_status="$2"
    local render_status="$3"
    local checkpoint="$4"
    local render_dir="$5"

    printf '%s\t%s\t%s\t%s\t%s\n' \
        "$scene" "$train_status" "$render_status" "$checkpoint" "$render_dir" \
        >> "$SUMMARY_FILE"
}

preflight() {
    [[ -f "$TRAIN_SCRIPT" ]] || die \
        "Không tìm thấy train script: $TRAIN_SCRIPT"
    [[ -f "$RENDER_SCRIPT" ]] || die \
        "Không tìm thấy render script: $RENDER_SCRIPT"
    [[ -d "$DATASET_ROOT" ]] || die \
        "Không tìm thấy dataset root: $DATASET_ROOT"

    local scene=""
    local scene_root=""

    for scene in "${SCENE_LIST[@]}"; do
        scene_root="$DATASET_ROOT/$scene"

        [[ -d "$scene_root/train/images" ]] || die \
            "Thiếu train/images: $scene_root/train/images"
        [[ -d "$scene_root/train/sparse/0" ]] || die \
            "Thiếu train/sparse/0: $scene_root/train/sparse/0"
        [[ -f "$scene_root/train/sparse/0/cameras.bin" ]] || die \
            "Thiếu cameras.bin: $scene_root/train/sparse/0/cameras.bin"
        [[ -f "$scene_root/train/sparse/0/images.bin" ]] || die \
            "Thiếu images.bin: $scene_root/train/sparse/0/images.bin"
        [[ -f "$scene_root/train/sparse/0/points3D.bin" ]] || die \
            "Thiếu points3D.bin: $scene_root/train/sparse/0/points3D.bin"
        [[ -f "$scene_root/test/test_poses.csv" ]] || die \
            "Thiếu test_poses.csv: $scene_root/test/test_poses.csv"
    done
}

run_training() {
    local scene="$1"
    local run_name="$2"
    local scene_root="$DATASET_ROOT/$scene"

    env \
        GPU_ID="$GPU_ID" \
        NUM_WORKERS="$NUM_WORKERS" \
        PUBLIC_SET_ROOT="$DATASET_ROOT" \
        DATASET_SOURCE="$scene_root" \
        WORK_ROOT="$WORK_ROOT" \
        REPO_DIR="$REPO_DIR" \
        OUT_ROOT="$OUT_ROOT" \
        TRAIN_PRESET="$TRAIN_PRESET" \
        APPEARANCE_MODE="$APPEARANCE_MODE" \
        NHT_FEATURE_DIM="$NHT_FEATURE_DIM" \
        NHT_DECODER_HIDDEN_DIM="$NHT_DECODER_HIDDEN_DIM" \
        NHT_DECODER_NUM_LAYERS="$NHT_DECODER_NUM_LAYERS" \
        NHT_DECODER_LR="$NHT_DECODER_LR" \
        NHT_DECODER_EMA_DECAY="$NHT_DECODER_EMA_DECAY" \
        PARTICLE_FEATURE_HALF="$PARTICLE_FEATURE_HALF" \
        FEATURE_OUTPUT_HALF="$FEATURE_OUTPUT_HALF" \
        CAP_MAX="$CAP_MAX" \
        MAX_STEPS="$MAX_STEPS" \
        GEOMETRY_STEPS="$GEOMETRY_STEPS" \
        COLOR_REFINE_STEPS="$COLOR_REFINE_STEPS" \
        SAVE_STEPS="$SAVE_STEPS" \
        RUN_NAME="$run_name" \
        AUTO_RESUME="$AUTO_RESUME" \
        OVERWRITE_EXPERIMENT="$OVERWRITE_EXPERIMENT" \
        INSTALL_SYSTEM_PACKAGES="$INSTALL_SYSTEM_PACKAGES" \
        FORCE_ENV_SETUP="$FORCE_ENV_SETUP" \
        USE_PTY_LOGGING="$CHILD_USE_PTY_LOGGING" \
        bash "$TRAIN_SCRIPT" "$scene"
}

run_rendering() {
    local scene="$1"
    local run_name="$2"
    local checkpoint="$3"
    local scene_root="$DATASET_ROOT/$scene"

    env \
        GPU_ID="$GPU_ID" \
        PUBLIC_SET_ROOT="$DATASET_ROOT" \
        DATASET_SOURCE="$scene_root" \
        WORK_ROOT="$WORK_ROOT" \
        REPO_DIR="$REPO_DIR" \
        OUT_ROOT="$OUT_ROOT" \
        APPEARANCE_MODE="$APPEARANCE_MODE" \
        MAX_STEPS="$MAX_STEPS" \
        CAP_MAX="$CAP_MAX" \
        NHT_FEATURE_DIM="$NHT_FEATURE_DIM" \
        RUN_NAME="$run_name" \
        CHECKPOINT_STEP="$CHECKPOINT_STEP" \
        CHECKPOINT_PATH="$checkpoint" \
        ENABLE_EVALUATION="$ENABLE_EVALUATION" \
        USE_NATIVE_DISTORTION="$USE_NATIVE_DISTORTION" \
        USE_FEATURE_DECODER_EMA="$USE_FEATURE_DECODER_EMA" \
        OVERWRITE_RENDER="$OVERWRITE_RENDER" \
        SAVE_ALPHA="$SAVE_ALPHA" \
        MAX_IMAGES="$MAX_IMAGES" \
        USE_PTY_LOGGING="$CHILD_USE_PTY_LOGGING" \
        bash "$RENDER_SCRIPT" "$scene"
}

# =============================================================================
# EXECUTION
# =============================================================================

preflight

section "VAI NVS ROUND 2 — TRAIN + RENDER TUẦN TỰ"
echo "Thời gian        : $(date --iso-8601=seconds)"
echo "Dataset          : $DATASET_ROOT"
echo "Train script     : $TRAIN_SCRIPT"
echo "Render script    : $RENDER_SCRIPT"
echo "Repository       : $REPO_DIR"
echo "Run root         : $OUT_ROOT"
echo "GPU vật lý       : $GPU_ID"
echo "Preset           : $TRAIN_PRESET"
echo "Appearance       : $APPEARANCE_MODE"
echo "Gaussian cap     : $CAP_MAX"
echo "NHT feature dim  : $NHT_FEATURE_DIM"
echo "NHT decoder      : ${NHT_DECODER_HIDDEN_DIM} x ${NHT_DECODER_NUM_LAYERS}"
echo "Steps            : $MAX_STEPS"
echo "Eval test        : $ENABLE_EVALUATION"
echo "Auto resume      : $AUTO_RESUME"
echo "Batch log        : $BATCH_LOG"
echo "Summary          : $SUMMARY_FILE"
echo "Scenes:"
printf '  - %s\n' "${SCENE_LIST[@]}"

completed_scenes=0
failed_stages=0

for scene in "${SCENE_LIST[@]}"; do
    CURRENT_SCENE="$scene"

    scene_root="$DATASET_ROOT/$scene"
    test_csv="$scene_root/test/test_poses.csv"
    run_name="${scene}_${RUN_TAG}"
    run_dir="$OUT_ROOT/$run_name"

    train_status="pending"
    render_status="pending"
    checkpoint=""
    render_dir=""

    section "SCENE: $scene"
    echo "Run name     : $run_name"
    echo "Train source : $scene_root/train"
    echo "Test CSV     : $test_csv"
    echo "Run dir      : $run_dir"
    echo "Test poses   : $(csv_pose_count "$test_csv")"

    # -------------------------------------------------------------------------
    # TRAIN
    # -------------------------------------------------------------------------

    if is_true "$SKIP_COMPLETED_TRAIN" && [[ -f "$run_dir/latest_run.json" ]]; then
        checkpoint="$(resolve_checkpoint "$run_dir" || true)"
        if [[ -n "$checkpoint" ]]; then
            train_status="skipped-complete"
            echo "[Train] Đã hoàn chỉnh, bỏ qua."
            echo "[Train] Checkpoint: $checkpoint"
        fi
    fi

    if [[ -z "$checkpoint" ]]; then
        echo "[Train] Bắt đầu hoặc resume scene $scene..."

        if run_training "$scene" "$run_name"; then
            train_status="completed"
        else
            train_status="failed"
            render_status="not-run"
            failed_stages=$((failed_stages + 1))
            append_summary "$scene" "$train_status" "$render_status" "" ""

            if is_true "$STOP_ON_ERROR"; then
                die "Train thất bại ở scene: $scene"
            fi

            echo "[Train] Thất bại; tiếp tục scene kế tiếp vì STOP_ON_ERROR=false."
            continue
        fi

        checkpoint="$(resolve_checkpoint "$run_dir" || true)"
        [[ -n "$checkpoint" ]] || die \
            "Train kết thúc nhưng không tìm thấy checkpoint trong: $run_dir"

        echo "[Train] Checkpoint đã chọn: $checkpoint"
    fi

    # -------------------------------------------------------------------------
    # RENDER
    # -------------------------------------------------------------------------

    if is_true "$SKIP_COMPLETED_RENDER"; then
        render_dir="$(render_is_complete "$run_dir" "$test_csv" || true)"
    fi

    if [[ -n "$render_dir" ]]; then
        render_status="skipped-complete"
        echo "[Render] Đủ ảnh, bỏ qua."
        echo "[Render] Output: $render_dir"
    else
        echo "[Render] Bắt đầu render test poses của $scene..."

        if run_rendering "$scene" "$run_name" "$checkpoint"; then
            render_status="completed"
        else
            render_status="failed"
            failed_stages=$((failed_stages + 1))
            append_summary "$scene" "$train_status" "$render_status" "$checkpoint" ""

            if is_true "$STOP_ON_ERROR"; then
                die "Render thất bại ở scene: $scene"
            fi

            echo "[Render] Thất bại; tiếp tục scene kế tiếp vì STOP_ON_ERROR=false."
            continue
        fi

        render_dir="$(render_is_complete "$run_dir" "$test_csv" || true)"
        [[ -n "$render_dir" ]] || die \
            "Render command kết thúc nhưng số ảnh output chưa đủ cho scene: $scene"

        echo "[Render] Hoàn thành: $render_dir"
        echo "[Render] Số ảnh: $(rendered_image_count "$render_dir")"
    fi

    completed_scenes=$((completed_scenes + 1))
    append_summary \
        "$scene" "$train_status" "$render_status" "$checkpoint" "$render_dir"

    echo "[Scene] HOÀN THÀNH: $scene"
    nvidia-smi -i "$GPU_ID" \
        --query-gpu=index,name,memory.used,memory.free \
        --format=csv,noheader 2>/dev/null || true
done

CURRENT_SCENE="finished"

section "BATCH HOÀN THÀNH"
echo "Scene hoàn thành : $completed_scenes / ${#SCENE_LIST[@]}"
echo "Stage thất bại   : $failed_stages"
echo "Run root         : $OUT_ROOT"
echo "Batch log        : $BATCH_LOG"
echo "Summary          : $SUMMARY_FILE"

if (( failed_stages > 0 )); then
    exit 2
fi
