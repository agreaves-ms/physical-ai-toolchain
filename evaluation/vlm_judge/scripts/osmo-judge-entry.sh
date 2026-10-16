#!/bin/bash
# OSMO entry for the VLM judge. Base64-injected into the workflow by submit-osmo-judge.sh and decoded by the
# task bootstrap. Installs the judge's locked runtime, judges a LeRobot dataset or a directory of rollout videos,
# checks every verdict, and writes the JSONL to the task output.
set -euo pipefail

: "${JUDGE_INPUT_DIR:?JUDGE_INPUT_DIR is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"
JUDGE_BACKEND="${JUDGE_BACKEND:-qwen3-vl}"
JUDGE_MODEL_ID="${JUDGE_MODEL_ID:-Qwen/Qwen3-VL-4B-Instruct}"
JUDGE_N_FRAMES="${JUDGE_N_FRAMES:-12}"
PAYLOAD_ROOT="${PAYLOAD_ROOT:-/workspace/judge_payload}"
# The workflow carries the instruction base64-encoded so quotes and backslashes survive its template.
JUDGE_INSTRUCTION="$(printf '%s' "${JUDGE_INSTRUCTION_B64:-}" | base64 -d)"

echo "=== VLM Judge ==="
echo "Input: ${JUDGE_INPUT_DIR}"
echo "Backend: ${JUDGE_BACKEND} (${JUDGE_MODEL_ID})"
echo "Frames per episode: ${JUDGE_N_FRAMES}"

# A LeRobot dataset brings its own episodes and instructions; anything else is a directory of rollout videos,
# which carry no task. Decide before the runtime install so a missing instruction fails fast.
if [[ -f "${JUDGE_INPUT_DIR}/meta/info.json" ]]; then
  input_kind=lerobot
elif [[ -n "${JUDGE_INSTRUCTION:-}" ]]; then
  input_kind=rollouts
else
  echo "ERROR: ${JUDGE_INPUT_DIR} is not a LeRobot dataset, and rollout videos need an instruction to judge against" >&2
  exit 1
fi

case "${JUDGE_BACKEND}" in
  qwen3-vl) judge_extras=(--extra qwen3-vl) ;;
  openai-compat) judge_extras=(--extra openai) ;;
  echo) judge_extras=() ;;
  *) echo "ERROR: unknown JUDGE_BACKEND ${JUDGE_BACKEND} (use qwen3-vl, openai-compat, or echo)" >&2; exit 1 ;;
esac

# The judge needs Python 3.12; uv downloads it, and the locked wheels carry their own CUDA libraries.
pip install --quiet --break-system-packages uv==0.7.12
uv python install 3.12
uv venv --python 3.12 /opt/judge-venv
# shellcheck disable=SC1091
source /opt/judge-venv/bin/activate

# The code archive arrives through the OSMO url: input; the standard library unpacks it.
mkdir -p "${PAYLOAD_ROOT}"
ARCHIVE_PATH="$(find "${OSMO_INPUT_0}" -maxdepth 2 -name '*.zip' | head -n1)"
if [[ -z "${ARCHIVE_PATH}" || ! -s "${ARCHIVE_PATH}" ]]; then
  echo "ERROR: no code archive found under the input mount; runtime payload is required." >&2
  exit 1
fi
python3 -m zipfile -e "${ARCHIVE_PATH}" "${PAYLOAD_ROOT}"
export PYTHONPATH="${PAYLOAD_ROOT}:${PYTHONPATH:-}"

JUDGE_PROJECT="${PAYLOAD_ROOT}/evaluation/vlm_judge"
if [[ ! -f "${JUDGE_PROJECT}/uv.lock" ]]; then
  echo "ERROR: judge lockfile not found at ${JUDGE_PROJECT}/uv.lock" >&2
  exit 1
fi
uv sync --active --frozen --no-config --no-install-project --project "${JUDGE_PROJECT}" "${judge_extras[@]}"

mkdir -p "${OUTPUT_DIR}"
JUDGMENTS="${OUTPUT_DIR}/judgments.jsonl"
judge_args=(
  --output "${JUDGMENTS}"
  --backend "${JUDGE_BACKEND}"
  --model-id "${JUDGE_MODEL_ID}"
  --n-frames "${JUDGE_N_FRAMES}"
  --cache-dir ""
)
[[ -n "${JUDGE_MODEL_REVISION:-}" ]] && judge_args+=(--model-revision "${JUDGE_MODEL_REVISION}")
[[ -n "${JUDGE_BASE_URL:-}" ]] && judge_args+=(--base-url "${JUDGE_BASE_URL}")
[[ -n "${JUDGE_LIMIT:-}" ]] && judge_args+=(--limit "${JUDGE_LIMIT}")

if [[ "${input_kind}" == "lerobot" ]]; then
  echo "Judging LeRobot episodes in ${JUDGE_INPUT_DIR}"
  # The attached form keeps argparse from reading an instruction that starts with "-" as another option.
  [[ -n "${JUDGE_INSTRUCTION:-}" ]] && judge_args+=("--instruction=${JUDGE_INSTRUCTION}")
  [[ -n "${JUDGE_INDICES:-}" ]] && judge_args+=(--indices "${JUDGE_INDICES}")
  python -m evaluation.vlm_judge.run --dataset "${JUDGE_INPUT_DIR}" "${judge_args[@]}"
else
  echo "Judging rollout videos under ${JUDGE_INPUT_DIR}"
  python -m evaluation.vlm_judge.policy_eval --rollout-root "${JUDGE_INPUT_DIR}" \
    "--instruction=${JUDGE_INSTRUCTION}" "${judge_args[@]}"
fi

# Every verdict must carry an outcome, a confidence, and per-frame progress.
python - "${JUDGMENTS}" << 'CHECK_VERDICTS'
import json
import math
import sys

records = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
if not records:
    sys.exit("ERROR: the judge wrote no verdicts")
for record in records:
    confidence = record.get("outcome_confidence")
    progress = record.get("progress_per_frame")
    if (
        not isinstance(record.get("outcome_success"), bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not isinstance(progress, list)
        or not progress
    ):
        sys.exit(f"ERROR: incomplete verdict for {record.get('episode_id')}: {record}")
successes = sum(record["outcome_success"] for record in records)
mean_confidence = sum(record["outcome_confidence"] for record in records) / len(records)
print(f"JUDGE_RESULT=episodes={len(records)} successes={successes} mean_confidence={mean_confidence:.3f}")
for record in records:
    print(f"  {record['episode_id']}: success={record['outcome_success']} confidence={record['outcome_confidence']:.3f}")
CHECK_VERDICTS
echo "=== VLM Judge Complete: ${JUDGMENTS} ==="
