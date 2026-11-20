#!/usr/bin/env bash
# Archive HEAD, version the private Kaggle source dataset, push the kernel,
# poll until it finishes, and download outputs to artifacts/kaggle/<ts>/.
# Kernel and dataset ids come from kaggle/kernel-metadata.json so this script
# is portable across repos.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

KAGGLE_BIN="${KAGGLE_BIN:-$HOME/.local/bin/kaggle}"
META="kaggle/kernel-metadata.json"
KERNEL_ID="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["id"])' "$META")"
SRC_DATASET="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["dataset_sources"][0])' "$META")"
SRC_TITLE="$(basename "$SRC_DATASET")"
TS="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="artifacts/kaggle/${TS}"

SHA="$(git rev-parse HEAD)"
if [ -n "$(git status --porcelain)" ]; then
  echo "warning: working tree is dirty; the dataset reflects HEAD ($SHA) only" >&2
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

mkdir -p "$TMP/ds"
git archive HEAD | tar -x -C "$TMP/ds"

cat > "$TMP/ds/dataset-metadata.json" <<EOF
{"title": "${SRC_TITLE}", "id": "${SRC_DATASET}", "licenses": [{"name": "other"}]}
EOF
cat > "$TMP/ds/run_info.json" <<EOF
{"git_sha": "${SHA}", "created_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
EOF

if "$KAGGLE_BIN" datasets status "$SRC_DATASET" >/dev/null 2>&1; then
  echo "versioning existing dataset $SRC_DATASET"
  "$KAGGLE_BIN" datasets version -p "$TMP/ds" -m "src ${SHA}" -r tar -q
else
  echo "creating dataset $SRC_DATASET"
  "$KAGGLE_BIN" datasets create -p "$TMP/ds" -r tar -q
fi

# Wait for the dataset version to finish processing.
DATASET_READY=0
for _ in $(seq 1 60); do
  STATUS="$("$KAGGLE_BIN" datasets status "$SRC_DATASET" 2>/dev/null || echo unknown)"
  echo "dataset status: $STATUS"
  case "$STATUS" in
    *ready*|*complete*) DATASET_READY=1; break ;;
  esac
  sleep 10
done
if [ "$DATASET_READY" -ne 1 ]; then
  echo "error: source dataset $SRC_DATASET did not become ready" >&2
  exit 1
fi

echo "pushing kernel $KERNEL_ID"
"$KAGGLE_BIN" kernels push -p kaggle

# Poll kernel status until a terminal state; bounded by the loop count.
TERMINAL=""
for _ in $(seq 1 240); do
  STATUS="$("$KAGGLE_BIN" kernels status "$KERNEL_ID" 2>&1 || echo "status-poll-error")"
  echo "kernel status: $STATUS"
  case "$STATUS" in
    *COMPLETE*|*ERROR*|*CANCEL*) TERMINAL="$STATUS"; break ;;
  esac
  sleep 30
done

mkdir -p "$OUT_DIR"
echo "downloading kernel output to $OUT_DIR"
"$KAGGLE_BIN" kernels output "$KERNEL_ID" -p "$OUT_DIR"

# Guard against the dataset-version race: the run must reflect HEAD.
MANIFEST="$OUT_DIR/benchmark/manifest.json"
if [ -f "$MANIFEST" ]; then
  RUN_SHA="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("git_sha",""))' "$MANIFEST")"
  if [ "$RUN_SHA" != "$SHA" ]; then
    echo "error: run manifest git_sha=$RUN_SHA does not match HEAD $SHA" >&2
    exit 1
  fi
fi

case "$TERMINAL" in
  *COMPLETE*) ;;
  *)
    echo "error: kernel did not complete (status: ${TERMINAL:-poll-timeout})" >&2
    exit 1
    ;;
esac

echo "done: kernel=$KERNEL_ID sha=$SHA out=$OUT_DIR"
