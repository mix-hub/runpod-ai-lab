# Source from the repository root: source scripts/env.sh
export HF_HOME="$PWD/.cache/huggingface"
export TORCH_HOME="$PWD/.cache/torch"
export XDG_CACHE_HOME="$PWD/.cache"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$PWD/data" "$PWD/runs"
