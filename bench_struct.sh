#!/bin/bash
# ============================================================
# 结构化输出基准（带实验日志，规范同 bench_all.sh）
#
# 用法: bash bench_struct.sh "实验目的" [bench_structured 参数...]
#
#   例: bash bench_struct.sh "0.5B/1.5B 结构化输出开销对比" --model models/qwen2.5-1.5b --fp16
#   例: bash bench_struct.sh "3B 嵌套 schema 结构化" --model models/qwen2.5-3b --fp16 --schema '{"type":"array",...}'
#
# 输出: bench_logs/exp_<时间戳>/  (README.txt 元数据 + bench_struct.log 完整输出)
# ============================================================
PURPOSE="${1:?用法: bench_struct.sh \"目的\" [bench_structured 参数...]}"
shift

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p bench_logs
EXP_DIR="bench_logs/exp_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$EXP_DIR"

GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
NVCC="$(command -v nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)"
CUDA_VER="$($NVCC --version 2>/dev/null | grep -oE 'release [0-9.]+' | awk '{print $2}')"
CMD="./build/bench_structured $*"

{
  echo "############################################################"
  echo "# NanoInfer 实验日志 (structured output)"
  echo "# 目的: $PURPOSE"
  echo "# 时间: $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "# GPU:  $GPU"
  echo "# CUDA: $CUDA_VER"
  echo "# 指令: $CMD"
  echo "############################################################"
} | tee "$EXP_DIR/README.txt"

./build/bench_structured "$@" 2>&1 | tee -a "$EXP_DIR/bench_struct.log"
rc=${PIPESTATUS[0]}
echo "# exit=$rc $(date '+%H:%M:%S')" >> "$EXP_DIR/README.txt"

[ -f bench_logs/INDEX.md ] || echo "| 时间 | 实验目录 | 目的 | 参数 |" > bench_logs/INDEX.md
echo "| $(date '+%Y-%m-%d %H:%M') | $(basename "$EXP_DIR") | $PURPOSE | $* |" >> bench_logs/INDEX.md

echo ""
echo "实验完成。日志: $EXP_DIR"
exit $rc
