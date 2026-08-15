#!/bin/bash
# ============================================================
# 投机解码基准（带实验日志，规范同 bench_all.sh）
#
# 用法: bash bench_spec.sh "实验目的" --dual <target> <draft>
#  或: bash bench_spec.sh "实验目的" --self <target>
#
#   例: bash bench_spec.sh "7B 目标 + 0.5B 草稿 双模型投机" --dual models/qwen2.5-7b models/qwen2.5-0.5b
#   例: bash bench_spec.sh "7B 自投机" --self models/qwen2.5-7b
#
# 输出: bench_logs/exp_<时间戳>/  (README.txt 元数据 + bench_spec.log 完整输出)
# ============================================================
PURPOSE="${1:?用法: bench_spec.sh \"目的\" --dual 目标 草稿 | --self 目标}"
shift

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p bench_logs
EXP_DIR="bench_logs/exp_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$EXP_DIR"

GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
NVCC="$(command -v nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)"
CUDA_VER="$($NVCC --version 2>/dev/null | grep -oE 'release [0-9.]+' | awk '{print $2}')"
CMD="./build/bench_spec_decode $*"

{
  echo "############################################################"
  echo "# NanoInfer 实验日志 (speculative decode)"
  echo "# 目的: $PURPOSE"
  echo "# 时间: $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "# GPU:  $GPU"
  echo "# CUDA: $CUDA_VER"
  echo "# 指令: $CMD"
  echo "############################################################"
} | tee "$EXP_DIR/README.txt"

./build/bench_spec_decode "$@" 2>&1 | tee -a "$EXP_DIR/bench_spec.log"
rc=${PIPESTATUS[0]}
echo "# exit=$rc $(date '+%H:%M:%S')" >> "$EXP_DIR/README.txt"

[ -f bench_logs/INDEX.md ] || echo "| 时间 | 实验目录 | 目的 | 参数 |" > bench_logs/INDEX.md
echo "| $(date '+%Y-%m-%d %H:%M') | $(basename "$EXP_DIR") | $PURPOSE | $* |" >> bench_logs/INDEX.md

echo ""
echo "实验完成。日志: $EXP_DIR"
exit $rc
