#!/bin/bash
# ============================================================
# NanoInfer 跨模型吞吐基准（带完整实验日志）
#
# 用法:
#   bash bench_all.sh [duration_sec] [arrival_rate] [max_batch_tokens] ["实验目的"] [模型列表] [kv_cache_mb]
#
#   例:
#     bash bench_all.sh 60 20 0 "固定负载 rate=20 跨模型吞吐对比 (fp16)"
#     bash bench_all.sh 30 50 0 "0.5B 高负载压测" 0.5b
#     bash bench_all.sh 60 20 4096 "全部模型 batch=4096 手动覆盖" "0.5b 1.5b"
#     bash bench_all.sh 60 50 0 "rate=50 加大 KV 池(10GB) 避免耗尽" "" 10000
#
# 参数:
#   duration_sec    模拟时长(秒), 默认 60
#   arrival_rate    每秒到达请求数, 默认 1.0
#   max_batch_tokens 批量 token 预算, 0=按显存自动推导, 默认 0
#   实验目的         记录在日志里, 建议每次填写
#   模型列表         空格分隔, 默认 "0.5b 1.5b 3b 7b"
#   kv_cache_mb     KV 池预算(MB), 0=自动(seq-len 上限), >0=显式指定(防高并发耗尽), 默认 0
#
# 输出: bench_logs/exp_<时间戳>/
#   README.txt       实验元数据(目的/指令/环境/关键指标汇总)
#   bench_<model>.log 各模型完整输出(含目的+指令头)
#   nohup 场景下同时追加索引到 bench_logs/INDEX.md
# ============================================================
DUR="${1:-60}"
RATE="${2:-1.0}"
MBT="${3:-0}"
PURPOSE="${4:-（未指定目的）}"
MODELS="${5:-0.5b 1.5b 3b 7b}"
KVMB="${6:-0}"

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p bench_logs

# ---- 本次实验目录（时间戳，互不覆盖） ----
EXP_DIR="bench_logs/exp_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$EXP_DIR"

# ---- 环境信息 ----
GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
NVCC="$(command -v nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)"
CUDA_VER="$($NVCC --version 2>/dev/null | grep -oE 'release [0-9.]+' | awk '{print $2}')"
[ "$MBT" -gt 0 ] 2>/dev/null && MBT_FLAG="--max-batch-tokens $MBT" || MBT_FLAG=""
[ "$KVMB" -gt 0 ] 2>/dev/null && KV_FLAG="--kv-cache-mb $KVMB" || KV_FLAG=""

# ---- README.txt：实验元数据 ----
{
  echo "############################################################"
  echo "# NanoInfer 实验日志"
  echo "# 目的: $PURPOSE"
  echo "# 时间: $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "# GPU:  $GPU"
  echo "# CUDA: $CUDA_VER"
  echo "# 参数: duration=${DUR}s arrival=${RATE} batch=${MBT:-auto} kvmb=${KVMB:-auto} precision=fp16"
  echo "# 模型: $MODELS"
  echo "############################################################"
} | tee "$EXP_DIR/README.txt"

# ---- 跑每个模型 ----
for m in $MODELS; do
  log="$EXP_DIR/bench_${m}.log"
  CMD="./build/bench_throughput $DUR $RATE --model models/qwen2.5-$m --fp16 $MBT_FLAG $KV_FLAG"
  {
    echo "########## $m ##########"
    echo "# 目的: $PURPOSE"
    echo "# 指令: $CMD"
    echo "# 时间: $(date '+%H:%M:%S')"
  } | tee -a "$log"
  # 实际执行（保持与日志中的指令一致）
  ./build/bench_throughput "$DUR" "$RATE" --model "models/qwen2.5-$m" --fp16 $MBT_FLAG $KV_FLAG >> "$log" 2>&1
  rc=$?
  echo "# exit=$rc $(date '+%H:%M:%S')" >> "$log"
  echo "  [done] $m (exit=$rc)" | tee -a "$EXP_DIR/README.txt"
done

# ---- 关键指标汇总 ----
{
  echo ""
  echo "===== 关键指标 ====="
  for m in $MODELS; do
    echo "--- $m ---"
    grep -E "Throughput:|^  TTFT|^  TPOT|^  Total" "$EXP_DIR/bench_${m}.log" | tail -4
  done
} | tee -a "$EXP_DIR/README.txt"

# ---- 追加到实验索引 ----
[ -f bench_logs/INDEX.md ] || echo "| 时间 | 实验目录 | 目的 | 参数 |" > bench_logs/INDEX.md
echo "| $(date '+%Y-%m-%d %H:%M') | $(basename "$EXP_DIR") | $PURPOSE | dur=${DUR} rate=${RATE} batch=${MBT:-auto} |" >> bench_logs/INDEX.md

echo ""
echo "实验完成。日志目录: $EXP_DIR"
echo "索引: bench_logs/INDEX.md"
