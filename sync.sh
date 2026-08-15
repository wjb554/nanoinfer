#!/bin/bash
# ============================================================
# NanoInfer — 本地 → 远程 GPU 服务器一键同步 + 编译
#
# 用法:
#   bash sync.sh              # 同步代码 + 远程增量编译
#   bash sync.sh --no-build   # 只同步代码，不编译
#   bash sync.sh --help
#
# 依赖:
#   - ssh（已装）; rsync 可选，没装时自动退化为 tar 全量同步
#   - 需先配好 SSH 免密登录（见 README/文档），否则每条命令都输密码
#
# 原理: 把本地工作区除 build/、models/、.git/ 外的内容镜像到服务器。
#   - models/ 是模型权重，体积大且已在服务器上，绝不传（"除模型参数外"）
#   - build/  是平台相关产物，服务器有自己的 ninja build，不覆盖
#   - .git/   是本地 git 元数据，服务器不需要
#   - 子模块改动（如 third_party/xgrammar 本地改的 config.cmake）
#     会随 rsync/tar 一起带过去 —— 这正是不用 git 也能同步本地改动的关键
# ============================================================

# ---------- 配置（改这里） ----------
REMOTE="root@YOUR_AUTODL_HOST"   # SSH 地址（AutoDL 控制台获取）
PORT=YOUR_PORT                   # SSH 端口（AutoDL 控制台获取）
REMOTE_PATH="~/NanoInfer"            # 服务器上的项目路径
ARCH="89"                           # GPU 算力: 89=4090, 80=A100, 90=H800, 86=3090, 75=2060
CUDA_PATH="/usr/local/cuda"         # 服务器 CUDA 工具链路径（nvcc 所在目录的上级）
# -----------------------------------

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------- 参数 ----------
DO_BUILD=1
case "${1:-}" in
  --no-build) DO_BUILD=0 ;;
  --help|-h)
    sed -n '1,30p' "$0" | sed 's/^# \?//'
    exit 0 ;;
esac

if [ -z "$REMOTE" ] || [ "$REMOTE" = "root@你的服务器地址" ]; then
  echo "错误: 请先编辑 sync.sh 顶部的 REMOTE / PORT / REMOTE_PATH" >&2
  exit 1
fi

# ---------- 同步 ----------
EXCLUDES=(
  --exclude=build
  --exclude=models
  --exclude=.git
  --exclude=.vscode
  --exclude=.codegraph
  --exclude=.claude
  --exclude=__pycache__
  --exclude='*.pyc'
  --exclude='*.bat'
  --exclude=build_stdout.txt
  --exclude=build_stderr.txt
  --exclude=concurrency_result.txt
)

echo "==> 同步 ${SCRIPT_DIR} → ${REMOTE}:${REMOTE_PATH}"
if command -v rsync >/dev/null 2>&1; then
  echo "    [rsync 增量同步]"
  rsync -avz --checksum -e "ssh -p ${PORT}" "${EXCLUDES[@]}" \
    ./ "${REMOTE}:${REMOTE_PATH}/"
else
  echo "    [tar 全量同步（未检测到 rsync，可装 rsync 提速）]"
  tar czf - "${EXCLUDES[@]}" . | \
    ssh -p "${PORT}" "${REMOTE}" \
      "mkdir -p ${REMOTE_PATH} && tar xzf - -C ${REMOTE_PATH}"
fi

# ---------- 远程编译 ----------
if [ "$DO_BUILD" -eq 0 ]; then
  echo "==> 完成（--no-build，跳过编译）"
  exit 0
fi

echo "==> 远程编译 (ARCH=${ARCH})"
ssh -p "${PORT}" "${REMOTE}" "
  set -e
  export PATH=${CUDA_PATH}/bin:\$PATH
  export CUDA_HOME=${CUDA_PATH}
  cd ${REMOTE_PATH}
  if [ -f build/build.ninja ]; then
    echo '    [增量] ninja'
    cd build && ninja -j\$(nproc)
  else
    echo '    [全量] build_linux.sh ${ARCH}'
    rm -rf build
    bash build_linux.sh ${ARCH}
  fi
"

echo "==> 完成 ✓"
