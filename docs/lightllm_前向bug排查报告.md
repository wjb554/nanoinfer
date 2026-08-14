# lightllm 推理前向 Bug 排查与修复报告

> 日期：2026-08-13 ~ 2026-08-14
> 范围：lightllm 引擎（C++ + CUDA，手写 LLM 推理）在逐模型复现 Qwen2.5（0.5B/1.5B/3B/7B-Instruct）过程中的 5 个前向正确性 bug。
> 结论：全部 5 个 bug 已修复，4 个模型均输出 " Paris" 与 transformers 对拍一致。

---

## 0. 背景

lightllm 是手写的 C++/CUDA LLM 推理引擎（PagedAttention + 前缀缓存 + Continuous Batching）。目标是**逐 token 复现 HuggingFace transformers 的输出**。

排查由"所有模型输出垃圾"的现象触发。此前曾误判为权重损坏，重下后仍复现。通过逐层 numpy 参考对拍，最终定位到 5 个**相互独立**的 bug：

| # | Bug | 影响 | 状态 |
|---|-----|------|------|
| 1 | q/k/v 投影 bias 缺失 | 所有 Qwen2 模型 logits 完全错乱（最直接原因） | ✅ 已修 |
| 2 | RoPE kernel 线程映射错误 | 多 token（pos>0）时 q/k 大多未旋转 | ✅ 已修 |
| 3 | 最终 RMSNorm 双重归一化 | 首 token logits 错误 | ✅ 已修 |
| 4 | first-prefill attention smem 超 48KB | hd=128 模型（1.5B/7B）kernel 启动失败 → 输出全 0 | ✅ 已修 |
| 5 | untied lm_head 未加载（7B 专属） | 7B 最终 logits 用错矩阵 → 自信预测垃圾 | ✅ 已修 |

**关键洞察**：前 4 个 bug 只影响特定的模型配置（如 1.5B hd=128），0.5B 因参数巧合"碰巧"正确，掩盖了问题。第 5 个 bug 只有 7B（`tie_word_embeddings: false`）能暴露。

---

## 1. 排查方法论（可复用）

垃圾输出在生成阶段，但根因可能在任何一层。采用**分层隔离 + 独立参考实现**：

```
transformers (llmref env)  →  纯 numpy 逐层参考（float64）  →  C++ 引擎 DEBUG dump
```

- **单 token (pos 0) 测试**：RoPE 恒等（`cos(0)=1, sin(0)=0`）、attention 退化为 `=V`（softmax 单元素）。能快速验证 embedding / 归一化 / 投影 / 残差链，但**不覆盖**多位置 RoPE 与因果 attention。
- **多 token 完整前向**：`ref_7b_multi.py` 用正确 HF token `[785,6722,315,9625,374]`（"The capital of France is"）跑 28 层 RoPE + 因果 GQA + SwiGLU。
- **对比锚点**：引擎 `DEBUG L0 q/attn/h`、`L27 h`、`logits top5` 打印 vs numpy 逐位对拍（fp16 舍入内一致即正确）。

**⚠️ 陷阱**：numpy 参考若复用了引擎的**同一个错误假设**（如 #5 用 `embed.T` 当 lm_head），则两者会"一致地错"——**参考实现必须独立于引擎**，且最终要用 transformers 或权威权重哈希兜底。

---

## 2. Bug #1：q/k/v 投影 bias 缺失（最直接原因）

### 现象
所有 Qwen2 模型 logits 完全错乱，输出全垃圾。

### 定位
权重加载 / embedding / RMSNorm / QK 投影 / O-proj 权重 / 注意力公式全部对拍正确后，聚焦到 q/k/v 本身。检查权重发现 Qwen2 的 q/k/v 投影**有 bias**（q 最大 79，k 最大 130，v 0.1），而引擎只算 `hn @ W^T` 未加 bias。

> Qwen2 与 LLaMA 的关键差异：LLaMA 系去掉 q/k/v bias，**Qwen2 保留**。漏掉 bias 后 q/k/v 完全错位。

### 修复
- `EngineLayerW` 增加 `qb/kb/vb`（F32 存储——GEMM 输出恒为 F32）。
- 新增 `ops::add_bias_inplace`（广播加）。
- `step()` / `run_layers` 的 GEMM 后调用。

### 验证
引擎输出 " Paris. This was the site of the great capital of France"，首 token `12095=" Paris"` 与 transformers 一致。

---

## 3. Bug #2：RoPE kernel 线程映射错误

### 现象
多 token（pos>0）时 q/k 大多未旋转，仅位置 0 的 token 正确。位置 0 时 RoPE 恒等**掩盖了 bug**——这也是为什么单 token 测试测不出来。

### 定位
`src/ops/rope.cu` 原 kernel 用 `tid = blockIdx*blockDim + threadIdx` 推导 `(token, head)`，但元素索引 `i = threadIdx.x` → 每个 `(token, head)` 只旋转 1 个元素，其余 31/32 不转。

### 修复
```cuda
int token_idx = blockIdx.x / num_heads;
int head_idx  = blockIdx.x % num_heads;
// threadIdx.x 索引元素
```
每个 block 负责一个 `(token, head)`。

### 验证
旋转后 q 与 numpy 逐位一致。

### 连带影响
dual 投机解码此前"86-100% 接受率"是**垃圾匹配垃圾**——草稿和 target 共用同一个坏 kernel，错得一样。修复后接受率数据作废，需重新测量。

---

## 4. Bug #3：最终 RMSNorm 双重归一化

### 现象
pre-sample 阶段首 token logits 错误（但 debug dump 与 numpy 对拍时因各层都错而难以定位）。

### 定位
`engine_server.cpp` 里 pre-sample 的 `next_hidden_state` 在 `h = rms_norm(h, final_norm_)` **之后**从 post-norm h 拷贝，随后又 norm 一次 → 双重归一化。

### 修复
prefill 的 last-token **pre-norm** hidden 在 norm 前捕获（`pre_norm_hiddens` 块），移除 post-norm 拷贝。

---

## 5. Bug #4：first-prefill attention smem 超 48KB

### 现象
hd=128 的模型（1.5B/7B）输出全 0。0.5B（hd=64）正常——再次被参数巧合掩盖。

### 定位
`paged_attention.cu` 的 `first_prefill_attn` kernel 动态共享内存超过 sm_75/89 默认 48KB 上限，kernel 启动失败（不报错，只是不执行 → 输出全 0）。

### 修复
```cuda
cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 64KB);
```
显式开启 64KB opt-in（sm_75 上限 64KB）。

---

## 6. Bug #5：untied lm_head 未加载（7B 专属）

### 现象
7B 对所有 prompt 自信预测垃圾：单 token 785("The") → `119712="钊"`（汉字）；正确 prompt "The capital of France is" → `86964="dcc"`（softmax 概率 ≈ 1.0）。而 1.5B/3B 同一输入 → `2701=" following"`（合理）。

### 权重真实性验证（先排除"重下损坏"）
1. 4 分片大小与 HF API **逐字节一致**（3945441440/3864726352/3864726424/3556377672）。
2. 本地 sha256 与 **ModelScope 官方** 一致（a1333e..=00001 等）→ 权重是真 Qwen2.5-7B-Instruct。
3. ⚠️ 教训：手写 safetensors 偏移校验曾误报"截断"——脚本 bug（把 `begin+end` 当越界判断）。实际最后 tensor end 恰等于 EOF。

### 前向对拍
- 单 token 785：L0 q/attn/h、L27 h、top5 logits 与 numpy **全部一致**（fp16 舍入内）→ 引擎忠实于权重，**前向本身没 bug**。
- 多 token `ref_7b_multi.py`（5 正确 token）**也用 `embed.T`** → top1=86964="dcc"，与引擎一致 → **numpy 参考和引擎共享同一错误假设**。
- 换成 `lm_head.weight` 后：top1=12095=" Paris"（"dcc" 从第 1 掉到第 74,678 名）。

### 根因
Qwen2.5-**7B `tie_word_embeddings: false`，有独立的 `lm_head.weight`**（~152064×3584，在 model-00004 分片）。而 0.5B/1.5B/3B 都是 `true`（共享 embed_tokens 当 lm_head）。引擎此前**无条件用 `embed_w_` 当 lm_head** → 7B 最终 logits 投影错矩阵。

### 修复
1. `EngineServer` / `InferenceEngine` 增加 `Tensor lm_head_w_` 字段 + helper：
   ```cpp
   const Tensor& lm_head_weight() const {
       return lm_head_w_.defined() ? lm_head_w_ : embed_w_;
   }
   ```
2. 构造时 `if (!cfg_.tie_word_embeddings) lm_head_w_ = load_weight(loader, "lm_head.weight", use_fp16);`
3. 全部 `lm_head_logits(..., embed_w_)` 调用点改为 `lm_head_weight()`：
   - `engine_server.cpp`：投机验收循环 / pre-sample / 常规 decode
   - `engine_paged.cpp`：InferenceEngine decode
   - `draft_engine.cpp`：self-spec target / dual draft 的 lm_head
   - `speculative_decoder.cpp`：target lm_head
4. KV 池显存预算 `add_w(lm_head_w_)`（untied 模型多 ~1GB）。

### 验证
| 模型 | 修复前 | 修复后 |
|------|--------|--------|
| 0.5B | " Paris. But as it was in the" | 无回归 |
| 1.5B | " Paris also" | 无回归 |
| 3B | " Paris Paris" | 无回归 |
| **7B** | "dcc"useiator.compileographs..." | **" Paris __"** |

---

## 6.5 InferenceEngine（engine_paged.cpp）的并行缺陷（2026-08-14 补修）

`EngineServer` 是主引擎（quick_generate / bench / server 全用它），修复后 4 模型全对。但 `InferenceEngine`（engine_paged.cpp，旧单请求 API，被 test_engine_paged / bench_concurrency 使用）是**并行实现，前 5 个 bug 只在 EngineServer 修了**，它一个都没修，且有自己的额外 bug。用 `ie_generate` 工具逐层 numpy 对拍，又找到并修复 4 个：

> **后续（同一天）**：修完 InferenceEngine 后，为避免"两套前向实现"再次分裂，将其**整体迁移删除**——所有调用方改用 EngineServer + BatchMainLoop，`engine_paged.cpp` / `engine.cpp`（死代码）/ `ie_generate.cpp` / `InferenceEngine` 类全部移除。EngineServer 成为唯一前向实现。详见迁移后的 git 记录。

| # | Bug | 定位方法 |
|---|-----|---------|
| IE-1 | **q/k/v bias 缺失**（同 Bug #1） | 与 EngineServer 同样缺，`LayerW` 无 qb/kb/vb 字段 |
| IE-2 | **prefill softmax sum 被 32x 放大** | attention 输出比 numpy 小 32x；`attn_softmax_kernel` 跨 warp sum 归约 bug——所有 lane 已持有总和后，又跑 `shfl_down` 累加，把总和乘了 32 次。修复：去掉冗余 warp reduction |
| IE-3 | **decode 双重归一化** | 循环末尾 `last_h = h_dec`，而 `h_dec` 已被 final norm 覆盖 → 下一步再 norm。修复：用独立 tensor 算 logits，`h_dec` 保持 pre-norm |
| IE-4 | **decode 输入用错（位置语义错误）** | 原代码把上一个 token 的 L28 输出当 layer-0 输入（层循环期望 embedding）；改成 embed 后首步又重嵌 prompt.back() 导致 off-by-one 位置。修复：**首 token 直接从 `lm_head(final_norm(h[P-1]))` 预测（不过层）**，decode 循环才过层处理已预测的 token |
| IE-5 | **prefill attention 非因果**（潜藏） | `attn_qk_kernel` 无 causal mask。虽不影响最后 token 预测（无未来 key 可泄漏），但中间 token hidden 错误。已加 `s<=t` mask |

**验证**：0.5B/1.5B/3B 首 token 全部 12095=" Paris"（numpy 对拍一致）。⚠️ **InferenceEngine 全 FP32 加载，7B 权重 30GB 会 OOM**——7B 只能用 EngineServer（--fp16）。

## 7. 架构事实速查（Qwen2.5 家族）

| 模型 | Hq | Hkv | hd | hidden | inter | vocab | tie_word_embeddings |
|------|-----|-----|-----|--------|-------|-------|---------------------|
| 0.5B | 14 | 2 | 64 | 896 | 4864 | 151936 | true |
| 1.5B | 12 | 2 | 128 | 1536 | 8960 | 151936 | true |
| 3B | 16 | 2 | 128 | 2048 | 11008 | 151936 | true |
| 7B | 28 | 4 | 128 | 3584 | 18944 | 152064 | **false** |

- tokenizer.json 全家族共享（vocab 151643，md5 相同），不依赖模型 vocab_size。
- Qwen2 保留 q/k/v bias（LLaMA 去掉了）。
- RoPE 是 HF `rotate_half` 语义：head 切成前后两半交叉旋转，非相邻对。
- GQA kv 头映射：`kv_head = q_head / (Hq/Hkv)`（7B: groups=7）。
- 字节级 tokenizer：C++ 编码 "The capital of France is" = 14 token（子最优但文本正确），HF = 5 token `[785,6722,315,9625,374]`。子最优切分不影响 1.5B/3B 出 " Paris"，但仍是短板。

---

## 8. 调试工具（可复用）

| 工具 | 用途 |
|------|------|
| `tools/quick_generate` | `--model` `--fp16` `--prompt` `--token N`（单 token 喂入，绕过 tokenizer）`--tokens a,b,c`（直接喂 ID）。⚠️ 不解析 `--greedy`。 |
| `ref_layer0.py` | 0.5B 单 token 层 0 参考 |
| `ref_layer0_multi.py` | 0.5B 多 token 层 0 参考（RoPE + 因果 attention） |
| `ref_7b_multi.py` / `ref_7b_rank.py` | 7B 完整 28 层多 token 参考 / top-k 排名诊断 |
| `dump_weight.cpp` | 权重加载正确性 |

服务器 numpy 旧（无 bfloat16），需手动 `uint16<<16` 重解释。
