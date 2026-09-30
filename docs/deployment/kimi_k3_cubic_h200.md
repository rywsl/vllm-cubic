# Kimi-K3 Cubic：上游同步与 H200 验证

本次把本地 vLLM 主分支完整合入 Cubic，目标是单机 8×H200、TP8+EP
上的 `QuantTrio/Kimi-K3-Cubic-2.5Bit`。优先改善首个非空生成内容的 P95
TTFT，吞吐最多下降 10%。**源码和 CPU 回归已验证；CUDA 编译、模型质量、
H200 吞吐与缓存收益尚未验证。** 下列配置是实验起点，不是已测出的最优参数。

## 固定版本与兼容范围

| 项目 | 固定版本 |
| --- | --- |
| 官方 Cubic fallback | `QuantTrio/vllm-cubic` release `v0.26.1+cubic.20260805` |
| 上游合入目标 | `82daf9f5756e1868be0aa751afaec4726beca12a` |
| 上游共同基点 | `073c510c916f385315a5366173c883781762bb9e` |
| 本地合并分支 | `feat/kimi-k3-kvv` |
| 合版前备份分支 | `backup/our-cubic-merged-before-v0.30.0-20260929` |
| 目标权重、tokenizer | `f29f15dc4afd99feb3349b538bbe7ed439787853` |
| `Inferact/Kimi-K3-DSpark` | `cf6b8244620e7ea4b0651d214f28e89eac75bed6` |

合入 1,977 个上游提交，同时迁移以下 Cubic 契约：

- Cubic W1–W8、动态 A8、compact metadata 和 TP/EP packed MoE 权重加载；
  适配新的 Marlin 参数接口，保留 FP16 scale / BF16 output。
- Cubic8 KV 的实际字节布局、allocator stride 和 K3 fused KV 写入；
  `CUBIC8_GROUPWISE=11` 避开上游新增的枚举值。
- `fp8_q16` FlashMLA speculative 多 query 的 causal 边界与 LSE 布局；
  DCP 下保持 query dtype 和跨 rank prefix context 处理。
- 混合 KDA/Mamba 前缀缓存、EAGLE checkpoint 与 recurrent block 对齐；
  保留 GDN MTP、fresh-prefill graph guard、Cubic/speculative warmup。
- 迁移 batch-invariant 模块与 MHC 固定 FP32 归约，保留其他下游模型路径、
  独立 NCCL protocol communicator 和 worker 清理预算。

上游 dSpark、RecoverSSM、细粒度 prefix match 与 DCP 是分别评估的变量。
当前 dSpark 不能与 DCP>1 组合；H200 draft 使用 `TRITON_MLA`，不使用
仅适配 SM100 的 FlashInfer MLA draft 路径。RecoverSSM 使用 runner V2、
Triton Mamba 与 FP32 SSM，其余预设显式使用 runner V1。

## 本地验证与限制

无 NVIDIA GPU/nvcc 的 Python 3.12 环境已完成：

- 31 项下游源码契约检查及其 pytest wrapper。
- correctness harness 22 项、Cubic policy 75 项。
- 压测客户端 21 项，包括真实本地 HTTP/SSE 回放；以上三组加下游 wrapper
  合计一次运行 119 项通过。
- Cubic 量化与 warmup 合集 180 项通过、1126 项 CUDA 测试跳过。
  最终与上述 119 项一起运行：**299 passed、1126 skipped**。
- Qwen/Cubic packed loader 13 项通过，覆盖上游 dense layout 转置旁路。
- KV/MLA 定向回归 29 项通过、50 项 GPU 测试跳过。
- scheduler/model config 72 项、prefix-cache/EAGLE checkpoint 66 项。
- env 58 项、shutdown 14 项、Cubic warmup 31 项通过及 1 项 CUDA 跳过。
- Marlin output dtype 2 项、Cubic target matching 4 项；新增 repack ABI
  的 exact W4 / A16 / A8 三项 CPU 行为回归包含在上述量化合集内。
- 通信选路/cleanup/EPLB 定向 13 项，PyNccl protocol 与环境恢复 6 个 mock 用例。
- H200 launcher 的 7 个预设通过真实 CLI 参数解析、shell 语法与 ShellCheck。
- FlashMLA patch 可应用于新 pin `0eee43b`；SM90 Marlin generator 生成了
  17 个 CUDA 编译单元，但没有执行原生编译。
- 相对固定上游的下游变更通过全部适用 pre-commit hooks（含 mypy）；
  3,375 个本次变动的 Python 文件通过 AST 语法解析。

以上是分组结果，存在交集，不应相加当作唯一测试总数。完整模型配置测试因
Llama-4 gated config 的 Hugging Face 401 中断；纯本地子集已通过。
CPU 测试不能证明 CUDA kernel 数值、TP8 collectives、量化质量或模型可装载。

## H200 构建、正确性与启动

在配有兼容 CUDA toolkit（含 nvcc）、驱动与 8 张 H200 的机器中执行。
合并版构建会编译当前 checkout 的扩展；不要用官方预编译 wheel 替代合并版 Cubic 扩展。
若合并版明确阻塞，客户试运行使用已发布的官方 Cubic wheel
`v0.26.1+cubic.20260805`。
先查看命令，再执行构建与 GPU 回归：

```bash
bash examples/quantization/kimi_k3_h200.sh build --dry-run
bash examples/quantization/kimi_k3_h200.sh build
bash examples/quantization/kimi_k3_h200.sh gpu-tests baseline
```

GPU 回归保存 `artifacts/kimi-k3-h200-gpu-tests.xml`。失败或意外跳过时先修复；
该测试包只代表本机覆盖，不能替代已有 correctness harness 所要求的完整
SM/bit/dtype 矩阵。可额外运行已有独立 Cubic 数值回归：

```bash
.venv/bin/python tools/run_cubic_correctness_gates.py
.venv/bin/python tools/cubic_correctness_harness.py --help
```

为每个配置使用独立 manifest 和日志，顺序启动，一次只占用一套 GPU：

```bash
mkdir -p artifacts
KIMI_MANIFEST=artifacts/merged-baseline-server.json \
  bash examples/quantization/kimi_k3_h200.sh serve baseline \
  2>&1 | tee artifacts/merged-baseline-startup.log
```

| 预设 | 相对 baseline 的实验变量 |
| --- | --- |
| `baseline` | TP8+EP，BF16，`fp8_q16`，DCP1，align prefix cache |
| `cache32` | `--prefix-match-unit 32` |
| `dcp2` / `dcp4` / `dcp8` | decode context parallel size 2 / 4 / 8 |
| `dspark` | 固定 draft revision，7 speculative tokens，DCP1 |
| `recoverssm` | dSpark + RecoverSSM + runner V2 + FP32 SSM |

性能调优默认使用 `max-model-len=auto`、max sequences 128、batched tokens 2048、
GPU memory utilization 0.965、CUDA graph capture 上限 128、`prefix-match-unit=32`、
seed 42。1M 上下文切换和验收后置，不把本阶段的 auto 结果当作 1M 能力证明。OOM 时先降低并发或 memory utilization，
并把相同参数应用于对照组。`KIMI_BATCHED_TOKENS` 可逐项尝试 1024/2048/4096；
`KIMI_KV_CACHE_DTYPE` 支持 `auto`、`bfloat16`、`fp8_q16`、`cubic8`，
dSpark/RecoverSSM 预设限定 target 为 `fp8_q16`。

脚本绑定 `127.0.0.1:8000`，开启供压测使用的 dev-mode cache reset/server-info
端点，并在 `KIMI_API_COMPAT=1` 时开启 Kimi Vendor Verifier 兼容校验和 K3
严格工具调用。manifest 记录源码、权重和 draft revision、参数、GPU 容量与
白名单环境。
保留启动日志中的真实 KV block/容量信息；混合 MLA/SSM 的容量不能直接等同于
`num_gpu_blocks × block_size` 个通用上下文 token。若从本地权重目录启动，
还应保存文件校验和；revision 参数本身不能证明本地文件内容。

服务健康后，至少运行现有中文长输出检查，再用相同固定数据集做 logprob/
perplexity、短问答、长上下文和多轮质量比较。中文检查只是 smoke test：

```bash
.venv/bin/python examples/quantization/validate_kimi_k3_cubic_online.py \
  --port 8000 --model Kimi-K3-Cubic-2.5Bit --seed 42 \
  --thinking-effort low --max-tokens 8192
```

## 可重复性能实验

用同一 tokenizer revision 生成一次请求，两个版本复用同一文件。生成器记录
chat-template 后的实际 token 数；指定长度是近似目标，统计以 response usage
为准。输出长度用 `ignore_eos` 固定，质量测试需另用自然停止。

```bash
.venv/bin/python benchmarks/kimi_k3_cubic_h200.py prepare \
  --tokenizer QuantTrio/Kimi-K3-Cubic-2.5Bit \
  --input-tokens 8192 --output-tokens 1024 --count 128 \
  --output artifacts/requests-8k.jsonl
.venv/bin/python benchmarks/kimi_k3_cubic_h200.py run \
  --requests artifacts/requests-8k.jsonl \
  --server-manifest artifacts/merged-baseline-server.json \
  --label merged-baseline --scenario cold --concurrency 8 --rounds 3 \
  --output artifacts/merged-8k-c8-cold.json
```

建议矩阵：输入 2K/8K/32K/64K，输出 1024，并发 1/8/32/64，每个配置至少
3 轮，每轮至少 128 请求。先比较旧源码与合并后 baseline，再分别测试各预设。
同一轮内没有其他请求；启动、编译、CUDA graph capture 完成后才计时。
脚本额外执行一个模型 warmup 请求；复杂 graph shape 首次出现仍可能编译，
正式采样前应另跑完整目标并发/长度的预热轮并检查日志。

| 场景 | 行为 |
| --- | --- |
| `cold` | 清空前缀缓存，给每条请求添加靠前的不同标记 |
| `repeat` | 清缓存，预热所有请求后精确重放 |
| `shared` | 清缓存，只预热第一条，其他请求共享长前缀 |
| `append` | 预热全部，将真实回答加入历史后继续一轮 |
| `pressure` | 预热全部，插入默认 256 个不同前缀，再重放原请求 |

`pressure` 不保证必然发生驱逐；用 `--pressure-requests` 增加干扰量，并检查
KV 占用、preemption、实际 hit rate 和服务日志。`append` 的历史依赖模型输出；
跨版本若历史不同，比较器会拒绝。此时先固化统一多轮历史到请求文件，使用
`repeat` 比较，并保留原动态 append 结果作为独立行为观察。

指标包括：首个非空 reasoning/content 的 TTFT P50/P95/P99、首个 content 时间、
TPOT、输出 token/s、服务端 ITL/排队时间均值、token 加权 GPU prefix hit rate、
external cache hit rate、preemption。SSE chunk 间隔另存为 `inter_chunk_ms`；
投机解码可一次输出多个 token，因此它不是 ITL。每轮 metrics 前后默认等待
6 秒让周期统计落盘，等待时间不计入吞吐；若修改服务端统计周期也须调大等待。
没有输出、缺 usage、缺 `[DONE]`、错误 finish 或失败请求均不允许通过比较。

如合并版被阻塞，fallback 与候选版使用相同权重、参数、GPU、负载与 runner 做
最小客户 smoke；launcher 和 benchmark 只作为外部测试工具，不改变官方 release。
保留 release 版本和工具 dirty 标记；若 release CLI 不支持新增参数，删去该参数并
记录实际行为差异。
manifest 必须来自实际被测服务，不可拿候选版本的 manifest 代替。

```bash
.venv/bin/python benchmarks/kimi_k3_cubic_h200.py compare \
  artifacts/old-8k-c8-cold.json artifacts/merged-8k-c8-cold.json
```

比较器要求权重/tokenizer revision、GPU、请求与实际历史 hash、场景、并发一致，
至少 3 轮且全部成功。以每轮 P95 TTFT 的中位数改善且吞吐中位数不低于 fallback
90% 为性能门禁。性能通过后仍须独立通过 GPU 数值与模型质量门禁；本次交付
不含任何已测性能提升比例，也不执行生产部署。

## Kimi-Vendor-Verifier API 预检

仓库外的 `Kimi-Vendor-Verifier` 固定到 `66092cf444c97356c0e11c5078c67116390615d9`
时，使用本 checkout 的 server URL 和 `Kimi-K3-Cubic-2.5Bit` 运行四套原版
pytest：`tests/params`、`tests/k3_features`、`tests/tool_call_json_schema`
和 `tests/prompt_tokens`。验收入口不修改 verifier 源码、断言或官方 skip；它
另外记录首轮失败、重试后的最终结果、跳过原因、HTTP 状态、request-id 和请求体
哈希，Authorization 只记录为 `Bearer ***`。

先在有网络的准备机执行：

```bash
.venv/bin/python tools/kimi_k3_kvv.py prepare \
  --verifier-dir ../Kimi-Vendor-Verifier
.venv/bin/python tools/kimi_k3_kvv.py check \
  --verifier-dir ../Kimi-Vendor-Verifier
```

`prepare` 会检查 Git LFS 指针和固定数据哈希；prompt-token JSONL、视觉图片
和 BEAM 资产未 hydrate 时会明确失败，不能把 pytest collection failure 当作
模型失败。准备完成后，在 H200 服务已启动且 `KIMI_API_KEY` 已通过环境变量
提供的机器上运行：

```bash
export KIMI_BASE_URL=http://127.0.0.1:8000/v1
export KIMI_API_KEY=local-only-key
export MODEL_NAME=Kimi-K3-Cubic-2.5Bit
.venv/bin/python tools/kimi_k3_kvv.py run \
  --verifier-dir ../Kimi-Vendor-Verifier \
  --work-dir artifacts/kvv-cubic
```

默认使用 `THINK_MODE=opensource`、四线程和 verifier 自己声明的 rerun 规则，
按参数、K3 feature、Tool Schema（thinking 开／关）、prompt token 顺序执行，
并生成 JUnit、attempt、collection、trace 和 summary JSON。测试通过只代表
这四类 API 行为在该服务上符合 verifier；它不代表 OCR/MMMU、BEAM 1M、DeepSWE
或量化后的模型质量达到了 README 中列出的参考分数。BEAM 1M 还需要独立 judge、
完整 1,048,576 context 和 K3 tokenizer 配置；本 launcher 默认使用 1,048,576，
但仍必须通过实际服务加载、边界请求和质量验收。
