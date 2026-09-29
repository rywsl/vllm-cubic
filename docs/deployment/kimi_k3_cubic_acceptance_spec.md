# Kimi K3 Cubic 验收 Spec

状态：`Ready for H200 acceptance`（本地源码、CPU 和离线 verifier 预检已完成；在线 KVV、CUDA/H200、性能和模型质量仍待执行）

本规范只用于验收 Kimi K3：包括 vLLM 主分支合版、Kimi Vendor Verifier（KVV）兼容改造，以及
`QuantTrio/Kimi-K3-Cubic-2.5Bit` 的性能优化候选。每个结论必须绑定版本、负载、
机器和证据文件。CPU、静态检查或 `build --dry-run` 结果不能作为 H200 性能、CUDA
数值、模型质量或在线 API 通过的替代证明。

当前候选树已包含上游 vLLM `0.30.0` 发布版的特性，发布 tag 为
`ced6857afa0ea7b2e3f0846a62e1394e90f15607`。当前 Cubic 合版基线
`82daf9f5756e1868be0aa751afaec4726beca12a` 与该 tag 从共同基点分叉，不在同一
ancestry 路径；合版中已包含其余发布修复的等价上游提交。因此采用选择性同步：保留
Kimi/Cubic 下游实现，并补入发布版 CPU 镜像中 Triton CPU SLEEF 子模块的浅克隆恢复
检查；不能把当前树描述为发布 tag 的逐字节重建。

## 1. 固定对象和范围

### 1.1 版本清单

| 对象 | 固定值 | 用途 |
| --- | --- | --- |
| 原 Cubic | `89af4f1ff792199b7c9260dd4a86feed445fd3d6` | 性能对照和回滚基线 |
| 上游合入目标 | `82daf9f5756e1868be0aa751afaec4726beca12a` | vLLM 主分支同步边界 |
| 上游共同基点 | `073c510c916f385315a5366173c883781762bb9e` | 合版审计锚点 |
| 合版父提交 | `b0cfd84b5a614ace8097756eedc8464202ab5d46` | 上游同步结果 |
| 候选 HEAD | `dfc3ec8808d27e413d582f1b9347740e1e4c966f` | KVV 改造后待验收版本 |
| 模型 | `QuantTrio/Kimi-K3-Cubic-2.5Bit` | 目标权重 |
| 模型/tokenizer/code revision | `f29f15dc4afd99feb3349b538bbe7ed439787853` | 服务和 fixture 必须一致 |
| dSpark draft revision | `cf6b8244620e7ea4b0651d214f28e89eac75bed6` | 仅 dSpark/RecoverSSM 实验 |
| KVV 仓库 | `https://github.com/MoonshotAI/Kimi-Vendor-Verifier` | 上游测试源 |
| KVV commit | `66092cf444c97356c0e11c5078c67116390615d9` | 不允许漂移 |

权重、tokenizer、code 和 draft 的 revision 必须写入服务 manifest；revision 字符串
不能替代本地文件校验和。旧版和候选版必须使用相同权重、tokenizer、GPU、运行参数和
请求 fixture。旧版只能保留 benchmark 工具差异，并在 manifest 中记录 dirty 状态。

### 1.2 范围边界

本规范验收：

- 上游 vLLM 合版后 Cubic 下游契约是否仍存在且可运行。
- K3 thinking、采样参数、动态 system tools、tool choice、response format、
  prompt token 和流式协议是否符合固定 KVV。
- Cubic 量化、KV cache、prefix cache、DCP、dSpark、RecoverSSM、warmup 和
  batch invariant 等候选优化在固定 H200 负载上的 TTFT、吞吐和缓存指标。

本规范不把 KVV 通过解释为 OCR、MMMU、BEAM 1M、DeepSWE 或通用模型质量通过。
自然停止质量、多轮、长上下文、logprob/perplexity 和中文输出必须走独立质量门禁。
如果对外宣称 Kimi K3 支持百万上下文，服务必须以 `max_model_len=1048576` 启动并通过
1M token 边界和质量验收；launcher 默认值已经设为 `1048576`，任何显式降低到
327680 或更低的运行只能作为 smoke，不能作为产品能力证明。

## 2. 验收门禁和结论规则

验收按以下顺序执行，前一层失败时不得把后一层标记为通过：

| 门禁 | 内容 | 结论 |
| --- | --- | --- |
| G0 | 版本、源码、LFS、依赖和证据环境固定 | `PASS` / `FAIL` |
| G1 | 合版后 Cubic contract、单元回归和静态质量检查 | `PASS` / `FAIL` |
| G2 | KVV source gate（prepare/check） | `PASS` / `FAIL` |
| G3 | H200 CUDA 构建、数值和模型加载 | `PASS` / `FAIL` / `APPROVED_SKIP` |
| G4 | 在线 KVV 四套测试 | `PASS` / `FAIL` |
| G5 | 性能比较和缓存率观测 | `PASS` / `FAIL` |
| G6 | 独立模型质量检查 | `PASS` / `FAIL` |

最终发布结论必须写明每个门禁状态。`APPROVED_SKIP` 只能由负责人针对明确的
硬件覆盖范围批准，并且不能转写为整体验收通过；没有 G3 不能宣布性能或质量通过。

## 3. G0/G1：vLLM 合版和 Cubic 契约

### 3.1 必须保留的下游能力

合版验收需要检查以下能力仍可用：

- Cubic W1-W8、dynamic A8、compact metadata、packed MoE loader，以及新的
  Marlin 参数接口适配。
- Cubic8 KV 的字节布局、allocator stride 和 K3 fused KV 写入。
- `fp8_q16` FlashMLA speculative 多 query 的 causal/LSE/DCP 边界。
- KDA/Mamba 混合前缀缓存、recurrent state alignment、EAGLE/MTP 和 warmup。
- batch-invariant 低 M 执行、MHC FP32 归约、独立 NCCL protocol communicator
  和 worker cleanup。
- K3 renderer/parser、strict tool calling、thinking precedence 和动态 tools。

机器化契约清单由 `tools/check_cubic_downstream.py` 维护；当前为 31 项。契约检查
失败即阻止合版验收，不能用某个单测通过抵消源码契约缺失。

### 3.2 本地验收步骤

在仓库根目录执行：

```bash
uv run --no-project --python 3.12 tools/check_cubic_downstream.py
uv run --no-project --python 3.12 -m pytest -q \
  tests/quantization/test_cubic_downstream.py::test_downstream_contract_survives_upstream_sync
```

再执行本 checkout 的适用 pre-commit hooks（ruff、format、typos、markdownlint、
shellcheck、mypy、SPDX 和 forbidden-import 检查）。至少保留：

```bash
git diff --check
pre-commit run --all-files
bash examples/quantization/kimi_k3_h200.sh build --dry-run
shellcheck examples/quantization/kimi_k3_h200.sh
```

无 NVIDIA GPU 的当前基线证据为：下游 contract 31 项通过；Cubic 重点回归合计
`299 passed, 1126 skipped`（测试组有交集，不能相加）；launcher 参数解析、shell
语法和 ShellCheck 通过。该结果只证明源码/CPU 边界，不证明 CUDA kernel、TP8
collective、量化质量或模型可加载。

### 3.3 CUDA/H200 构建和数值门禁

在独占的 8×H200 主机上使用当前 checkout 编译扩展，禁止用官方预编译 wheel
替代 Cubic extension：

```bash
bash examples/quantization/kimi_k3_h200.sh build --dry-run
bash examples/quantization/kimi_k3_h200.sh build
bash examples/quantization/kimi_k3_h200.sh gpu-tests baseline
.venv/bin/python tools/run_cubic_correctness_gates.py
```

`gpu-tests` 的 JUnit、完整 correctness harness 的 JUnit/日志和编译日志必须归档。
correctness harness 的 `STRICT_RTOL=0`、`STRICT_ATOL=0` 是硬门禁；当执行性能命令
时，必须完成所有 required Cubic cases，并覆盖支持的 SM `(80, 86, 89, 90, 100,
103, 110, 120)` 的 BF16 路径。实验容差不能让性能门禁通过。

服务启动使用 launcher 的 `baseline`，并保存启动 manifest：

```bash
KIMI_MANIFEST=artifacts/merged-baseline-server.json \
  bash examples/quantization/kimi_k3_h200.sh serve baseline \
  2>&1 | tee artifacts/merged-baseline-startup.log
```

默认固定：TP8+EP、PP1、BF16、Cubic、FlashMLA、`fp8_q16` KV、prefix caching、
Mamba align/TRITON、chunked prefill、产品验收 `max_model_len=1048576`、max sequences 128、
batched tokens 2048、GPU memory utilization 0.95、seed 42、K3 API compatibility
和 strict tool calling。性能 benchmark 所需的 `/metrics`、`/server_info` 和
`/reset_prefix_cache` 依赖 `VLLM_SERVER_DEV_MODE=1`。manifest 必须显示 8 张 H200、
模型/tokenizer/code revision
和 clean/dirty 状态；健康检查、模型加载和 CUDA 数值均通过后才进入 G4/G5。

## 4. G2/G4：KVV 改造验收

### 4.1 改造要求

兼容开关 `VLLM_KIMI_K3_API_COMPAT` 默认关闭，KVV 服务验收时必须开启。协议行为
要求如下：

1. thinking 生效优先级为顶层 `thinking` > `reasoning_effort` >
   `chat_template_kwargs.thinking`/`thinking_effort` > 默认；支持 `low/high/max`，
   `reasoning_effort=none` 表示关闭思考。
2. 参数限制在 raw request 进入 Pydantic 前返回受控 HTTP 400：`temperature` 在
   0-1，`top_p=0.95`，`presence_penalty=0`，`frequency_penalty=0`，`n=1`。
3. 支持 text、json、json_schema response format，保留 OpenAI stream/non-stream
   语义、finish reason、usage 和 `[DONE]`。
4. dynamic tools 只允许 system message，保留原消息位置；能与 top-level tools
   合并传给 renderer/parser；校验重复工具名、工具 schema、工具名、tool call ID
   关联以及非法非 system 用法，并返回预期 400。
5. compat 模式启用 K3 renderer/model guard；服务和 streaming parser 使用同一组
   effective tools，结构化 tag 解析结束后恢复原始 top-level tools。

### 4.2 Source gate：固定 verifier 和 LFS

准备机执行，不能修改 verifier 的测试、断言或官方 skip：

```bash
.venv/bin/python tools/kimi_k3_kvv.py prepare \
  --verifier-dir ../Kimi-Vendor-Verifier
.venv/bin/python tools/kimi_k3_kvv.py check \
  --verifier-dir ../Kimi-Vendor-Verifier
```

`check` 必须记录 verifier HEAD、origin、clean 状态、LFS OID/size 和完整 collection。
固定 commit 的 collection 应为 611 个 nodeid：

| Suite | nodeid 数 |
| --- | ---: |
| `tests/params` | 18 |
| `tests/k3_features` | 126 |
| `tests/tool_call_json_schema` | 408 |
| `tests/prompt_tokens` | 59 |
| 合计 | 611 |

LFS 需要 8/8 文件 hydrated 且 OID/size 匹配。当前 source gate 证据目录为
`/tmp/kimi-k3-kvv-prep/` 和 `/tmp/kimi-k3-kvv-prep2/`；它们不包含在线模型结论。

`tests/prompt_tokens` 的 59 个 nodeid 中，55 个是 live API case（44 个文本 fixture、
11 个视觉 fixture），4 个是本地容差边界单测。live case 读取 streaming usage 的
`prompt_tokens`，允许值为 `expected` 到 `expected + 3`；这 4 个本地单测不能计入
在线模型覆盖。

### 4.3 Live gate：四套测试和证据

服务启动后，由 secret manager 或运行环境注入 `KIMI_API_KEY`；密钥不得出现在
命令参数、文档、日志、trace 或 request body。仅允许运行时读取环境变量：

```bash
export KIMI_BASE_URL=http://127.0.0.1:8000/v1
export MODEL_NAME=Kimi-K3-Cubic-2.5Bit
test -n "$KIMI_API_KEY"
.venv/bin/python tools/kimi_k3_kvv.py run \
  --verifier-dir ../Kimi-Vendor-Verifier \
  --work-dir artifacts/kvv-cubic
```

runner 按以下五个 stage 执行，其中 tool schema 跑 thinking 关闭和开启两遍：

`params`、`k3_features`、`tool_schema_off`、`tool_schema_on`、`prompttokens`。
因此 JUnit 预期 testcase 总数为 `18 + 126 + 408 + 408 + 59 = 1019`。
9 个允许的官方基础 skip nodeid 在 stream/non-stream 参数化后通常对应约 18 个
JUnit skip；验收按 runner 的基础 nodeid 归并，并要求 unexpected skip 为 0。

Live gate 的通过条件：

- verifier commit、LFS 和 collection 与 G2 相同；`summary.json` 状态为 `pass`。
- 所有适用用例通过；固定 source 中允许的 9 个官方 skip 之外不得出现 skip，且
  unexpected skip 为 0。
- failed/error 为 0；rejection nodeid 实际收到 HTTP 400，不能只看异常文本。
- 每个请求有 request-id；attempt、JUnit、trace 和 summary 可互相追溯。
- request body 只保存 SHA-256；Authorization 只保存 `Bearer ***`；证据目录通过
  credential scan。

KVV 通过只证明协议和工具调用行为，不能证明模型回答质量、视觉能力或百万上下文。
官方 skip（包括 thinking effort 行为、named tool choice rejected 和 tokenization
groundtruth）必须原样记录，并在报告中说明其覆盖边界。

## 5. G5：性能优化验收

### 5.1 候选优化和证据

所有候选都必须相对同一 `baseline` 独立比较；一次只改变一个变量，dSpark 不得和
DCP>1 组合。候选、目标指标和最低要求如下：

| 优化点 | 变量/实现边界 | 主要指标 | 通过要求 |
| --- | --- | --- | --- |
| Cubic W1-W8、dynamic A8、compact metadata、packed MoE loader/repack | `cubic.py`、`cubic_policy.py`、`cubic_kernels.py` | 加载、数值、质量、吞吐 | GPU correctness 和质量先通过；无 unexpected failure |
| `fp8_q16` native MLA KV | FlashMLA/K3 fused KV 路径 | TTFT、吞吐、KV 容量 | 服从统一 compare gate |
| `cubic8` KV layout | Cubic8 allocator/写入路径 | 显存、cache hit、TTFT | 无正确性回归；收益需实测 |
| prefix cache 对齐 / `prefix-match-unit=32` | `baseline` vs `cache32` | prefix hit、TTFT | 同负载下报告 token-weighted hit rate；目标不低于 baseline |
| DCP2/4/8 | `dcp2`、`dcp4`、`dcp8` | TTFT、吞吐、队列/ITL | TTFT 改善且吞吐达到 floor |
| dSpark speculative | 7 token、固定 draft revision、DCP1 | TTFT、吞吐、自然停止质量 | 性能改善且质量通过 |
| RecoverSSM | dSpark + runner V2 + FP32 SSM | decode 吞吐、cache、质量 | 性能改善且质量通过 |
| Cubic/FlashMLA/Mamba kernels 和 warmup | GPU kernels、Cubic/K3 Triton warmup、batch-invariant | 编译、数值、首轮延迟 | correctness 完整覆盖；不得以编译成功替代数值通过 |

缓存率是独立验收指标。当前 benchmark compare 只对 TTFT 和吞吐设置硬门禁，
因此报告必须同时给出 token-weighted `prefix_cache_hit_rate`、
`external_prefix_cache_hit_rate`、preemption、KV 配置和容量；若项目要求把缓存率
作为发布门禁，应在验收单中预先填写相对 baseline 的定量阈值，不能事后挑选结果。

### 5.2 固定服务 profile

| profile | 唯一变量 |
| --- | --- |
| `baseline` | TP8+EP、DCP1、`fp8_q16`、align prefix cache |
| `cache32` | 另加 `--prefix-match-unit 32` |
| `dcp2`/`dcp4`/`dcp8` | decode context parallel size |
| `dspark` | 7 speculative tokens、`TRITON_MLA` draft、DCP1 |
| `recoverssm` | dSpark + runner V2 + FP32 SSM、DCP1 |

每个 profile 使用独立 manifest、启动日志和输出目录；不得复用已有 evidence 目录。

### 5.3 Benchmark protocol

用固定 tokenizer revision 生成一次 fixture，旧版和候选版复用同一 JSONL。建议矩阵：

- 输入 2K、8K、32K、64K；输出预算 1024。
- 并发 1、8、32、64；每个配置至少 3 轮，每轮至少 128 请求。
- 场景 `cold`、`repeat`、`shared`、`append`、`pressure`。
- warmup、CUDA graph capture 和首个目标 shape 编译均在正式采样前完成。
- 每轮 reset prefix cache；metrics 前后默认等待 6 秒，等待时间不计吞吐。

生成 fixture：

```bash
.venv/bin/python benchmarks/kimi_k3_cubic_h200.py prepare \
  --tokenizer QuantTrio/Kimi-K3-Cubic-2.5Bit \
  --revision f29f15dc4afd99feb3349b538bbe7ed439787853 \
  --input-tokens 8192 --output-tokens 1024 --count 128 \
  --output artifacts/requests-8k.jsonl
```

运行时 benchmark 客户端读取 `OPENAI_API_KEY`；若服务启用认证，应从 secret manager
向该环境变量注入同一密钥，不能把 key 写入命令行或结果文件：

```bash
.venv/bin/python benchmarks/kimi_k3_cubic_h200.py run \
  --requests artifacts/requests-8k.jsonl \
  --server-manifest artifacts/merged-baseline-server.json \
  --label merged-baseline --scenario cold --concurrency 8 --rounds 3 \
  --output artifacts/merged-8k-c8-cold.json
```

必须采集：首个非空 reasoning/content 的 TTFT P50/P95/P99、first content、TPOT、
output tokens/s、inter-chunk interval、队列/ITL 均值、prefix/external cache hit
rate、preemption、usage、finish reason、`[DONE]` 和 request-id。缺少非空输出、usage、
`[DONE]`、合法 `finish_reason` 或有失败请求，整轮无效。

### 5.4 Compare gate

旧版和候选版分别运行后执行：

```bash
.venv/bin/python benchmarks/kimi_k3_cubic_h200.py compare \
  artifacts/old-8k-c8-cold.json artifacts/merged-8k-c8-cold.json
```

比较器必须拒绝以下情况：fixture、场景、并发、模型/tokenizer revision、GPU devices
不一致；轮数少于 3；同轮 workload hash 不一致（包括 append 生成的历史）；任一请求
失败；TTFT/吞吐非有限正数。

通过条件为：

1. 候选各轮 P95 TTFT 的中位数严格小于 baseline。
2. 候选 output tokens/s 中位数不低于 baseline 的 90%。
3. GPU correctness 和独立质量门禁均已通过。
4. 缓存率、容量、preemption 和失败重试均有原始证据；未达到预设缓存率阈值时，
   性能结论只能标记为 `PARTIAL`。

`pressure` 不保证一定发生驱逐；`append` 如果跨版本输出历史不同，比较器会拒绝，
此时应固化统一多轮历史后用 `repeat` 完成可比实验，并保留 append 作为行为观察。

## 6. G6：独立质量门禁

在性能通过后，以自然停止（不使用 `ignore_eos`）运行相同模型和参数的：

- 中文长输出和短问答 smoke test。
- 多轮对话、2K/8K/32K/64K 长上下文和工具调用。
- 量化模型的 logprob/perplexity 或项目批准的等价基线。
- dSpark/RecoverSSM 各自的自然停止质量对照。

输出截断、异常重复、工具参数错误、reasoning/content 丢失、请求失败或显著质量
回退均为 `FAIL`。KVV 通过、TTFT 改善或吞吐达标都不能替代本门禁。

## 7. 证据包和当前状态

### 7.1 必须归档的文件

每次候选验收至少生成以下文件，并在 `manifest.json` 中记录 SHA-256：

```text
manifest.json
server-startup.log
server-info.json
gpu-build.log
gpu-tests.junit.xml
cubic-correctness.junit.xml
kvv/prepare.json
kvv/check.json
kvv/collection.json
kvv/summary.json
kvv/attempt-*.jsonl
kvv/trace-*.jsonl
requests-*.jsonl
baseline-*.json
candidate-*.json
compare-*.json
checksums.sha256
credential-scan.txt
quality-report.json
```

证据必须包含 git SHA、dirty 状态、模型/tokenizer/draft revision、GPU 型号和数量、
CUDA/PyTorch/vLLM 版本、完整 CLI（删除 secret）、环境白名单、fixture/workload hash、
时间、request-id 和失败重试历史。禁止保留 API key、cookie、Authorization 原值、
完整 prompt 中的用户秘密或未经批准的响应原文。

### 7.2 当前状态（本 spec 编写时）

| 项目 | 状态 | 证据/限制 |
| --- | --- | --- |
| G0 版本和源码边界 | `PASS` | 当前 HEAD `dfc3ec8808`，固定版本表 |
| G1 合版 contract/CPU/pre-commit | `PASS` | contract 31；CPU 回归 `299 passed, 1126 skipped`；无 CUDA |
| G2 KVV prepare/check | `PASS` | verifier `66092cf`；LFS 8/8；collection 611；`/tmp/kimi-k3-kvv-prep*` |
| G3 CUDA/H200 构建、数值、加载 | `PENDING` | 当前环境无 NVIDIA GPU/nvcc |
| G4 在线 KVV | `PENDING` | 未执行真实模型/API 请求 |
| G5 TTFT/吞吐/缓存率 | `PENDING` | 未启动 H200 服务，未有实测收益 |
| G6 模型质量 | `PENDING` | 未执行自然停止、长上下文和独立质量集 |

只有 G0-G6 的证据齐全且所有强制门禁通过，才可把候选标记为 `ACCEPTED`。否则使用
`PARTIAL`（明确已通过和缺失项）或 `REJECTED`，不得用“已优化”“性能已提升”等表述
替代缺失实测。

## 8. 关联实现和运行手册

- 上游合版和 launcher 说明：[kimi_k3_cubic_h200.md](kimi_k3_cubic_h200.md)
- 首字 TTFT 渐进调优和两小时缓存策略：[kimi_k3_cubic_ttft_tuning_plan.md](kimi_k3_cubic_ttft_tuning_plan.md)
- KVV runner：`tools/kimi_k3_kvv.py`、`tools/kimi_k3_kvv_plugin.py`
- K3 协议回归：`tests/entrypoints/openai/chat_completion/test_kimi_k3_protocol.py`
- KVV runner 回归：`tests/tools/test_kimi_k3_kvv.py`
- 性能客户端：`benchmarks/kimi_k3_cubic_h200.py`
- 下游 contract：`tools/check_cubic_downstream.py`
- H200 launcher：`examples/quantization/kimi_k3_h200.sh`
