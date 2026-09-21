# Kimi K3 Cubic 首字 TTFT 调优计划

目标是单机 8×H200、64 路并发下，让首个非空 reasoning 或正文的 P95 TTFT
进入 10 秒以内。典型请求由约 297K token 的稳定公共前缀和不超过 3K token 的
私有输入组成。调优必须逐项改变参数，每个阶段都保留完整的 cache、排队、失败和
首字证据。

## 先解决两个前置条件

当前 launcher 默认 `max_model_len=131072`，不能测试 297K 公共前缀。先根据 tokenizer
真实计数、chat template、私有输入和输出余量设置，例如：

```bash
KIMI_MAX_MODEL_LEN=327680 \
  bash examples/quantization/kimi_k3_h200.sh serve baseline
```

`327680` 只是起始值，必须由实际 token 数和模型配置上限确认；提高上下文长度会增加
显存 profiling、KV 容量和启动风险。先用单请求确认模型能加载，再进入 64 路压测。

`--prefix-cache-retention-interval` 不是时间 TTL，它的单位是 token，只对 sliding-window
和 Mamba checkpoint 生效。K3 的本地 GPU prefix cache 仍按容量压力下的 LRU/FIFO 淘汰，
没有按 wall-clock 两小时自动删除的配置；`--prefix-match-unit 32` 也只改变匹配粒度。

如果“只保存两小时”是硬性数据保留要求，推荐按以下方式实现：

1. 双实例滚动：新实例启动并预热 297K 公共前缀，切流后让旧实例排空并销毁，保证旧
   实例中的 GPU KV 不超过两小时。预热和切流过程必须单独计时，不能把冷实例结果混入
   热缓存 TTFT。
2. 单实例低峰清理：仅在无运行请求时调用本地管理接口清空 cache，随后重新预热。该
   方案会产生冷缓存窗口，不适合作为 10 秒 SLO 的唯一方案。
3. 若必须做到每个 cache block 的精确两小时 idle TTL，需要给 BlockPool 增加按访问
   时间的淘汰逻辑和并发测试；现有 vLLM 配置不能宣称已经支持该语义。

`/reset_prefix_cache` 依赖 `VLLM_SERVER_DEV_MODE=1`，只应绑定 localhost 并由受控
sidecar 使用，不能把开发端点直接暴露到公网。无论采用哪种方案，都要记录 reset、
切流、预热和销毁时间。若私有输入也不能留在 KV 中，现有 API 没有“只缓存公共前缀、
禁止写入私有 suffix”的独立开关；`cache_salt` 只能改变整条 prompt 的 cache key，
会同时破坏公共前缀复用，需要额外的请求路由或引擎改造。

两小时到期和 10 秒 SLO 需要分开定义：清 cache 后的下一条请求是 cold request，
重新计算 297K 前缀，不能假定也能在 10 秒内完成。若客户的 SLO 覆盖过期后的第一条
请求，必须使用双实例滚动，在旧实例到期前让备用实例完成预热、命中检查和切流；单实例
定时 reset 只能保证保留上限，不能同时保证无 cold penalty。

## 渐进实验顺序

所有实验固定模型/code/tokenizer revision、8×H200、TP8+EP、DCP、KV dtype、GPU 显存
上限和 fixture hash。每个点至少 3 轮、每轮 128 请求、并发 64；预热、CUDA graph
capture 和 Triton 编译不计入正式采样。一个阶段未通过时停止后续阶段。

### P0：确认公共前缀真的命中

用同一份 297K 公共前缀 fixture，只改变末尾私有输入，先跑单请求，再跑 64 路：

```bash
.venv/bin/python benchmarks/kimi_k3_cubic_h200.py prepare \
  --tokenizer QuantTrio/Kimi-K3-Cubic-2.5Bit \
  --revision f29f15dc4afd99feb3349b538bbe7ed439787853 \
  --input-tokens 297000 --output-tokens 128 --count 128 \
  --output artifacts/requests-297k.jsonl
```

请求必须保持 system、工具定义、资料顺序和模板完全一致，私有内容只放在末尾。先用
`shared` 验证一条请求预热后其余请求是否复用，再用 `repeat` 验证完整重放。通过条件：

- token-weighted prefix hit rate 接近 100%，命中 token 数接近 297K；不能只看请求数命中率。
- MLA KV 和 KDA/Mamba 状态都命中；不能只证明某一个 cache group 命中。
- 64 路无 preemption、无失败请求、usage 和 `[DONE]` 完整。
- 记录 semantic TTFT、排队时间、prefill 时间、KV 容量和每轮 workload hash。

如果 P0 失败，先修复公共前缀字节/token 不一致、上下文长度、KV 容量或缓存布局，
不要继续调 `max-num-batched-tokens`。

### P1：调 prefill 调度预算

固定 P0 的命中 fixture，依次测试：

`2048 → 4096 → 8192 → 16384 → 32768`，必要时再测试 `65536`。

通过 `KIMI_BATCHED_TOKENS` 设置，每次只改变该变量。较大的预算通常降低 TTFT，但会
增加单步显存和 decode 抢占风险；较小的预算通常更有利于 ITL。选择满足以下条件的
最小预算：64 路首个非空 token P95 ≤ 10 秒，P99 没有异常长尾，preemption 为 0，
吞吐不低于 baseline 的 90%。

### P2：调并发上限和排队

固定 P1 预算，保持客户端并发 64，比较 `max-num-seqs=64/96/128`。同时记录
`request_queue_time_seconds`、running/waiting 请求数和 preemption。若 64 路已经排队，
继续增大 `max-num-seqs` 通常不会降低首字；此时优先降低排队、拆分副本或分流。

### P3：调 DCP 和匹配粒度

固定 P1/P2，按 `DCP1 → DCP2 → DCP4 → DCP8` 测试。每次确认 297K 公共前缀在所有
rank 的命中 token 数一致，且通信时间没有抵消 prefill 收益。再比较默认匹配粒度和
`--prefix-match-unit 32`；后者不是 TTL，也不保证一定降低 TTFT。

### P4：调 KV dtype、容量和内核

在已通过的调度参数下比较 `fp8_q16` 与 `cubic8`（若模型和 kernel correctness 已
通过），记录实际 KV 容量、cache hit、prefill kernel 时间、显存峰值和错误率。只有当
容量增加确实减少抢占或追加 prefill 时间时，才把显存节省折算为 TTFT 收益。

### P5：最后评估 dSpark/RecoverSSM

dSpark 和 RecoverSSM 主要影响 decode 吞吐，不能替代公共前缀命中和追加 prefill 优化。
在 P0-P4 稳定后再比较 `baseline`、`dspark`、`recoverssm`，并用自然停止质量集确认
reasoning/content 没有丢失或异常截断。

## 两小时保留验收

在实际运行中记录公共前缀 warm-up 完成时间 `t0`，至少在 `t0+119m` 和 `t0+121m`
各做一次同负载 probe：前者应保持公共 prefix hit，后者应按选定的滚动/reset 策略
落到新实例并重新预热，或明确记录为 cold miss。没有等待两小时的实验只能作为缩短的
演练（例如临时 60 秒策略），不能宣称生产 TTL 已通过。

## 统一首字门禁

客户端以第一个非空 reasoning 或正文 chunk 作为 TTFT 起点，不把 role/空 chunk 当首字。
每轮必须有完整 SSE、usage、合法 finish reason 和 `[DONE]`；任何失败请求、缺 usage、
缺 `[DONE]` 或 cache 指标不完整都会使该轮作废。

每个参数点至少生成：

```text
server-manifest.json
requests-297k.jsonl
benchmark.json
compare.json
metrics-before-after.json
startup.log
```

最终选择规则：

1. 先满足 297K 公共前缀的完整命中和两小时保留策略。
2. 再选择 64 路 P95 semantic TTFT ≤ 10 秒的最小调度预算。
3. 在满足 TTFT 的候选中，选择吞吐较高、preemption 为 0、缓存率稳定的配置。
4. 任何缓存 TTL、GPU 正确性或模型质量未验证的结果只能标记为 `PARTIAL`，不能写成
   “首字已达标”。

当前没有 H200 服务和实测数据，因此以上是可执行的调优顺序和门禁，不是已经达成 10 秒
SLO 的性能结论。
