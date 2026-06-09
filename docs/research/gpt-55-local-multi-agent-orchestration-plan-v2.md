# GPT 5.5 Research Plan v2: Local Multi-Agent Orchestration on vMLX

Date: 2026-06-09
Author: GPT 5.5
Status: v2 research plan
Supersedes: `docs/research/gpt-55-local-multi-agent-orchestration-plan-v1.md`

## Why v2 exists

v1 was written before the external agent plan was provided. That plan adds a
specific operational proposal:

- a thin FastAPI orchestrator on port 7010;
- one proxy wrapper per vMLX model server;
- Redis Pub/Sub for wake, handoff, and result messages;
- Discord/Slack/Kanban as read-only human mirrors;
- deep sleep as a mandatory RAM control on a 64GB machine;
- a fixed model/port role map;
- future Flash-MoE exploration;
- Hermes Agent as a possible harness/UI/memory/tool reference.

This v2 document reviews that proposal and folds in the parts that improve the
plan, while correcting parts that would make the system fragile.

## v2 decision summary

The external plan changes the v1 plan in useful ways, but it does not overturn
the core architecture.

Accepted changes:

1. Add a proxy/control boundary for model sessions so orchestration code does
   not patch bundled app files or depend on MLXStudio internals.
2. Treat sleep/wake and "max active agents" as first-class scheduler/admission
   decisions, not only background power-management settings.
3. Use Redis for low-latency notifications and wake/handoff fan-out.
4. Keep Discord, Slack, Linear, Notion, or Kanban systems as read-only mirrors
   or task trackers, not the critical path bus.
5. Add explicit RAM, disk, wake-latency, OOM, and idle-time observability.
6. Add a role-to-model map as a starting deployment profile.
7. Consider Hermes Agent as a UI/memory/tool-infra reference.

Rejected or corrected changes:

1. A 200-line orchestrator is not enough for durable multi-agent work. It may be
   enough for the first spike, but the real system still needs durable runs,
   steps, events, leases, artifacts, cancellation cleanup, and replay.
2. Redis Pub/Sub must not be the source of truth. It drops messages when
   consumers are offline. Use SQLite/Postgres event storage as the durable
   ledger; Redis Pub/Sub or Redis Streams can be used for live fan-out.
3. The proxy should wrap the vMLX HTTP admin/API endpoints first. Do not assume
   a "vMLX GUI API" is the control surface. The repo already documents
   `POST /admin/{soft-sleep,deep-sleep,wake}` in `docs/ARCHITECTURE.md`.
4. "Spill to CPU inference" is not a reliable vMLX fallback plan for Apple
   Silicon MLX workloads. The safer fallback is queueing, using a smaller local
   model, reducing concurrency/context/cache pressure, or routing to a fallback
   provider.
5. `nvidia-smi` is not applicable to the Apple Silicon target. Use vMLX
   `/health`, MLX/Metal memory telemetry, process RSS via `psutil`, and system
   memory pressure.
6. Fixed active RAM estimates and 1M-context claims must be measured. They
   should seed admission tests, not be treated as capacity truth.
7. Flash-MoE belongs behind a later measured feasibility gate, not the first
   orchestration milestone.

## Final v2 architecture

```text
Human UI / Local API
  |
  v
Agent Orchestrator API
  - /task
  - /runs
  - /runs/{id}/events
  - /agents
  - /status
  - /handoff
  |
  v
Durable Control Plane
  - agent registry
  - run manager
  - step/event store
  - scheduler/admission control
  - resource leases
  - artifact store
  - approval state
  |
  +--> Live Event Bus
  |     - Redis Pub/Sub for low-latency fan-out, or Redis Streams if durable
  |       consumer groups are needed
  |
  +--> vMLX Proxy Layer
  |     - one proxy per managed model/session or one proxy with per-agent routes
  |     - wraps vMLX /health, /admin/soft-sleep, /admin/deep-sleep,
  |       /admin/wake, /v1/chat/completions, /v1/responses, and cancellation
  |
  +--> Provider Adapters
  |     - vMLX primary
  |     - llama.cpp fallback
  |     - LM Studio fallback
  |     - Ollama fallback
  |
  +--> Tool/MCP Gateway
  |     - schema validation
  |     - tool policy
  |     - sandboxing
  |     - approval gates
  |     - audit events
  |
  +--> Human Mirrors
        - Discord/Slack webhook bridge
        - Linear/Notion/Kanban sync
        - static status dashboard or Prometheus/Grafana
```

The main difference from v1 is the explicit `vMLX Proxy Layer`. The main thing
that stays from v1 is the durable run/event control plane.

## Proxy layer: accepted with corrections

The proxy idea is useful because it keeps the orchestration system outside the
signed/bundled app and avoids patching:

- `/Applications/vMLX.app/Contents/Resources/bundled-python/`;
- packaged vMLX internals;
- MLXStudio app files;
- model bundle files.

The proxy should be treated as a compatibility and policy shim, not as the
durable orchestrator.

### Proxy responsibilities

For each managed model session, the proxy should:

- forward `/v1/chat/completions`;
- forward `/v1/responses`;
- forward cancellation routes;
- forward `/health`;
- wrap `/admin/soft-sleep`;
- wrap `/admin/deep-sleep`;
- wrap `/admin/wake`;
- track last activity;
- emit wake/sleep/error/heartbeat events;
- add agent/session metadata;
- normalize health state for the orchestrator;
- enforce proxy-level idle timeout;
- expose a local status endpoint for the orchestrator.

### Proxy non-responsibilities

The proxy should not:

- own durable run state;
- own task planning;
- execute tools directly;
- rewrite model tool arguments;
- hide exactness failures;
- mutate packaged/bundled vMLX files;
- claim media support where vMLX only preserves weights.

### Proxy topology

The external plan says:

```text
vMLX instance :8083 <-> proxy :8083-proxy -> orchestrator
```

Implementation should use real numeric ports or path routing, for example:

```text
vMLX :8083  <-> proxy :18083
vMLX :8084  <-> proxy :18084
vMLX :8089  <-> proxy :18089
```

or:

```text
proxy :7011
  /agents/architect/* -> vMLX :8083
  /agents/lead/*      -> vMLX :8084
  /agents/vision/*    -> vMLX :8089
```

The second shape is easier to operate if the orchestrator and proxy live in the
same service. The first shape is easier to debug per model.

## Orchestrator: thin API first, durable core next

The external plan proposes a small FastAPI app:

```text
POST /task
GET  /status
POST /handoff
GET  /agents
```

This is a good first API surface, but it should sit over the v1 durable
orchestration objects.

### Required API surface

Keep the external plan's endpoints:

```text
POST /task
GET  /status
POST /handoff
GET  /agents
```

Add durable-run endpoints early:

```text
POST /runs
GET  /runs
GET  /runs/{run_id}
GET  /runs/{run_id}/events
POST /runs/{run_id}/cancel
POST /runs/{run_id}/retry
POST /runs/{run_id}/approve
```

`POST /task` can be a convenience wrapper around `POST /runs`.

### Orchestrator responsibilities

The orchestrator should:

- decide target worker/agent;
- ask the scheduler for admission;
- wake target proxy/session when needed;
- create durable run records;
- stream run events;
- aggregate health;
- apply retry policy;
- enforce max active agents;
- enforce tool and model budgets;
- record failure boundaries;
- notify human mirrors.

It should not:

- use Redis Pub/Sub as the only state;
- call bundled Python internals;
- depend on Discord/Slack for agent-to-agent handoff;
- run all agents concurrently on 64GB RAM;
- route to a model family whose tool/exactness/media proof is red for the task.

## Message bus: Redis is useful, but not sufficient

The external plan's topic split is good:

```text
agent:events
agent:handoff
agent:results
agent:heartbeat
```

v2 keeps these topics for live fan-out. The durable source of truth remains the
run event store:

```text
agent_run_events
  seq
  run_id
  step_id
  event_type
  payload_json
  created_at
```

Recommended bus semantics:

- SQLite/Postgres event store is authoritative.
- Redis Pub/Sub is optional low-latency notification.
- Redis Streams can be used if the team wants replayable consumer groups.
- If Redis is down, the orchestrator continues in direct-proxy degraded mode and
  writes durable events locally.
- When Redis returns, mirrors and subscribers resync from the durable event
  store, not from missed Pub/Sub messages.

## Human dashboards

The external plan is correct that human channels should be mirrors, not the
critical path.

Use:

- Discord/Slack webhooks for summaries, alerts, and manual review prompts.
- Linear/Notion/Kanban for task tracking.
- A local status page or dashboard for live model, run, and memory state.

Do not use:

- Discord or Slack as the structured handoff bus.
- Chat messages as the durable run ledger.
- Human dashboards as the only place where failure state is visible.

## Sleep/wake and RAM scheduling

The external plan's most important correction to v1 is operational: on 64GB RAM,
deep sleep is not optional.

### Scheduling invariant

The scheduler must enforce:

```text
max_active_heavy_agents = 2
max_active_agents_total = measured value from proof gates
```

The exact values must be measured. The external plan estimates peak active RAM
around 87GB if all listed agents are awake, which exceeds a 64GB machine. That
is enough to make deep sleep mandatory, but not enough to finalize limits.

### Admission checks

Before waking or routing a run, the scheduler should check:

- current active agents;
- model family and model size;
- expected context length;
- expected media usage;
- cache mode and block disk size;
- current system memory pressure;
- current Metal/MLX memory reported by vMLX;
- process RSS;
- current run priority;
- whether a smaller model can satisfy the task;
- whether the task can wait for a warm model.

### Wake policy

Recommended wake flow:

1. Create durable run event: `wake_requested`.
2. Call proxy `/admin/wake`.
3. Poll proxy/vMLX `/health`.
4. If wake succeeds, create event: `wake_ready`.
5. If wake fails, retry with bounded backoff.
6. If still failing, mark agent unhealthy and route to fallback or queue.

### Sleep policy

Recommended sleep flow:

1. Proxy tracks last model activity.
2. Orchestrator also tracks run leases.
3. Proxy may request sleep after idle timeout.
4. Orchestrator approves sleep only if no active lease exists.
5. Proxy calls vMLX `/admin/deep-sleep` or `/admin/soft-sleep`.
6. Durable event records mode, reason, and result.

This prevents a proxy from deep-sleeping a model while a run still owns it.

## Corrected fallback plan

The external plan says:

```text
On OOM -> fallback to lower --max-num-seqs or spill to CPU
```

The corrected plan is:

1. Queue until another heavy model sleeps.
2. Retry with lower active concurrency.
3. Reduce request context or require summarization.
4. Disable optional media path for text-only tasks.
5. Route to a smaller local model.
6. Route to llama.cpp/LM Studio/Ollama fallback provider if the task allows it.
7. Mark the run blocked with an explicit admission reason.

CPU spill should not be the default fallback for MLX. It may be tested as a
future diagnostic mode only if the provider supports it cleanly and the quality
and latency tradeoff is acceptable.

## Observability corrections

The external plan's monitoring table is directionally right, but the metrics
sources need Apple Silicon corrections.

| Metric | Correct source | Action |
| --- | --- | --- |
| Active model memory | vMLX `/health`, MLX/Metal telemetry, process RSS, system memory pressure | Queue, sleep, or reject when unsafe |
| Sleep/wake cycles | proxy events plus durable run events | Detect wake flaps and failed reloads |
| Response latency | proxy timing plus provider usage | Detect model degradation |
| OOM count | orchestrator admission and proxy/vMLX failures | Lower concurrency or mark agent unhealthy |
| Idle time | proxy and orchestrator leases | Deep sleep only when no active lease exists |
| Cache reuse | vMLX usage details with `cached_tokens` and `cache_detail` | Verify no cross-agent contamination |
| Disk cache pressure | vMLX cache stats and filesystem usage | Prune or deny new cache-heavy runs |

Do not use `nvidia-smi` for the Apple Silicon path.

## Role and model deployment profile

The external plan's role map is a useful starting profile, not a capacity proof.

| Port | Role | Model | v2 classification |
| --- | --- | --- | --- |
| 8083 | Architect | Qwen3.6-27B 1M ctx | Heavy reasoning/planning candidate; requires live context and cache proof |
| 8084 | Lead Engineer | Qwen3.6-27B 1M ctx | Heavy coding candidate; separate port is valid if resource limits allow |
| 8089 | Vision Best | Qwen3.6-35B MoE | Heavy vision/model candidate; requires media proof and memory gate |
| 8087 | Test Engineer | Qwen3.5-9B | Good smaller worker candidate |
| 8090 | Vision Backup | Qwen3-VL-8B | Good smaller media fallback if live media proof passes |
| 8091 | Vision Backup | Qwen2.5-VL-7B | Good smaller media fallback if live media proof passes |
| 8092 | Summarizer | Qwen3.5-27B-8bit | Heavy summarizer candidate; consider smaller summarizer first |

v2 adds this rule:

```text
Role assignment is a policy preference. The scheduler must still route by
current proof status, loaded state, memory pressure, parser/tool support, media
support, and task risk.
```

## File structure recommendation

The external plan proposes creating a new `~/agent-system/` tree. That is a
reasonable deployment shape for a standalone spike:

```text
~/agent-system/
  orchestrator/
    main.py
    config.yml
    requirements.txt
  vmlx-proxy/
    proxy.py
    config.yml
    requirements.txt
  bus/
    redis-compose.yml
  bridge/
    discord.py
  README.md
```

For this repository, the long-term location should be decided before
implementation:

Option A: standalone repo or `~/agent-system`

- best for experimentation;
- avoids touching vMLX release code;
- easiest to run against installed app and source servers.

Option B: new `agent_system/` or `orchestrator/` directory in this repo

- best if it becomes an officially supported vMLX/MLXStudio feature;
- easier to test in CI;
- higher risk of coupling to release gates.

Option C: MLXStudio main-process integration

- best operator UX;
- highest coupling;
- should come after standalone proof.

Recommendation: start with Option A for the spike, then promote stable pieces
into this repo only after proof gates pass.

## Hermes Agent note

The external note says to consider Hermes Agent as a harness/main UI and
memory/tool-infra reference.

v2 recommendation:

- review Hermes Agent for memory, tool registry, UI, and self-evolution ideas;
- do not use it as the primary scheduler or durable run store before the local
  vMLX-aware scheduler exists;
- borrow memory abstractions only if memory writes are explicit, reviewed,
  replayable, and tied to run events.

Hermes-style memory should be an optional subsystem:

```text
Run events -> reviewed memory candidates -> approval/policy -> memory store
```

Never:

```text
model output -> hidden autonomous long-term memory write
```

## Flash-MoE

The external plan correctly demotes Flash-MoE to future work.

v2 keeps it out of the first implementation.

Required before Flash-MoE:

- actual RAM headroom measurements with current 27B/35B workloads;
- disk bandwidth benchmark on the target 4TB SSD;
- measured wake latency;
- measured cache restore latency;
- measured MoE expert paging behavior;
- quality proof on long/code/tool tasks;
- explicit fallback when expert streaming stalls or degrades output.

Expected risk:

- 100B+ MoE with SSD streaming may be possible, but should be expected to be
  much slower than RAM-resident inference until measurements prove otherwise.

## Updated implementation sequence

### Phase 0: Document and align

1. Land v1 and v2 research docs.
2. Confirm whether the first spike lives outside the repo in `~/agent-system`
   or inside this repo under a new directory.
3. Confirm whether Redis is required for the spike or whether direct proxy calls
   plus SQLite event storage are enough for the first proof.

### Phase 1: Proxy and status spike

1. Create a proxy that wraps one vMLX session.
2. Forward `/health`, `/admin/soft-sleep`, `/admin/deep-sleep`,
   `/admin/wake`, `/v1/chat/completions`, `/v1/responses`, and cancellation.
3. Add proxy heartbeat and last-activity tracking.
4. Add proxy metadata: role, model, vMLX port, proxy port, idle timeout, cache
   policy, max active lease count.
5. Prove sleep/wake through the proxy without editing bundled vMLX files.

### Phase 2: Durable orchestrator MVP

1. Add `agents`, `agent_runs`, `agent_run_steps`, `agent_run_events`, leases,
   and artifacts.
2. Implement `/task`, `/runs`, `/runs/{id}/events`, `/agents`, and `/status`.
3. Implement direct proxy calls without Redis first.
4. Add Redis Pub/Sub or Redis Streams only after durable event storage exists.
5. Add run cancellation that releases leases and calls provider/proxy cancel.

### Phase 3: Sleep/wake scheduler

1. Encode the role/model/port map as config.
2. Add max active heavy-agent policy.
3. Add wake admission checks.
4. Add bounded wake retry.
5. Add sleep approval based on leases.
6. Measure two-agent and three-agent RAM headroom.

### Phase 4: Handoff and tool gateway

1. Implement handoff as durable event plus optional Redis notification.
2. Add planner-worker-verifier graph.
3. Add tool policy and approval events.
4. Add human mirror bridge for Discord/Slack summaries.
5. Keep Discord/Slack out of the critical path.

### Phase 5: Reliability proof

1. Run 100 handoff cycles.
2. Kill Redis and prove direct-proxy degraded mode.
3. Kill a proxy and prove unhealthy state plus retry/fallback.
4. Kill the orchestrator mid-run and prove replay/recovery.
5. Force memory pressure and prove queue/reject/fallback behavior.
6. Capture cache telemetry and prove no cross-agent contamination.

### Phase 6: UI and promotion

1. Build local run/status dashboard.
2. Add MLXStudio integration only after standalone proof.
3. Promote stable interfaces into this repo if desired.

## Updated validation gates

### Proxy gates

- Proxy health reflects vMLX health plus role/session metadata.
- Proxy sleep calls vMLX admin sleep endpoint.
- Proxy wake calls vMLX admin wake endpoint.
- Proxy refuses sleep while an active run lease exists.
- Proxy emits heartbeat and state events.
- Proxy survives vMLX restart/update without bundled file patches.

### Bus gates

- Redis down does not lose durable run state.
- Redis down does not block direct proxy execution.
- Redis restart allows mirrors to resync from durable events.
- Handoff messages are schema-validated.
- Duplicate handoff delivery is idempotent.

### Scheduler gates

- More than the allowed active heavy-agent count queues or rejects with reason.
- Wake OOM retries are bounded.
- Smaller fallback model routing works for eligible tasks.
- A media task never routes to text-only runtime.
- A tool-required task never routes to a model/parser whose tool proof is red.

### Sleep/wake gates

- Soft sleep and deep sleep work through proxy.
- Inference-triggered wake works where supported.
- Wake preserves max context/output/parser/cache settings.
- Cache restore after wake reports `cached_tokens` and `cache_detail`.
- Failed wake records durable failure events.

### Run durability gates

- Run state survives orchestrator restart.
- Tool execution events survive restart.
- Partial model output is persisted.
- Cancellation releases proxy/model/tool leases.
- Replay reconstructs the full run timeline.

## What did not change from v1

v2 keeps these v1 decisions:

- vMLX remains the preferred inference provider.
- Provider abstraction remains mandatory.
- MLXStudio/vMLX session and cache infrastructure remain the best harness base.
- Durable run/event storage is mandatory.
- Tool/MCP calls must go through a policy gateway.
- The orchestrator should be graph-driven, not unbounded group chat.
- Human channels are mirrors, not control-plane state.
- llama.cpp, LM Studio, and Ollama remain fallback providers.
- Release blockers still gate production claims.

## Current release boundary

This remains a research and implementation-planning document only.

Still open:

- MiMo exactness, long-prompt, continuous-batching/system-prompt pressure,
  source-vs-quant classification when memory allows, and media runtime wiring.
- MiniMax issue #179 reporter parity/root-cause.
- DSV4 long-output/code/file-generation proof.
- Real Electron UI live model matrix.
- Per-family live cache/API/tool/media proof.

Forbidden until green:

- release packaging;
- Developer ID signing;
- notarization;
- tags;
- appcast/latest download updates;
- public release claims.

## Final v2 recommendation

The other agent's plan improves v1 by introducing a clean proxy boundary and by
making deep-sleep orchestration explicit. Those changes should be adopted.

The plan should not be implemented as only a 200-line FastAPI router plus Redis
Pub/Sub. That would be a useful demo, but not a reliable local multi-agent
system.

The best v2 foundation is:

```text
vMLX primary provider
  + provider abstraction
  + proxy/session control boundary
  + durable run/event store
  + resource-aware scheduler
  + Redis live bus as optional fan-out
  + policy-enforced MCP/tool gateway
  + read-only human mirrors
```

This keeps vMLX as the center of the system while avoiding bundled-file patches,
avoiding fragile chat-as-state, and preserving the release-lock boundary until
runtime/model/UI/cache proof gates are green.
