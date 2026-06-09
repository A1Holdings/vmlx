# GPT 5.5 Research Plan v1: Local Multi-Agent Orchestration on vMLX

Date: 2026-06-09
Author: GPT 5.5
Status: v1 research plan

## Scope

This document captures the research review for building a local multi-agent
orchestration system on top of the existing MLX/vMLX ecosystem. It covers:

1. The local inference layer, including the three relevant MLX-based forks and
   adjacent fallback providers.
2. The harness layer, using the existing vMLX/MLXStudio process, session, chat,
   tools, and cache surfaces.
3. The orchestration layer, borrowing patterns from local and public
   multi-agent frameworks without adopting a cloud-first control plane.

This is a planning document only. It does not claim release readiness and does
not authorize packaging, signing, notarization, tags, appcast updates, public
downloads, or release claims.

## Methodology and limits

The review used:

- The checked-out repository: `A1Holdings/vmlx`, a fork of `jjang-ai/vmlx`.
- Local docs and source files, including:
  - `README.md`
  - `pyproject.toml`
  - `docs/ARCHITECTURE.md`
  - `docs/guides/tool-calling.md`
  - `docs/guides/mcp-tools.md`
  - `panel/PROJECT.md`
  - `panel/README.md`
  - `panel/src/main/sessions.ts`
  - `panel/src/main/ipc/chat.ts`
  - `panel/src/main/tools/*`
  - `vmlx_engine/server.py`
- GitHub repository metadata for related forks under `A1Holdings` and
  `jjang-ai`.
- Parallel research tracks for:
  - local inference/fork selection;
  - harness/process-management design;
  - orchestration patterns and external references.

Limitations:

- I did not find a local file containing the other agent's exact proposed plan,
  so the plan review here evaluates the inferred proposal: use vMLX as the
  local inference layer, MLXStudio as the local harness, and add a durable
  multi-agent orchestration layer above them.
- The GitHub token could not list starred repositories, so saved/starred repos
  outside the fork inventory were not directly inspectable from this
  environment.
- The current release guard says the vMLX/MLXStudio release objective still has
  open runtime/model/UI/cache rows. This plan preserves that boundary.

## Executive recommendation

Use vMLX as the preferred inference foundation, but put it behind a provider
adapter so the orchestration system can fall back to llama.cpp, LM Studio, or
Ollama when a model family or proof gate is not release-green.

Use MLXStudio's existing session/process/chat/tool/cache machinery as the local
harness substrate, but add first-class orchestration objects:

- agents;
- runs;
- run steps;
- append-only run events;
- resource leases;
- artifacts;
- scheduler/admission decisions.

Use a hybrid orchestration model:

```text
typed graph workflow
  + supervisor-worker routing
  + planner-executor-verifier loop
  + blackboard/task-board state
  + MCP/tool gateway
  + local resource scheduler
  + durable checkpoint/replay
```

Do not use unbounded group chat as the primary control plane. It is harder to
replay, harder to audit, and unsafe for local tool execution.

## Part 1: Local inference layer

### Fork and provider inventory

| Candidate | Role | Finding |
| --- | --- | --- |
| `A1Holdings/vmlx` | Primary local MLX inference server and MLXStudio app foundation | Best strategic foundation. It already targets OpenAI Chat, OpenAI Responses, Anthropic Messages, Ollama routes, MCP/tools, cache telemetry, cancellation, and MLXStudio session management. It is not release-green yet. |
| `A1Holdings/mlx-vlm` | MLX VLM/omni model package | Strong component for image/audio/video model support. It is not a full orchestration server or scheduler. Use as a backend dependency and diagnostic reference. |
| `A1Holdings/dflash-mlx` | Exact speculative decoding on MLX | Useful later as an acceleration component for selected Qwen-style planner/verifier paths. It is a focused generator, not a provider server or orchestration foundation. |
| `jjang-ai/mlx-swift` | Swift API for MLX | Relevant for future native Swift lanes, but not the current Python engine plus Electron app objective. Do not switch to this lane unless explicitly requested. |
| `mlx-lm` upstream | Text runtime baseline | Use as the compatibility anchor and low-level dependency baseline. Not enough server/orchestration surface by itself. |
| `mlx-vlm` upstream | Multimodal runtime baseline | Use as model-family backend reference. Not enough process, cache, tool, or run orchestration by itself. |
| LM Studio | Productized local provider | Good operator fallback and desktop provider. Less transparent and less suited as the hackable core. |
| llama.cpp server | Mature non-MLX fallback | Best reliability/performance baseline for GGUF text/tool agents. Does not cover MLX/JANG-specific requirements. |
| Ollama | Simple local lifecycle provider | Good fallback for easy model serving. Not enough typed cache/scheduler/tool proof for the core. |

### vMLX strengths for orchestration

vMLX is the best-aligned foundation because it already includes:

- OpenAI-compatible `/v1/chat/completions`.
- OpenAI-compatible `/v1/responses` and `previous_response_id` support.
- Anthropic-compatible `/v1/messages`.
- Ollama-compatible routes.
- Tool parser registry for multiple model dialects.
- MCP tool discovery and execution endpoints.
- Streaming and non-streaming paths.
- Cancellation endpoints.
- Prefix cache, paged cache, typed/native cache families, block disk L2, and
  cache telemetry.
- Usage metadata with `cached_tokens` and `cache_detail`.
- MLXStudio session creation, process launch, health checks, chat UI, cache UI,
  settings UI, and benchmark UI.

### vMLX risks

The release state is the core risk, not the architectural shape.

Current blockers that matter for an agent system:

- MiMo V2.5 JANGTQ_2 exact literal/tool/JSON mutations remain open.
- MiMo long-prompt, continuous-batching/system-prompt pressure, and media
  runtime rows remain open.
- MiniMax issue #179 reporter parity/root-cause remains open.
- DSV4 exact code/file/long-output proof remains memory-gated/open.
- Real Electron UI live model matrix remains open.
- Source-only/no-heavy API/cache contracts do not replace live family proof.
- Hybrid/SSM/SWA/composite/media cache rows require typed proof; generic KV
  cache evidence is not enough.

### Inference-layer decision

Build a provider abstraction first:

```ts
interface LocalModelProvider {
  chatCompletion(request: ChatRequest): AsyncIterable<ModelEvent>
  responses(request: ResponsesRequest): AsyncIterable<ModelEvent>
  cancel(requestId: string): Promise<void>
  health(): Promise<ProviderHealth>
  capabilities(): Promise<ModelCapabilities>
  cacheStats(): Promise<CacheStats>
}
```

Initial providers:

1. `VmlxProvider` as the primary path.
2. `LlamaCppProvider` as deterministic GGUF fallback.
3. `LmStudioProvider` as desktop/operator fallback.
4. `OllamaProvider` as simple lifecycle fallback.

The orchestrator should route by capability rather than brand:

- tool parser support;
- streaming support;
- max context;
- cache family;
- media support;
- model load state;
- measured tokens/sec;
- memory estimate;
- current release/proof status.

## Part 2: Harness layer

### Current harness capabilities

MLXStudio already has a useful substrate for orchestration:

- Session registry and process lifecycle.
- Per-session ports and health checks.
- Process adoption for existing `vmlx-engine serve` processes.
- Streaming chat requests.
- Request cancellation with `AbortController`.
- Engine-side cancel endpoints.
- SQLite persistence for sessions, chats, messages, overrides, settings, and
  benchmarks.
- Tool-call replay fields in message storage.
- Built-in local coding tools.
- MCP discovery/execution integration.
- Cache stats, cache entries, cache warm, and cache clear paths.
- Benchmark probes for TTFT/TPS and cache-sensitive behavior.
- Settings UI for cache, parser, MTP, max output, max context, media, and tools.

This means the first orchestration implementation should not duplicate the
model streaming/tool loop. It should wrap and progressively factor the existing
chat loop into reusable execution primitives.

### Harness gaps

The current system is chat/session-centric, not agent/run-centric.

Missing concepts:

- first-class agent registry;
- durable run entity;
- run step entity;
- append-only event log;
- queue and scheduler;
- resource admission policy;
- per-agent tool policy;
- per-agent working directory or sandbox;
- run-level cancellation and timeout cleanup;
- tool lease ownership;
- explicit cache namespace/isolation;
- artifact store;
- crash/restart recovery for active runs;
- orchestration telemetry and replay.

### Harness architecture

Add these main-process services:

```text
AgentManager
  owns agent profiles, defaults, tool policy, model/session affinity

RunManager
  creates durable runs, maps runs to chats/messages, owns status transitions

Scheduler
  admits, queues, leases sessions/tools/resources, applies model capability rules

ExecutionAdapter
  wraps existing chat/model/tool loop as one run step backend

EventStore
  append-only event log for scheduler, model, tool, approval, cache, and errors

ArtifactStore
  stores files, logs, command outputs, traces, screenshots, and eval records
```

### Minimal schema

```sql
agents (
  id,
  name,
  description,
  default_session_id,
  model_path,
  system_prompt,
  tool_policy_json,
  working_directory,
  cache_policy_json,
  resource_policy_json,
  created_at,
  updated_at
)

agent_runs (
  id,
  agent_id,
  chat_id,
  session_id,
  status,
  priority,
  input_json,
  output_json,
  config_snapshot_json,
  started_at,
  finished_at,
  attempt,
  parent_run_id
)

agent_run_steps (
  id,
  run_id,
  type,
  status,
  started_at,
  finished_at,
  input_json,
  output_json,
  error_json,
  metrics_json
)

agent_run_events (
  id,
  run_id,
  step_id,
  event_type,
  seq,
  payload_json,
  created_at
)

agent_resource_leases (
  id,
  run_id,
  resource_type,
  resource_id,
  status,
  acquired_at,
  released_at
)

agent_artifacts (
  id,
  run_id,
  step_id,
  kind,
  path,
  metadata_json,
  created_at
)
```

### Minimal IPC/API

```text
agents:list
agents:get
agents:create
agents:update
agents:delete

runs:start
runs:list
runs:get
runs:cancel
runs:retry
runs:events
runs:queue/status
runs:admission-check

artifacts:list
artifacts:read
```

Pass `agent_id` and `run_id` through:

- chat request metadata;
- tool execution context;
- cache namespace or cache salt where supported;
- benchmark/proof metadata;
- telemetry events;
- cancellation paths.

## Part 3: Orchestration layer

### External patterns to borrow

| Source | Borrow | Avoid |
| --- | --- | --- |
| LangGraph | typed graph, checkpoints, replay/fork, human interrupts | unnecessary dependency bloat if a small native graph is enough |
| AutoGen | actor/message patterns, direct vs pub/sub communication | classic group chat as the primary control plane |
| CrewAI | flow owns deterministic control, crew handles specialist work | role-play agents as a safety boundary |
| OpenAI Agents/Swarm | simple agent specs, handoffs, guardrails | hosted-provider assumptions |
| Semantic Kernel/Microsoft Agent Framework | approval-required tools, HITL workflows | heavy enterprise framework coupling |
| LlamaIndex Workflows | event-driven steps, shared context, async fan-out | implicit event race semantics |
| Haystack Agents | tool confirmation and retrieval/tool composition | making retrieval pipeline semantics drive all orchestration |
| NVIDIA NeMo Agent Toolkit / AgentIQ / AI Blueprints | YAML-configured workflows, profiling, eval artifacts, MCP/A2A ideas | NIM/cloud/GPU assumptions |
| Agent Orchestrator | worktree isolation, dashboard, feedback routing, runtime plugins | treating coding-agent PR flow as the whole local model scheduler |
| NemoClaw/OpenShell | sandbox, policy-enforced filesystem/network/inference calls | alpha stack as production dependency |
| PydanticAI / pydantic-graph | typed state, schema-first tools and outputs | replacing local runtime controls with another agent platform |
| Temporal-style workflows | workflow/activity/signal split, durable retry | full Temporal server for the first desktop-local version |
| mem0/Letta/ByteRover | memory blocks, long-term memory patterns | hidden autonomous memory writes |

### Recommended orchestration model

Use deterministic workflow control around bounded model autonomy.

```text
IntakeNode
  normalize request, classify risk/capabilities

PlannerNode
  create bounded task graph and budgets

DispatchNode
  choose worker, model, provider, session, and cache policy

WorkerNode
  call local provider, stream events, parse model/tool outputs

ToolGatewayNode
  validate args, enforce policy, request approval, execute, persist result

VerifierNode
  check exactness, evidence, code syntax, tool results, and leak conditions

SynthesisNode
  produce final answer with artifacts and run summary

ArchiveNode
  checkpoint final state, record eval candidate, close leases
```

### Typed run state

```ts
type RunState = {
  runId: string
  userGoal: string
  messages: Message[]
  plan: Task[]
  blackboard: Record<string, ArtifactRef>
  activeAgent?: string
  approvals: ApprovalRequest[]
  budgets: {
    maxSteps: number
    maxToolCalls: number
    maxTokens: number
    maxWallSeconds: number
  }
  modelConstraints: {
    requiredContext: number
    needsTools: boolean
    needsVision: boolean
    privacyLevel: string
  }
  scheduler: {
    preferredModel?: string
    loadedModel?: string
    cacheKey?: string
  }
  safety: {
    riskLevel: "read" | "write" | "exec" | "network" | "destructive"
  }
  trace: {
    checkpointId: string
    parentCheckpointId?: string
  }
}
```

### Worker model

Define workers as profiles, not as unbounded autonomous identities:

```yaml
workers:
  planner:
    good_for: [classification, task_decomposition]
    required_capabilities: [json_exactness]
    max_tool_calls: 0

  coder:
    good_for: [code_editing, tests]
    required_capabilities: [tools, code_exactness]
    allowed_tools: [read_file, edit_file, run_tests]

  verifier:
    good_for: [review, tests, exactness]
    required_capabilities: [long_context, structured_output]
    max_tool_calls: 4

  media_worker:
    good_for: [image, audio, video]
    required_capabilities: [media]
```

The scheduler maps worker profiles to actual loaded models at runtime.

## How to use the saved/forked orchestration repos

Use these as references, not foundations:

- `A1Holdings/agent-orchestrator`
  - Borrow worktree/session isolation, dashboard, plugin slots, feedback loops.
  - Do not treat PR/CI orchestration as the full local model runtime scheduler.

- `A1Holdings/NemoClaw`
  - Borrow policy-enforced sandboxing, status/logs commands, provider routing,
    and declarative security posture.
  - Do not depend on the alpha stack for the first local desktop version.

- `A1Holdings/composio`
  - Borrow tool catalog/auth/workbench ideas.
  - Keep local policy enforcement inside the vMLX/MLXStudio harness.

- `A1Holdings/OpenSandbox` and `A1Holdings/E2B`
  - Borrow sandbox lifecycle and artifact patterns.
  - Prefer a local, policy-scoped sandbox path for offline workflows.

- `A1Holdings/mem0`, `byterover-cli`, and Letta-style systems
  - Borrow memory blocks and recall APIs.
  - Require audited, replayable memory writes.

- `A1Holdings/browser-use`, `crawl4ai`, `firecrawl`
  - Borrow browser/web task decomposition and extraction patterns.
  - Route all web actions through the same tool gateway and approval policy.

- `A1Holdings/hermes-agent`, `agentscope`, `sim`, `multica`
  - Borrow UX and agent-team abstractions selectively.
  - Avoid adopting a second control plane before the local scheduler/event log
    exists.

## Validation gates

### Inference gates

Before vMLX becomes the default provider for agent runs:

- Chat Completions streaming and non-streaming visible output.
- Responses streaming and non-streaming visible output.
- `previous_response_id` continuation.
- Anthropic and Ollama parity where claimed.
- required tool, auto tool, no-tool, tool-result continuation.
- raw markup leak checks.
- strict JSON/XML/code/whitespace exactness.
- cold miss and warm hit with `cached_tokens` and `cache_detail`.
- block disk L2 write and fresh-process/restart restore.
- cancellation cleanup.
- largest-context tail inspection.
- media rows only where runtime is actually wired.
- installed-app/MLXStudio settings parity.

### Harness gates

- Create two agents pinned to different sessions.
- Concurrently create/start many sessions and prove unique ports.
- Start a multi-step run, kill/restart Electron main process, and verify
  persisted run state.
- Cancel queued, streaming, tool-running, and session-stopped runs.
- Verify tool sandbox denial/truncation/timeout events are persisted.
- Verify cache namespaces prevent cross-agent reuse unless explicitly shared.
- Verify scheduler admission denials include explicit reasons.

### Orchestration gates

- Every graph node accepts and emits valid typed state.
- Replay from each checkpoint.
- Human approval can approve, reject, or modify tool arguments.
- Tool side effects are idempotent or guarded by a side-effect ledger.
- Verifier catches bad tool args, malformed JSON, bad code syntax, and missing
  evidence.
- Resource scheduler prefers warm safe models but rejects unsafe capability
  mismatches.
- Every run records model identity, provider, parser, cache state, token usage,
  tool policy, tool versions, artifacts, and final output.

## Implementation sequence

### Phase 1: Documentation and contracts

1. Land this v1 plan.
2. Define provider adapter interfaces.
3. Define agent/run/event schemas.
4. Define the minimal graph node contract.
5. Define tool policy and approval contract.

### Phase 2: Harness MVP

1. Add `agents`, `agent_runs`, `agent_run_steps`, `agent_run_events`, and
   `agent_artifacts`.
2. Implement `AgentManager`.
3. Implement `RunManager`.
4. Wrap the existing chat/tool loop as `ExecutionAdapter`.
5. Add run cancellation that fans out to local abort, engine cancel, tool
   cleanup, and lease release.

### Phase 3: Scheduler MVP

1. Add capability registry for sessions/models.
2. Add admission checks for model status, context, tools, media, memory, and
   concurrency.
3. Add per-session and per-tool concurrency limits.
4. Record admission events and denial reasons.

### Phase 4: Orchestrator MVP

1. Implement deterministic graph runner with checkpoint after each node.
2. Add planner, worker, verifier, and synthesis nodes.
3. Add blackboard/artifact writes.
4. Add approval-required tool flow.
5. Add replay from checkpoint.

### Phase 5: Evaluation and UI

1. Add trajectory evals for tool calls, routing, handoffs, cancellation, and
   verifier behavior.
2. Add a Runs UI for live timeline, queue, events, artifacts, and replay.
3. Add resource scheduler UI.
4. Add local trace export.

## Non-goals for v1

- No release packaging/signing/notarization work.
- No public download updates.
- No hidden model exactness repair.
- No parser-side tool argument rewriting to hide bad model outputs.
- No full Temporal/Nemo/Agent-Orchestrator dependency as the core.
- No unbounded autonomous group chat.
- No media support claims from preserved weights alone.

## Current release boundary

This plan must not be read as a release-ready claim. The active release rows
remain open until current proof artifacts say otherwise.

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

## Final decision

The best plan is to build a small native orchestration layer inside or beside
MLXStudio, using vMLX as the preferred inference provider and MLXStudio as the
local process/session/tool/cache harness.

The system should be provider-abstracted, graph-driven, event-sourced,
resource-aware, and strict about tool/model exactness. The key differentiator is
not another agent chat framework. It is a local-first control plane that knows
which Apple Silicon model is loaded, which parser is safe, which cache family is
valid, which tools are allowed, how much memory is available, and how to replay
every model/tool/scheduler decision after a crash.
