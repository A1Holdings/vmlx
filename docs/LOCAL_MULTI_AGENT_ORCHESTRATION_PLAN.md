# Local Multi-Agent Orchestration System — Finalized Build Plan

**Status:** Research + architecture decision (not yet implementation)
**Date:** 2026-06-09
**Scope:** Choose the foundations and finalize the plan to build a **local (Apple-Silicon, privacy-preserving, no cloud API fees) multi-agent orchestration system** on top of the org's existing forks.

> This document reviews the proposal *"use vMLX to build a local multi-agent orchestration system,"* evaluates all three MLX-engine forks plus the wider fork inventory in `A1Holdings`, pulls in external/NVIDIA orchestration ideas, and finalizes a concrete three-layer plan. It is the synthesis of three parallel research passes: **(1) local inference layer, (2) agent-harness layer, (3) orchestration layer.**

---

## 0. TL;DR verdict on the proposed plan

**The proposal is sound. vMLX is the correct inference foundation — but only as one of three layers, and it needs concurrency work before it can serve a fleet.** The naive reading ("build the whole orchestration system *inside* vMLX") is wrong; vMLX is a single-model inference *server*, not an orchestrator. The right shape is a clean **three-layer stack** where each layer is taken from a different, best-fit fork we already have:

| Layer | Recommendation | Source fork | License |
|---|---|---|---|
| **L1 — Inference** | **vMLX**, run with continuous batching enabled, fronted by a new headless multi-model gateway | `A1Holdings/vmlx` | Apache-2.0 |
| **L2 — Harness** (per-agent loop) | **Codex** primary, **Claude Code (Agent SDK)** secondary | `A1Holdings/codex`, `A1Holdings/claude-code` | Apache-2.0 / proprietary-use |
| **L3 — Orchestration** | **agent-orchestrator** (build thin on top), made deterministic | `A1Holdings/agent-orchestrator` | MIT |

Three engineering facts drive the whole design:

1. **vMLX is already a 4-protocol endpoint** (OpenAI Chat Completions, OpenAI Responses, Anthropic Messages, Ollama). Verified in `vmlx_engine/server.py` (`/v1/chat/completions`, `/v1/messages`, `/v1/responses`, `/api/chat`). This means **vMLX *is* the universal model shim** — every candidate harness can point at it directly, and we do **not** need a separate translation router (ClawRouter/factory-cursor-bridge) for a local-only build.
2. **vMLX defaults to `max_num_seqs = 1`** (verified `vmlx_engine/scheduler.py:159`; CLI clamps to 1 for some models). It is tuned for single-user local chat. Multi-agent concurrency is a *config + validation gap*, not an architectural wall, but it is the #1 thing to fix and load-test.
3. **All three MLX forks are complementary, not competing.** vMLX is the server; `mlx-vlm` is already vMLX's embedded vision runtime (a dependency); `dflash-mlx` is a single-stream latency accelerator that is *mutually exclusive with batching* and therefore only an optional side-pool.

---

## 1. The three MLX-engine forks (the question that was asked)

The org has forks of three different MLX engines. The proposal only named vMLX; the other two matter for how the inference layer is composed.

### 1.1 `A1Holdings/vmlx` ← `jjang-ai/vmlx` — **the foundation**
- **What:** Python/FastAPI self-hosted inference *server* for LLM/VLM/omni/image-gen/embeddings/rerank/TTS-STT on MLX/Metal. Apache-2.0, v1.5.56, "Beta." Built on `mlx-lm` and `mlx-vlm`.
- **Why it wins:** It is the only fork that is *simultaneously* an OpenAI/Anthropic/Ollama server **and** a continuous-batching scheduler **and** a 5-layer KV cache (paged + memory/prefix + **L2 disk that survives restart** + KV quant + TurboQuant codec) **and** the broadest model zoo (90+ family configs), all on Apple Silicon. Its differentiators that matter for an agent fleet:
  - **JANG/JANGTQ ultra-compression** → fit a *larger, stronger* shared model in unified memory so more agents share one good model.
  - **L2 disk cache survives restart** (`disk_cache.py`, `block_disk_store.py`) → long-lived agents keep warm prefixes across restarts.
  - **Tool + reasoning parsers for ~13 model families** and MCP as a hard dependency.
- **Maturity caveat:** `AGENTS.md` ledger reports `prepackage_ready=false` / `release_ready=false`, but the open blockers are **per-exotic-model-family correctness/quality and packaging/signing** (MiMo JANGTQ2 exactness, MiniMax #179, DSV4 code quality, Step-3.7 VLM), **not** core scheduler/cache/API defects. The serving substrate is the mature part.

### 1.2 `A1Holdings/mlx-vlm` ← `Blaizzy/mlx-vlm` — **embedded vision complement**
- It *is already a dependency* of vMLX (`pyproject.toml`: `mlx-vlm>=0.5.0; darwin`). vMLX routes VLM/omni traffic through its own `mllm_scheduler.py`/`models/mllm.py` using mlx-vlm model classes.
- **Decision:** keep it as the embedded runtime; **do not** run its standalone server (it is single-request, no batching). Keep vMLX's pin current with upstream so new VLM families are available. Vision agents (screenshot/OCR/UI) go through vMLX.

### 1.3 `A1Holdings/dflash-mlx` ← `Aryagm/dflash-mlx` — **optional latency side-pool**
- Exact (bit-for-bit) speculative/block-diffusion decoding for **single-stream** latency. **Architecturally incompatible with batching** — confirmed by vMLX's own `speculative.py::should_use_speculative()` which returns `False` when batched. Narrow model support (Qwen3-4B, partial Qwen3.5) and needs matched draft checkpoints.
- **Decision:** *optional* dedicated latency lane for one foreground/interactive "lead" agent, only if benchmarks beat vMLX's built-in speculative/prompt-lookup decoding (`vmlx_engine/speculative.py`, `prompt_lookup.py`). Do **not** adopt speculatively.

### 1.4 Other inference forks considered (and rejected as the backend)
`claude-code-local` (a *wiring blueprint*, not an engine), `exo` (scale-out for model **size**, not request count), `anemll-flash-llama.cpp` (its flash-MoE idea is already absorbed into vMLX as "Smelt"), `qvac-fabric`/`bitnet`/`luce-megakernel` (different runtime / CPU / **CUDA**), `ollama` (viable GGUF fallback but loses MLX/JANG edge), `lmstudio-js` (client SDK), `unsloth`/`tinygrad` (training/low-level). **Documented hedge:** upstream `vllm-mlx` / `vllm-project/vllm-metal` offer the same API surface and are a credible drop-in if vMLX maturity blocks production.

---

## 2. Layer 1 — Local inference (vMLX), and what must be added

vMLX is the right foundation, but it is built and defaulted for single-user chat. The gaps for fleet serving, in priority order:

| # | Gap | Evidence | Fix to build |
|---|---|---|---|
| **1** | **Concurrency defaulted off** (`max_num_seqs=1`) | `scheduler.py:159`, CLI clamp `cli.py:387-390` | Serve with `--continuous-batching --max-num-seqs N` (+ `--completion-batch-size`, `--prefill-batch-size`); **load-test N=4/8/16** and find the unified-memory ceiling. No in-repo N-parallel proof exists. |
| **2** | **No headless multi-model routing** in the Python engine | Routing/hot-swap lives in the Electron app (`panel/src/main/api-gateway.ts`); one Python process = one model | **Build a headless gateway** that owns N vMLX processes (one per served model), routes by model name, handles spawn/health/JIT-load. *Largest net-new item.* |
| **3** | **Weak per-agent isolation/fairness** | Shared paged/prefix/L2 caches, FCFS scheduling, no per-agent quota | Add scheduler priority/fairness, per-agent cache-key salt, gateway-level rate limits + global concurrency cap. |
| **4** | **Latency vs throughput is either/or per process** | Batching excludes speculative/SimpleEngine (`speculative.py`) | **Two-pool design:** a batched pool for the worker fleet + a small optional latency pool (vMLX SimpleEngine+spec or dflash-mlx) for one interactive lead. |
| **5** | **Maturity / release-lock** | `AGENTS.md` `release_ready=false` | **Pin a known-good commit**, standardize on 2–3 vetted families (Qwen3 / Gemma / Llama), run our own acceptance suite; ignore exotic-quant blockers we won't use. |
| **6** | **Vision-with-images bypasses KV cache** | `docs/ARCHITECTURE.md` compat matrix ("VLM+images → cache SKIP") | Capacity-plan vision agents separately; reuse `vision_embedding_cache.py` where possible. |

**Hard limit to respect (applies to *all* MLX servers, including vMLX):** current MLX `KVCache` cannot fully do block-level paged attention; realistic scaling is ~**4× aggregate throughput at ~16 concurrent**, not NVIDIA-vLLM-class. One Mac is a single shared bottleneck — this constrains fleet size.

**Free wins vMLX already gives us (do not rebuild):** OpenAI+Anthropic+Ollama+Responses wire compatibility, SSE streaming with keep-alive, ~13 tool/reasoning parsers, prefix+paged+L2-disk caching, KV quant, hybrid-SSM handling, JANG compression, request cancellation (`/v1/.../cancel` routes verified in `server.py`), sleep/wake.

---

## 3. Layer 2 — Agent harness (per-agent execution loop)

The harness is the per-agent ReAct/tool-use loop: takes a task + tools, drives the model, executes shell/file/browser/MCP tools, manages context, returns a result. Pointed at vMLX.

### 3.1 Recommendation: standardize the *contract*, support **two** engines

| Rank | Harness | Why | Local wiring |
|---|---|---|---|
| **1 (primary)** | **`A1Holdings/codex`** (Rust, **Apache-2.0**) | Native local providers (`ollama`/`lmstudio` crates), `codex exec --json` + `--output-schema` + `resume`, TS/Python SDK with `runStreamed`/resume, persistent `app-server` JSON-RPC, **best OS sandbox** (`linux-sandbox`/seatbelt/`execpolicy`), strong MCP. Cleanest license to ship on. | `model_providers.vmlx { base_url="http://localhost:8000/v1", wire_api="responses" }` → vMLX `/v1/responses`. **Caveat:** this fork removed `wire_api="chat"`; it is **Responses-API only**, so validate Codex↔vMLX `/v1/responses` (tool items, reasoning items, `previous_response_id`) end-to-end first. |
| **2 (secondary)** | **`A1Holdings/claude-code`** (real CLI, via Agent SDK) | Best raw agent quality; cleanest headless flags (`claude -p`, `--output-format stream-json`, `--json-schema`, `resume`, `claude agents --json`, hooks, subagents). | `ANTHROPIC_BASE_URL` → vMLX `/v1/messages` (exactly how `claude-code-local` works). Closed engine + Anthropic ToS — *use, don't embed/modify*. MCP tool-search needs `ENABLE_TOOL_SEARCH=true` on non-first-party hosts. |

**Route by task type:** Codex for autonomous, sandbox-heavy, long edit/CI loops; Claude Code for highest-quality reasoning/review. (Mirrors the common "Codex reasons better, Claude Code writes cleaner code" split.)

### 3.2 Required headless contract (orchestrator ↔ harness)
Each worker harness must support, as an SDK/library call (not TUI scraping): **spawn** (task + cwd + tool allow-list) · **stream** structured events (tool calls/tokens/usage) · **cancel/timeout** · **structured result** (JSON schema) · **resume** session · **tool gating/approval callback** · **sandbox isolation** · **point at local model**. Both Codex and Claude Code clear this bar (mapping table in the harness research; Codex's `app-server` is the cleanest long-lived fit, Claude Code's `claude agents --json` gives ready-made parallel monitoring).

### 3.3 Harness forks rejected / role-limited (with reasons)
- **`openclaude`** (full Claude-Code-equivalent loop in TS, locally modifiable) — **license/IP risk** (derived from the March-2026 leaked Anthropic source, *no license*). Research/internal only; do **not** ship a product on it.
- **`claw-code-`** (Rust clean-room harness) — promising but **too immature** and unlicensed.
- **Routers `ClawRouter` / `factory-cursor-bridge`** — **redundant**: vMLX is the shim. ClawRouter's reason-to-exist is paid cloud (USDC/x402), the opposite of "no cloud fees." Add a *local* capacity-router only if we run multiple vMLX instances.
- **Workflow/skill layers `oh-my-codex`, `gstack`, `superpowers`, `everything-claude-code`, `*-skills`** — ride *on top of* a harness; adopt selectively as skill/role/methodology content (notably `oh-my-codex`'s `$team`/`$ralph` multi-agent patterns over Codex).
- **Prompt corpora `claude-code-prompts`, `system-prompts-and-models-of-ai-tools` (GPL-3.0), `CL4R1T4S` (AGPL-3.0)** — mine for battle-tested tool-use prompting to harden fragile local-model tool calls; mind copyleft/leak provenance, don't vendor verbatim into a closed product.
- **`hermes-agent` / `hermes-agent-self-evolution`** — reference for memory/self-improvement; the GEPA/DSPy evolution is an *offline optimization* layer for later, not a worker harness.

---

## 4. Layer 3 — Orchestration (the brain)

The orchestration layer plans/decomposes tasks, spawns and supervises parallel agents, isolates their work, routes CI/review/merge feedback, manages shared memory, and gives the user observability/control.

### 4.1 Recommendation: build **thin on `A1Holdings/agent-orchestrator`** (fork of `ComposioHQ/agent-orchestrator`, MIT)
It is purpose-built for this exact problem and is the closest match in the org:
- **Two-tier model:** a read-only *orchestrator agent* spawns *worker sessions* via the `ao` CLI; workers do all implementation (clean supervisor/worker split, enforced by prompt).
- **8 swappable plugin slots** (`packages/core/src/types.ts`): Runtime (`tmux`|docker|k8s|process), **Agent (`claude-code`|`codex`|`aider`|`opencode`)**, Workspace (`worktree`|clone), Tracker (github|linear), SCM, Notifier, Terminal, Lifecycle.
- **Git-worktree isolation per session** (own branch + PR + tmux, hash-namespaced so checkouts never collide).
- **Event-driven reactions:** `ci-failed → send-to-agent`, `changes-requested → send-to-agent (escalateAfter)`, `approved-and-green → auto-merge`; plus a `recovery/` self-healing subsystem and an SSE observability dashboard (`:3000`).
- **The decisive local-fit fact:** workers launch via `Agent.getLaunchCommand()` + **`Agent.getEnvironment()`** (`types.ts:293-296`) — the natural injection point for `ANTHROPIC_BASE_URL`/`OPENAI_BASE_URL`/`model` pointing at vMLX. "Local model" is just an env var on the worker.

**Required change:** the LLM **decomposer hardcodes the cloud path** (`decomposer.ts`: `new Anthropic()`, `claude-sonnet-…`) and parses JSON loosely (throws on bad output). Repoint it to vMLX (`/v1/messages` or `/v1/chat/completions`) **and** make control flow deterministic (below).

**Runner-up:** `multica` (Go + Next.js + pgvector + agent daemon) — borrow its pgvector skill/memory store, WebSocket streaming, and task-lifecycle state machine; too heavy to adopt whole for one private Mac. `agentscope` = best source of reusable Python primitives (message hub, planner, memory, A2A, OTel) if we outgrow the TS core. `sim`/`paperclip` = control-plane UX ideas (visual DAG canvas; org-chart/budgets/heartbeats/audit log). `OpenManus`/`odysseus`/`agency-agents` = worker/persona references only.

### 4.2 Orchestration architecture to adopt (synthesized from external + NVIDIA research)
- **Supervisor/lead + parallel workers** with **git-worktree isolation** (Anthropic multi-agent research system; CAID paper; every 2026 parallel-coding orchestrator — Conductor, Vibe Kanban, Claude Squad, Bernstein).
- **Deterministic control flow.** Given local-model planning unreliability, keep the *task graph, scheduling, and merge queue in plain code* (Bernstein/Temporal style); use the LLM only for decomposition + leaf reasoning. Add NeMo-style `parse_agent_response_max_retries`, schema-constrained decoding, and persist the plan to durable store (survives context truncation).
- **Structured signaling only.** Manager↔worker communication is **structured JSON + git commits, never free-form chat** (CAID identifies free-form inter-agent dialog as the primary failure mode).
- **Merge-queue serialization + cross-model diff review** before landing; inject per-worker `PORT`/service namespaces (worktrees alone leave port/DB collisions). The cross-model diff review is an **autonomous LLM reviewer**, not a human approval — it accepts/rejects/sends-back automatically.
- **DAG dependency scheduling, fully autonomous.** Decomposition runs without a human-approval gate (`requireApproval: false` / removed). The validity bar is enforced by *automated* checks instead — schema-constrained decomposition output, the cross-model diff reviewer, and the `terminal-bench` quality gate — so the pipeline never blocks on a human.

> **Autonomy principle (design constraint):** the system is **fully autonomous end to end** — no human-in-the-loop approval anywhere on the hot path (not on decomposition, not on egress, not on merges). All gates are *automated* policy/checks. A human can still inspect after the fact via the dashboard/audit log, but nothing waits for a person.

### 4.3 NVIDIA ideas to borrow (`A1Holdings/NemoClaw` + NeMo Agent Toolkit)
*(adapted for full autonomy — we take the routing/blueprint/workflow patterns but drop the human-approval and telemetry pieces)*
- **Routed `inference.local` gateway:** agents always talk to one fixed hostname; the gateway owns routing/credentials and maps it to the Mac MLX server. NemoClaw even validates the endpoint by probing `/responses`→`/chat/completions`→`/v1/messages` — exactly vMLX's APIs. **This is our inference gateway pattern.**
- **Egress: static, pre-declared allowlist enforced autonomously — NO human-in-the-loop approval.** Keep deny-by-default *only* as an automatic guardrail so a worker can't silently exfiltrate to the cloud (which would break the "local-only, no cloud fees" goal). Unlisted destinations are auto-denied (and logged), never escalated to a person. The operator sets the allowlist once in config (e.g., `github.com` for `git`/`gh`, package registries); at runtime there are zero prompts. *(If you'd rather not constrain egress at all, this whole guardrail can be disabled — it is not required for autonomy, only for the local-only privacy guarantee.)*
- **Declarative, digest-verified "blueprint" lifecycle** (`plan → apply → status → rollback`) for reproducible, auditable agent infra — applied automatically, no approval step.
- **YAML config-driven workflows** with parse-retries; **systematic eval harness** (`terminal-bench`); **latency-aware routing/priority/caching** for a single backend (Dynamo idea); **NeMo Guardrails** to validate planner/agent outputs on weaker local models. vMLX itself is our local **NIM-equivalent**. *(Skipped on purpose: OpenTelemetry/Phoenix distributed tracing — see §4.5.)*

### 4.5 Observability without OpenTelemetry
Per the autonomy/simplicity preference, **no OpenTelemetry tracing.** Use lightweight, local-only visibility instead:
- **`agent-orchestrator`'s built-in SSE dashboard** (`:3000`) for live session/worker state.
- **A local append-only event/audit log** (plain JSONL or SQLite) written by the orchestrator and gateway — task graph transitions, spawns, merges, tool calls, and rejections. This is enough to debug and replay runs.
- **Per-worker harness logs** (`codex exec --json` JSONL / Claude Code `stream-json`) captured to disk per session.

This keeps full after-the-fact visibility and replayability with zero external collectors, agents, or OTel infrastructure.

### 4.4 Supporting infrastructure to adopt from org forks
| Concern | Adopt | Notes |
|---|---|---|
| **Sandbox/isolation** | tiered: **git-worktree (default) → Docker/Colima → `OpenSandbox`** | `OpenSandbox` (unified protocol, Docker+K8s, ingress/egress). `E2B` is cloud/Linux-first; on Apple Silicon microVM isolation isn't native. |
| **Memory (working)** | **`mem0`** or **`byterover-cli`** | Self-hostable with local vector DB + local LLM; byterover is coding-agent-native (MCP, optional cloud-sync off). |
| **Memory (long-term)** | **`mempalace`** | ChromaDB raw store-everything, fully local, MCP tools. |
| **Vector store** | **`qdrant`** | Rust, lightest/fastest on a single Mac (over weaviate/milvus). |
| **Tools** | **`composio`** (MCP, gate cloud connectors by egress), **`crawl4ai`** (local) over hosted `firecrawl`, **`browser-use`** | Expose to workers via MCP. |
| **Eval/CI gate** | **`terminal-bench`/harbor** | Validate that local-model workers actually complete end-to-end coding tasks before landing. |

---

## 5. Target architecture

```
            ┌───────────────────────────────────────────────┐
   User ──► │  Control plane / Dashboard (AO web UI :3000)    │  ← borrow sim canvas, paperclip audit log
            └───────────────┬───────────────────────────────┘
                            │
        ┌───────────────────▼──────────────────────┐
        │  Orchestrator brain (thin on agent-orch.)  │
        │  • DETERMINISTIC task graph + scheduler    │  ← Bernstein/Temporal style (not LLM-controlled)
        │  • LLM only for decompose + leaf reasoning │  ← decomposer→vMLX, retries + guardrails, plan persisted
        │  • merge queue + cross-model diff review   │  ← CAID/Bernstein
        │  • reactions: ci-failed / changes-req.     │  ← AO reactions + SCM webhooks
        └───┬───────────────┬───────────────┬───────┘
            │ spawn          │ spawn         │ spawn   (3–5 parallel; capped)
   ┌────────▼──┐     ┌───────▼───┐    ┌──────▼────┐
   │ Worker 1  │     │ Worker 2  │    │ Worker N  │  each = AO session
   │ worktree+ │     │ worktree+ │    │ worktree+ │  git-worktree isolation, own branch/PR
   │ branch+   │     │ branch+   │    │ branch+   │  runtime: tmux → Docker/OpenSandbox
   │ HARNESS   │     │ HARNESS   │    │ HARNESS   │  agent: codex (primary) / claude-code (secondary)
   └────┬──────┘     └────┬──────┘    └────┬──────┘
        │  structured JSON + git commits only (no free-form chat)   ← CAID
        ▼                 ▼                ▼
   ┌───────────────────────────────────────────────┐
   │  Inference gateway  (inference.local)           │  ← NemoClaw routed pattern
   │  • global concurrency cap + priority queue      │  ← single-Mac bottleneck control
   │  • route by model name → N vMLX procs           │  ← replaces Electron api-gateway, headless
   │  • static egress allowlist (auto, NO human gate) │  ← optional local-only guardrail
   └───────────────┬───────────────────────────────┘
                   ▼
        ┌──────────────────────┐     Shared services:
        │  LOCAL vMLX server(s) │     • Vision/omni: mlx-vlm (embedded in vMLX)
        │  --continuous-batching│     • Optional latency pool: vMLX SimpleEngine+spec / dflash-mlx
        │  --max-num-seqs N     │     • Memory: mem0/byterover + mempalace + qdrant
        │  OpenAI+Anthropic+    │     • Tools: composio/MCP, crawl4ai, browser-use (egress-gated)
        │  Responses+Ollama     │     • Eval gate: terminal-bench/harbor in CI
        └──────────────────────┘
```

**One-sentence pattern:** a **fully autonomous** supervisor/lead + parallel git-worktree-isolated workers (Codex/Claude Code harnesses), driven by a deterministic orchestrator, all inference routed through one local gateway in front of concurrency-enabled vMLX, with local memory/tools/eval, an optional auto-enforced egress allowlist, and local-only audit logging (no human gates, no OpenTelemetry).

---

## 6. Build roadmap (phased, dependency-ordered — no calendar estimates)

- **Phase 0 — De-risk the bottleneck (must come first).** Pin a known-good vMLX commit. Stand up vMLX with `--continuous-batching --max-num-seqs N`; **benchmark N=4/8/16** on Qwen3/Gemma/Llama to find the real throughput curve and unified-memory ceiling. Validate **Codex↔vMLX `/v1/responses`** end-to-end (tool items, reasoning items, `previous_response_id`) and **Claude Code↔vMLX `/v1/messages`**. *Exit criterion:* a documented concurrency/memory budget and two working harness↔vMLX paths.
- **Phase 1 — Inference gateway.** Build the headless multi-model gateway (`inference.local`): route by model name to N vMLX processes, spawn/health/JIT-load, global concurrency cap + priority queue, per-agent rate limit, and an **optional static egress allowlist enforced automatically (no human approval)**. Replaces the Electron `api-gateway` logic without the GUI.
- **Phase 2 — Orchestrator core.** Fork-rebase `agent-orchestrator`; repoint `decomposer.ts` + `Agent.getEnvironment()` to the gateway; **replace LLM-driven control flow with a deterministic scheduler + merge queue + autonomous cross-model diff review**; disable human-approval gates (`requireApproval: false`) for full autonomy; keep AO's worktree/reaction/recovery/dashboard machinery. Enforce structured-JSON+git-commit signaling.
- **Phase 3 — Memory + tools + isolation.** Layer `mem0`/`byterover` (working) + `mempalace` (long-term) + `qdrant`; expose `composio`/`crawl4ai`/`browser-use` via MCP (gated by the auto egress allowlist if enabled); add Docker/OpenSandbox runtime tier.
- **Phase 4 — Quality gate + local observability.** Wire `terminal-bench`/harbor as the autonomous agent-quality gate; **local-only audit logging (JSONL/SQLite) + the AO dashboard — no OpenTelemetry**; NeMo-Guardrails-style automated output validation for weak local models.
- **Phase 5 (optional) — Latency pool + self-improvement.** Add the single-stream latency lane (vMLX spec / dflash-mlx) if benchmarks justify; introduce offline GEPA/DSPy (`hermes-agent-self-evolution`) skill/prompt optimization from session histories.

---

## 7. Consolidated risk register

| Risk | Severity | Mitigation |
|---|---|---|
| **Local-model planning/tool reliability** (malformed JSON, over-decompose, `blue-cat`→`blue cat` exactness) | High | Deterministic control flow; LLM only for leaf reasoning; schema-constrained decode + retries + guardrails; mine leaked CC/Codex tool prompts; route planning to the strongest local model. |
| **Single-Mac contention** (multi-agent ≈ 15× tokens; ~4×@16 ceiling on MLX) | High | Gateway global concurrency cap + priority queue; bound fleet size; exploit vMLX paged/L2/prefix cache for shared prompts; stagger spawns; backpressure via queue. |
| **vMLX concurrency unproven at N** (`max_num_seqs=1` default) | High | Phase 0 load-test; find memory ceiling before committing. |
| **vMLX maturity / release-lock** (`release_ready=false`) | Medium | Pin commit; standardize on vetted families; own acceptance suite; `vllm-mlx`/`vllm-metal` documented hedge. |
| **No headless multi-model routing** (lives in Electron) | Medium | Phase 1 gateway (largest net-new build). |
| **State/merge consistency across worktrees** | Medium | Git-as-truth; serialized merge queue + cross-model review; per-worker PORT/service namespaces; SQLite/Temporal event log for durability. |
| **Privacy/egress leakage** (cloud tools/telemetry) | Medium | Optional auto-enforced static egress allowlist (no human gate); force `*_BASE_URL` to gateway so no accidental cloud fallback. Guardrail is automatic, never blocks on a person. |
| **Full autonomy without human gates → runaway/bad merges** | Medium | Automated gates replace human ones: schema-constrained outputs, autonomous cross-model diff review, `terminal-bench` quality gate, serialized merge queue, per-task budget caps, and AO `recovery/` self-healing; everything is replayable from the local audit log. |
| **License/IP** | Medium | Build on Apache/MIT (vMLX, codex, agent-orchestrator). **Avoid shipping on leaked-source `openclaude`/`claude-code-*`** (no/unknown license); copyleft prompt corpora not vendored verbatim; real Claude Code = use-only under Anthropic ToS. |
| **Fork staleness** (snapshot forks: orchestration ~2026-03-25, harness ~2026-04, dflash ~2026-04) | Low-Med | Rebase onto upstreams periodically; digest-verify before adopting. |
| **Vision turns bypass KV cache** | Low-Med | Capacity-plan vision agents separately; reuse `vision_embedding_cache.py`. |

---

## 8. License posture (ship-safety)

- **Safe to build a product on:** `vmlx` (Apache-2.0), `codex` (Apache-2.0), `agent-orchestrator`/`mem0`/`qdrant`/`superpowers`/`gstack`/`hermes-agent`/`ECC`/`skills` (MIT/Apache), `mlx-vlm`/`dflash-mlx` (MIT), `agentscope`/`OpenSandbox`/`exo` (Apache-2.0).
- **Use-only (don't embed/modify source):** real `claude-code` (Anthropic Commercial ToS).
- **Avoid as a foundation:** `openclaude`, `claude-code-rev`, `claude-code-sourcemap`, `claude-code-source-code` (leaked Anthropic source; `openclaude` has no license). Treat `claw-code-`, `claude-code-local`, `factory-cursor-bridge`, `oh-my-codex` as all-rights-reserved until license is clarified.
- **Copyleft (don't vendor verbatim into closed product):** `system-prompts-and-models-of-ai-tools` (GPL-3.0), `CL4R1T4S` (AGPL-3.0).

---

## 9. Final answer to the proposal

**Yes — build on vMLX, but as the inference layer of a three-layer stack, not as the orchestrator itself.** Use **vMLX (concurrency-enabled, behind a new headless `inference.local` gateway)** for inference; **Codex (primary) + Claude Code (secondary)** as the per-agent harnesses pointed at vMLX; and **agent-orchestrator (made deterministic, worktree-isolated, structured-signaling)** as the orchestration brain. Keep **mlx-vlm** embedded for vision and **dflash-mlx** as an optional latency side-pool. Borrow NVIDIA's routed-gateway + declarative-blueprint + eval patterns, but run **fully autonomously** — no human-in-the-loop approval anywhere (decomposition, egress, or merges all use *automated* gates), and **no OpenTelemetry** (local JSONL/SQLite audit log + the AO dashboard instead). An optional static egress allowlist is enforced automatically purely to preserve the local-only/no-cloud guarantee, and can be turned off entirely. The dominant constraints are local-model reliability and single-Mac contention — both handled by keeping orchestration control flow deterministic and gating concurrency at the gateway.

---

### Appendix — evidence trail (files/repos actually inspected)
- **vMLX (this repo):** `vmlx_engine/server.py` (4 API routes + cancel routes), `scheduler.py:159/387-392` (`max_num_seqs=1`), `pyproject.toml` (Apache-2.0, `mlx-vlm>=0.5.0`), `docs/ARCHITECTURE.md`, `AGENTS.md` release ledger, `speculative.py`/`prompt_lookup.py`, `disk_cache.py`/`block_disk_store.py`, `flash_moe_integration.py`.
- **Inference forks:** `vmlx`, `dflash-mlx`, `mlx-vlm`, `claude-code-local`, `exo-exolabs-`, `anemll-flash-llama.cpp`, `qvac-fabric-llm.cpp`, `bitnet`, `luce-megakernel`, `ollama`, `lmstudio-js`, `unsloth`, `tinygrad`; web: `vllm-mlx`/`vllm-project/vllm-metal`, `mlx-openai-server`, `mlx_lm.server`.
- **Harness forks:** `codex` (`codex-rs/core/src/model_provider_info.rs`, `lmstudio/src/client.rs`, `exec/src/cli.rs`, `sdk/typescript`), `claude-code`, `openclaude` (`src/services/api/openaiShim.ts`, `providerConfig.ts`), `claude-code-local`, `claw-code-`, `ClawRouter`, `factory-cursor-bridge`, `oh-my-codex`, `gstack`, `superpowers`, `everything-claude-code`, `learn-claude-code`, `hermes-agent`(+`-self-evolution`), `openclaw`, prompt/leak corpora.
- **Orchestration forks:** `agent-orchestrator` (`packages/core/src/types.ts`, `decomposer.ts`, `orchestrator-prompt.ts`, `recovery/*`, `ARCHITECTURE.md`), `multica`, `agentscope`, `sim`, `paperclip`, `OpenManus`, `odysseus`, `agency-agents`, `NemoClaw` (`inference-profiles.md`, `network-policies.md`, `architecture.md`), `OpenSandbox`, `E2B`, `mem0`, `mempalace`, `byterover-cli`, `qdrant`/`weaviate`/`milvus`, `composio`, `browser-use`, `crawl4ai`, `firecrawl`, `terminal-bench`.
- **Web/NVIDIA:** NeMo Agent Toolkit docs + GitHub, NIM, NeMo Guardrails, AI Blueprints; Anthropic multi-agent research system; CAID (arXiv 2603.21489); LangGraph, AutoGen/Magentic-One, CrewAI, OpenAI Swarm/Agents SDK, Temporal; 2026 parallel-coding orchestrator surveys.
