# Technology Guide: What Is Used, Where, and Why

This document covers every technology, library and design technique in the Multi-Agent
Research Assistant. For each one it explains:

- **What it is**, in plain words;
- **Where it is used**, with the exact files, classes and functions;
- **Why it was chosen** over the alternatives;
- **How it works inside this project**, step by step;
- **Pitfalls and decisions** we ran into and how the code handles them.

You can read it top to bottom as a course on the project, or jump to one technology. Section 1
gives the overall picture. Section 2 follows one research question through the whole system and
names each technology as it comes into play. Sections 3–9 go deep on each technology. Section 10
maps the project to the resume bullet points and to common interview questions.

---

## Table of contents

1. [The big picture](#1-the-big-picture)
2. [One request, end to end](#2-one-request-end-to-end)
3. [Python as the base language](#3-python-as-the-base-language)
4. [The LLM layer: Anthropic API and Claude](#4-the-llm-layer-anthropic-api-and-claude)
5. [Orchestration: LangGraph](#5-orchestration-langgraph)
6. [Orchestration: Microsoft AutoGen](#6-orchestration-microsoft-autogen)
7. [Data contracts and configuration: Pydantic and pydantic-settings](#7-data-contracts-and-configuration-pydantic-and-pydantic-settings)
8. [Tools: web search, HTTP, HTML and PDF](#8-tools-web-search-http-html-and-pdf)
9. [Shared memory and the standard library](#9-shared-memory-and-the-standard-library)
10. [Interfaces: Typer, Rich, FastAPI, Uvicorn, SSE](#10-interfaces-typer-rich-fastapi-uvicorn-sse)
11. [Quality: pytest, fakes, Ruff](#11-quality-pytest-fakes-ruff)
12. [Packaging and delivery: hatchling, Docker, GitHub Actions, Make](#12-packaging-and-delivery-hatchling-docker-github-actions-make)
13. [Design patterns used across the code](#13-design-patterns-used-across-the-code)
14. [Mapping to the resume and interview questions](#14-mapping-to-the-resume-and-interview-questions)

---

## 1. The big picture

The project takes a research question such as *"How have solar and wind costs changed since
2010, and why?"* and produces a written report with numbered citations. A team of AI agents
does the work, and each agent has a single job:

- The **Planner** splits the question into a few smaller sub-questions that can be researched
  independently.
- The **Researchers** (one per sub-question, running at the same time) search the web, read
  pages and submit a grounded finding.
- The **Summarizer** shrinks long web pages into short focused notes, so researchers don't
  drown in text.
- The **Critic** reviews all findings, scores them, and either approves them or sends the team
  back to research specific gaps. This is the self-correction loop.
- The **Writer** turns the approved findings into the final report.

Around these agents are the supporting technologies:

| Layer | Technology | Role in one sentence |
|---|---|---|
| Intelligence | **Anthropic API (Claude)** | Every agent's reasoning, tool use and structured output comes from Claude. |
| Orchestration (primary) | **LangGraph** | Decides which agent runs when, runs researchers in parallel, and loops back on critic feedback. |
| Orchestration (alternative) | **AutoGen** | Runs the same agents as a group chat, to show the same system on a second framework. |
| Contracts | **Pydantic v2** | Defines the exact shape of every message that passes between agents. |
| Configuration | **pydantic-settings** | Loads all settings from environment variables or `.env`. |
| Tools | **ddgs / Tavily, httpx, BeautifulSoup, pypdf** | Web search, page download, HTML-to-text, PDF-to-text. |
| Memory | **Python standard library** (`threading`, `difflib`, `json`, …) | Task ledger, knowledge cache and findings blackboard, shared by all agents. |
| Interfaces | **Typer + Rich**, **FastAPI + Uvicorn** | A terminal app with live progress, and a REST API with live event streaming. |
| Quality | **pytest**, **Ruff** | 41 offline tests, plus linting and formatting. |
| Delivery | **hatchling**, **Docker**, **GitHub Actions**, **Make** | Installable package, container image, CI pipeline, shortcuts. |

The guiding principle across all of it: **agents talk through typed contracts and shared
memory, not loose text**, and **every loop has a hard stop**. These two rules are why the
system behaves predictably even though an LLM drives each step.

---

## 2. One request, end to end

Here is what happens, in order, when you run:

```bash
research-assistant run "How have solar and wind costs changed since 2010?"
```

1. **Typer** (`cli.py → run`) parses the command line: the question, `--backend`,
   `--max-revisions` and so on.
2. **pydantic-settings** (`config.py → get_settings`) loads defaults, then `.env`, then
   environment variables. The CLI flags are applied on top through
   `factory.py → build_orchestrator`, which copies the settings with the overrides.
3. `build_orchestrator` creates a **`ResearchOrchestrator`** (LangGraph backend) with a
   **`ClaudeLLM`** (the Anthropic SDK wrapper), a **search provider** (DuckDuckGo via `ddgs`)
   and an **`HttpPageFetcher`** (httpx + BeautifulSoup + pypdf).
4. `ResearchOrchestrator.run` creates a fresh **`SharedMemory`** for this run. It loads the
   persistent cache from disk (`.research_memory/cache.json`), so searches and pages from
   earlier runs are reused. It also creates an **`EventBus`** and subscribes the CLI's printer,
   which uses **Rich** to color each agent's events.
5. `build_graph` compiles a **LangGraph `StateGraph`** with four nodes: `planner`, `research`,
   `critic`, `writer`.
6. **Planner node.** `PlannerAgent.plan` calls `ClaudeLLM.structured`, which sends Claude a
   request with a **strict JSON Schema** built from the **Pydantic** `ResearchPlan` model
   (`schema.py → strict_json_schema`). Claude's response is guaranteed to match the schema and
   is validated back into a `ResearchPlan` object. The sub-tasks go into the **task ledger**
   (`SharedMemory.add_tasks`), which drops near-duplicates using **`difflib`** similarity.
7. **Fan-out.** The routing function `dispatch` reads the pending tasks from the ledger and
   returns one **LangGraph `Send`** per task. LangGraph runs these `research` branches **in
   parallel** (up to 4 at a time).
8. **Each researcher** (`ResearcherAgent.research`) runs `ClaudeLLM.tool_loop`, an agentic loop
   over **Claude tool use**. Claude calls `web_search`, `fetch_page` and `recall_memory`.
   Several calls in one turn run concurrently in a **`ThreadPoolExecutor`**. Every tool goes
   through **`ResearchToolbox`**, which checks the shared cache first, then calls **ddgs** or
   **httpx**. Long pages are condensed by the **`SummarizerAgent`** (another Claude call).
   The researcher finishes by calling the `submit_findings` tool, whose input is a validated
   `FindingDraft`. Any cited URL no tool ever returned is removed.
9. **Join.** LangGraph's **reducer** (`operator.add`) merges each branch's finding into the
   `findings` list in state. Once every branch is done, the `critic` node runs.
10. **Critic node.** `CriticAgent.review` gets a structured `Critique` from Claude, then
    `apply_review_policy` applies the approval rule in Python. If the findings aren't approved,
    the follow-up questions go into the ledger (minus duplicates), and `dispatch` sends only
    the *new* ones back to `research`. This is the **self-correction loop**, capped by
    `max_revisions`.
11. **Writer node.** `WriterAgent.write` numbers the unique sources, asks Claude for a
    structured `ReportDraft` with `[n]` citations, and returns it.
12. The orchestrator builds the final **`ResearchReport`** (Pydantic). It attaches findings,
    critiques, token usage per agent (`UsageTracker`) and memory statistics. It saves the
    cache and a run log to disk with **`json`**.
13. The CLI renders the report as Markdown with **Rich**, saves it to `reports/<slug>.md`, and
    prints a summary table.

The REST API (`api.py`) runs the same `orchestrator.run` in a **background thread pool**. It
streams the same events to the browser as **Server-Sent Events**. The **AutoGen** backend
(`autogen_backend.py`) replaces steps 5–11 with an AutoGen group chat that uses the same
memory, tools and writer.

---

## 3. Python as the base language

**What:** Python 3.10+ (developed on 3.11, CI tests 3.10, 3.11 and 3.12).

**Why Python:** Every major agent framework (LangGraph, AutoGen) and the Anthropic SDK are
Python-first. Python also has the best libraries for web scraping and text processing.
Version 3.10 is the minimum because the code uses the `X | Y` union syntax and modern typing.

**Language features used, and where:**

- **Type hints everywhere**, with `from __future__ import annotations` at the top of modules.
  Hints aren't decoration here: **LangGraph** reads `TypedDict` hints to build its state,
  **AutoGen** reads function hints to generate tool schemas, **FastAPI** reads them to validate
  requests, and **Typer** reads them to parse command-line options.
- **`typing.Protocol`** (`llm.py → LLMClient`, `tools/search.py → SearchProvider`,
  `tools/fetch.py → PageFetcher`). A Protocol describes *what methods an object must have*
  without forcing inheritance. Agents depend on `LLMClient`, not on `ClaudeLLM`, so tests can
  pass in `FakeLLM` and everything just works. This is "structural typing".
- **`dataclasses`** (`ToolSpec`, `ToolOutcome`, `ToolLoopResult`, `UsageTracker`,
  `ResearchTeam`, `FetchedPage`). Lightweight records for internal plumbing where Pydantic's
  validation isn't needed.
- **`TypedDict` and `Annotated`** (`graph.py → ResearchState`). LangGraph requires them (see
  §5).
- **`enum.Enum`** (`models.py → TaskStatus`). Task statuses are a fixed set: `pending`,
  `in_progress`, `done`, `failed`. An enum prevents typos like `"complete"` vs `"done"`.
- **`asyncio`** (`autogen_backend.py`). AutoGen is asynchronous, so that backend runs with
  `asyncio.run(...)`, and blocking tools are moved off the event loop with
  `asyncio.to_thread`. The API's SSE stream is an `async` generator (§10).
- **Closures**. `build_graph` defines the node functions inside itself so they can capture
  `team` and `memory` without global variables. `_build_tools` in the AutoGen backend does the
  same for tool functions.

**Project layout: the `src/` layout.** Code lives in `src/research_assistant/`, not at the
repository root. This forces tests to import the *installed* package, as a user would, and
avoids accidentally importing files from the working directory. It is the layout Python
packaging guides recommend.

---

## 4. The LLM layer: Anthropic API and Claude

### 4.1 What it is

The **Anthropic API** gives programmatic access to **Claude** models. We use the official
Python SDK, the `anthropic` package (version 1.15+), and its `Messages API`: you send a list
of messages, optional tools and options, and get back a message made of *content blocks*
(text, tool calls, thinking).

### 4.2 Where it is used

All direct use is in one file, **`llm.py`**, plus the AutoGen adapter in
**`autogen_backend.py`**. No agent imports `anthropic`. Agents call three methods on the
`LLMClient` interface:

| Method | What it does | Who uses it |
|---|---|---|
| `complete(...)` | Plain text in, text out | `SummarizerAgent` |
| `structured(..., schema=Model)` | Returns an instance of a Pydantic model, guaranteed to match | `PlannerAgent` (`ResearchPlan`), `CriticAgent` (`Critique`), `WriterAgent` (`ReportDraft`), `ResearcherAgent` recovery (`FindingDraft`) |
| `tool_loop(...)` | Runs an agentic tool-calling loop until a "finish tool" is called | `ResearcherAgent` |

### 4.3 Why Claude and why this design

- **Strong tool use and long-context reading.** Research means reading many pages and
  deciding what to call next, which is exactly what Claude is good at.
- **Structured outputs.** The API can *guarantee* the response matches a JSON Schema. That
  removes the most common failure in multi-agent systems: one agent producing text the next
  agent can't parse.
- **One narrow interface (`LLMClient`).** The rest of the code never sees SDK details, so the
  provider is swappable, the agents are testable with fakes, and the API knowledge lives in
  one place.

### 4.4 Model choice and per-agent effort

`config.py` sets two model slots:

- `RA_MODEL` (default `claude-opus-5-5`) for the **planner, critic and writer**. These are
  judgment-heavy steps where quality matters most.
- `RA_WORKER_MODEL` (default `claude-opus-5-5`) for the **researcher and summarizer**. These
  run many times per question, so this is the setting to change (for example to
  `claude-sonnet-5-5`) to reduce cost.

Each agent also gets an **effort level**, sent as `output_config={"effort": ...}` on every
request. Effort controls how deeply the model thinks and how many tokens it spends:

| Agent | Default effort | Reason |
|---|---|---|
| Planner | `high` | A bad decomposition ruins the whole run. |
| Researcher | `medium` | Many tool turns; good balance of quality and cost. |
| Critic | `high` | Judging coverage and consistency needs care. |
| Writer | `high` | Final quality is what the user sees. |
| Summarizer | `low` | A simple extraction task, run often. |

On current Claude models, thinking is **adaptive**: the model decides how much to think, and
`effort` is the control. That is why the code never sends a `thinking` parameter or a
token budget. On these models those parameters are either unnecessary or rejected.

### 4.5 Structured outputs (`ClaudeLLM.structured`)

**How it works:**

1. Take a Pydantic model, for example `Critique`.
2. Convert it to a strict JSON Schema with `schema.py → strict_json_schema` (details in §7.3).
3. Send it as `output_config={"format": {"type": "json_schema", "schema": ...}}`.
4. Claude's text block is now guaranteed to be valid JSON that matches the schema.
5. Parse it with `schema.model_validate_json(raw)`, which also runs our validators (for
   example, clamping a score into the 0–1 range).
6. If the response hit `max_tokens` (truncated), or validation fails anyway, raise a clear
   `LLMError` instead of passing garbage downstream.

**Why not "please reply in JSON" in the prompt?** Prompt-only JSON can come back with extra
prose, missing fields or broken syntax. Schema-constrained decoding removes that whole class of
errors.

### 4.6 Tool use and the agentic loop (`ClaudeLLM.tool_loop`)

**Tool use** means Claude can respond with a `tool_use` block ("call `web_search` with
`{"query": "..."}`") instead of a final answer. Our code runs the tool and sends back a
`tool_result`, and the loop repeats.

**How the loop in `llm.py` works:**

```
messages = [user prompt]
repeat (at most budget + 4 model turns):
    response = Claude(messages, tools)
    append the FULL assistant content to messages      (keeps thinking blocks, see 4.9)
    if Claude called the finish tool (submit_findings): return its input   → done
    if Claude called no tool at all:
        first time  → add "Please call submit_findings now" and continue
        second time → give up (stopped_reason = "no_finish")
    run every tool call in this turn concurrently (ThreadPoolExecutor, max 4)
    calls beyond the budget get an is_error result: "budget exhausted, submit now"
    tool exceptions become is_error results, never crashes
    append ONE user message containing ALL tool_results
```

**Key decisions and why:**

- **Strict tools (`"strict": True`).** Every tool definition (`ToolSpec.to_api`) is strict, so
  Claude's arguments always match the schema. No missing `query`, no wrong types.
- **The finish-tool pattern.** Instead of hoping the final message is well-formed, the
  researcher must call `submit_findings`. The tool's input schema *is* the `FindingDraft`
  model, so the loop ends with validated structured data. Current models don't support
  forcing a specific tool, so the prompt instructs it, and the one-time nudge and the
  transcript recovery (§4.10) cover the rare miss.
- **All tool results in one user message.** When Claude asks for several tools in one turn
  (parallel tool use), all the results must come back together in a single message. Splitting
  them across messages teaches the model to stop making parallel calls.
- **Concurrent execution.** Those parallel calls really do run at the same time (threads),
  since most of their time is network waiting.
- **Errors go back to the model.** A failing fetch (for example a 403) becomes a `tool_result`
  with `is_error: True`. Claude reads it and tries another source. The run never crashes
  because one website blocked us.
- **Why a manual loop instead of the SDK's tool runner?** The SDK has a ready-made tool runner,
  but this project needs controls it doesn't expose: a hard per-task tool budget, a finish
  tool that ends the loop with structured data, and the nudge-then-recover behavior.

### 4.7 Prompt caching

In `ClaudeLLM._create`, every string system prompt is converted to:

```python
[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
```

**What it does:** the API stores the processed prompt prefix, and later requests that start
with the same bytes read it from cache. Cached reads are much cheaper and faster.

**Why it pays off here:** many researchers run in parallel with *the same* system prompt,
and the tool loop resends the same prefix every turn. That's why `prompts.py` keeps prompts
**static**, with no timestamps or run IDs inside. Any changing byte in the prefix would break
the cache. `UsageTracker` records `cache_read_input_tokens`, so you can confirm caching works.

### 4.8 Refusals and server-side fallbacks

Claude can decline a request: the response has `stop_reason == "refusal"`, with a category in
`stop_details`. The code handles this in two layers:

1. **Server-side fallbacks (on by default, `RA_ENABLE_FALLBACKS=true`).** Requests go through
   `client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default")`.
   If the main model declines for a policy reason, the API itself retries on a recommended
   fallback model *within the same call*. Benign research topics can occasionally trip a
   safety classifier, so this keeps runs going.
2. **Explicit check.** If the final response is still a refusal, `_create` raises
   `LLMRefusalError` (with the category) instead of reading empty content.

### 4.9 Thinking blocks and the append-only history

With adaptive thinking, Claude's responses can include `thinking` blocks. In a tool loop,
these must be **sent back unchanged** in later requests. Dropping or editing earlier turns can
invalidate them. That is why the tool loop appends `response.content` as-is, and never just the
text. The AutoGen backend needed special handling for this (§6.6).

### 4.10 Error handling and retries

- **Retries:** the SDK automatically retries rate limits (429), server errors (5xx) and
  connection failures with exponential backoff.
- **Typed exceptions:** `_create` catches errors from most specific to least specific
  (`RateLimitError` → `BadRequestError` → `AuthenticationError` → `APIStatusError` →
  `APIConnectionError`) and turns each into an `LLMError` with a clear message. For example,
  authentication failures say *"Set ANTHROPIC_API_KEY or run `ant auth login`"*.
- **Graceful degradation per sub-task:** if one researcher hits an `LLMError`,
  `ResearcherAgent.research` marks that task `failed`, records a zero-confidence finding that
  explains the gap, and the rest of the run continues.
- **Recovery when no submission arrives:** if the loop ends without `submit_findings`, the
  researcher makes one `structured(..., schema=FindingDraft)` call over the notes it collected
  and lowers its confidence. The event bus records this as a `recovery` event.

### 4.11 Usage tracking

`UsageTracker` (in `llm.py`) is thread-safe and records, per agent, the number of calls and the
input, output and cached tokens, plus tool-call counts. `graph.py → _usage_delta` subtracts a
before-snapshot, so a report shows exactly what *that run* cost. This appears in the report
footer, the CLI summary table and the API JSON.

### 4.12 Authentication

`anthropic.Anthropic()` with no arguments finds credentials automatically:
`ANTHROPIC_API_KEY`, then `ANTHROPIC_AUTH_TOKEN`, then a profile created by `ant auth login`.
The code never hardcodes or logs a key.

---

## 5. Orchestration: LangGraph

### 5.1 What it is

**LangGraph** (version 1.2+) is a library for building LLM workflows as **graphs**: *nodes*
are steps (our agents), *edges* decide what runs next, and a typed *state* object flows
through the graph. It supports loops, conditional routing, and running branches in parallel.
These three features are exactly what a planner → parallel researchers → critic-loop system
needs.

### 5.2 Where it is used

Only in **`graph.py`**: `ResearchState`, `ResearchTaskInput`, `build_graph`, and
`ResearchOrchestrator.run`, which calls `graph.invoke`.

### 5.3 Why LangGraph

- **Explicit, inspectable control flow.** The workflow is code you can read and draw, not
  emergent behavior from agents chatting. That makes it predictable and debuggable.
- **First-class parallel fan-out** with the `Send` API, which is ideal for "one researcher per
  sub-question".
- **Cycles are allowed.** Many pipeline libraries only support straight-line chains. The
  critic → research loop needs a cycle.
- **Reducers** merge results from parallel branches safely.
- **Alternatives:** a hand-written loop would work but means re-implementing parallelism, joins
  and state merging. AutoGen is chat-centric (see §6). LangGraph matches the problem best,
  which is why it is the default backend.

### 5.4 State and reducers

```python
class ResearchState(TypedDict, total=False):
    question: str
    plan: ResearchPlan
    findings: Annotated[list[Finding], operator.add]
    critiques: Annotated[list[Critique], operator.add]
    revision: int
    draft: ReportDraft
    sources: list[Source]
```

- Each node returns a **partial update**, for example `{"findings": [finding]}`, not the whole
  state.
- A **reducer** tells LangGraph how to combine updates. `Annotated[list, operator.add]` means
  *"append, don't overwrite"*. This is what lets four researchers running in parallel each
  add their finding without overwriting each other.
- Keys without a reducer (`plan`, `revision`, `draft`) are simply replaced.
- `total=False` lets keys be absent until a node sets them.

### 5.5 Nodes

All four node functions are defined inside `build_graph`, so they capture the `team`
(the agents) and `memory`:

| Node | Calls | Returns |
|---|---|---|
| `planner_node` | `PlannerAgent.plan`, then `memory.add_tasks(..., origin="planner")` | `plan`, `revision=0` |
| `research_node` | `ResearcherAgent.research(task, question)` | `{"findings": [finding]}` |
| `critic_node` | `CriticAgent.review(...)`, and if rejected, `memory.add_tasks(follow_ups, origin="critic")` | `{"critiques": [critique], "revision": n+1}` |
| `writer_node` | `WriterAgent.write(...)` | `draft`, `sources` |

Note that `research_node` receives a **`ResearchTaskInput`** (the question plus one task), not
the whole state. That is how `Send` works (next section).

### 5.6 Conditional edges and `Send` (parallel fan-out)

```python
def dispatch(state):
    if latest critique approved:   return "writer"
    pending = memory.pending_tasks()
    if not pending:                return "writer" if critiques else "critic"
    return [Send("research", {"question": ..., "task": t}) for t in pending]
```

- `add_conditional_edges("planner", dispatch, [...])` and
  `add_conditional_edges("critic", dispatch, [...])` attach this *same* routing function after
  both the planner and the critic.
- Returning a **node name** routes to that node.
- Returning a **list of `Send` objects** starts *one `research` branch per pending task*, each
  with its own input. This is the "map" step of map-reduce.
- `add_edge("research", "critic")` is the "reduce" step. LangGraph waits until *every* branch
  in the step finishes, then runs `critic` **once** with all findings merged by the reducer.

### 5.7 The self-correction cycle

`critic → dispatch → research → critic` is a **cycle in the graph**. It stops when:

1. the critic approves, **or**
2. `max_revisions` is reached (the critic's final round discards follow-ups), **or**
3. the critic's follow-ups were all duplicates of earlier tasks, so there is nothing pending
   and `dispatch` goes to the writer.

Rule 3 is what "reduce redundant iteration" means in practice: the ledger, not the prompt,
guarantees the team never re-researches the same question.

### 5.8 Runtime configuration

```python
graph.invoke({"question": question}, config={
    "max_concurrency": 4 if parallel_research else 1,
    "recursion_limit": 10 + 3 * (max_revisions + 1),
    "run_name": run_id,
})
```

- **`max_concurrency`** caps how many research branches run at once. This protects API rate
  limits and the search provider. `--sequential` sets it to 1.
- **`recursion_limit`** is LangGraph's safety net against infinite loops. It is computed from
  `max_revisions`, so it is high enough for legitimate loops but always finite.
- **`run_name`** labels the run for tracing tools.

### 5.9 Why the task ledger lives in memory, not in graph state

The graph state holds the *results* (findings, critiques, plan). The **task ledger** (statuses,
attempts, duplicates) lives in `SharedMemory`. There are two reasons:

1. The AutoGen backend uses the exact same ledger, so it must not depend on LangGraph.
2. Researchers update task status *during* their work (pending → in_progress → done), which
   suits a thread-safe shared object better than end-of-node state updates.

---

## 6. Orchestration: Microsoft AutoGen

### 6.1 What it is

**AutoGen** (`autogen-agentchat` and `autogen-ext`, version 0.7.5+) is Microsoft's framework
for multi-agent **conversations**: agents are chat participants that take turns in a group
chat and can call tools.

### 6.2 Where it is used

Only in **`autogen_backend.py`**, selected with `--backend autogen` or
`"backend": "autogen"` in the API. It is an **optional extra**
(`pip install "research-assistant[autogen]"`). `factory.py` imports it lazily and prints
an install hint if it is missing.

### 6.3 Why include AutoGen as well as LangGraph

- **It shows the design is framework-independent.** The agents' *capabilities* (tools, shared
  memory, writer, review policy) are separate from *orchestration*. Swapping LangGraph for
  AutoGen only replaces the turn-taking layer.
- **It shows the two paradigms side by side:** graph-based workflow (LangGraph) vs
  conversation-based collaboration (AutoGen). That's a common interview topic.
- **Both produce an identical `ResearchReport`,** so results are directly comparable.

### 6.4 How the AutoGen team is built

Inside `AutoGenOrchestrator.arun`:

- **Three `AssistantAgent`s**: `planner`, `researcher`, `critic`, each with its own system
  message, tools, and model client (with its own effort level).
- **Tools are plain async Python functions** (`_build_tools`): `register_plan`,
  `get_pending_tasks`, `web_search`, `fetch_page`, `recall_memory`, `submit_finding`,
  `get_findings`, `submit_review`. AutoGen reads each function's **type hints and docstring**
  to create the tool schema automatically. That is why these functions have precise
  signatures, such as `key_points: list[str]`.
- **Shared memory through tools.** AutoGen agents only see each other's *final chat messages*,
  not each other's tool results. So the shared state is exposed as tools: the planner
  *writes* the ledger (`register_plan`), the researcher *reads and updates* it
  (`get_pending_tasks`, `submit_finding`), and the critic *reads the blackboard*
  (`get_findings`) and *adds follow-ups* (`submit_review`). It is the same `SharedMemory`
  object the LangGraph backend uses, so the same de-duplication and caching apply.
- **`max_tool_iterations`** lets an agent call tools several times in one turn: planner 3,
  critic 4, researcher up to 60 (enough to work through all pending tasks). Without this,
  AutoGen allows one tool round per turn and then forces a "reflection". During development
  that broke the critic, which needs `get_findings` and *then* `submit_review`.
- **`reflect_on_tool_use=True`** makes each agent end its turn with a short text summary,
  which becomes its chat message.

### 6.5 Turn-taking: `SelectorGroupChat` with a deterministic `selector_func`

A `SelectorGroupChat` normally asks an LLM "who should speak next?". We pass a
**`selector_func`** instead: a plain Python function that decides the next speaker from the
conversation and the ledger:

```
user          → planner
planner/critic→ researcher if tasks are pending, else critic
researcher    → researcher again if tasks are still pending (bounded), else critic
```

**Why:** it is deterministic and testable, and it saves an LLM call on every turn.
Termination combines **`ExternalTermination`** (set by `submit_review` when the review is
approved, the revision cap is hit, or no new follow-ups were added) with a
**`MaxMessageTermination`** safety net, using AutoGen's `|` (OR) operator. Messages are
consumed with **`team.run_stream`**, and each one is forwarded to our `EventBus`, so the CLI
and the API show live progress for both backends.

After the chat ends, the backend-agnostic **`WriterAgent`** (using `ClaudeLLM`) writes the
report from shared memory, run through `asyncio.to_thread` because it is synchronous.

### 6.6 The `_ClientAdapter`: making AutoGen work with current Claude models

AutoGen's built-in Anthropic client (`AnthropicChatCompletionClient`) was written for older
Claude models. We found three incompatibilities while building the backend and fixed them
with a small wrapper, `_ClientAdapter` / `_MessagesAdapter`, which sits between AutoGen and
the real `AsyncAnthropic` client:

1. **Sampling parameters.** AutoGen always sends `temperature=1.0`. Current Claude models
   reject sampling parameters, so the adapter removes `temperature`, `top_p` and `top_k`, and
   adds `output_config.effort`.
2. **Thinking blocks are lost.** AutoGen rebuilds earlier assistant turns from its own message
   types, dropping thinking blocks and sometimes inserting an empty text block. The API
   rejects empty text, and thinking with tool use needs the original blocks. The adapter
   remembers each response's **original content**, keyed by its tool-call IDs (or its text),
   and puts it back when that turn is resent. The history becomes exactly append-only again.
3. **Usage and refusals.** The adapter records tokens per agent in the shared `UsageTracker`
   and raises `LLMRefusalError` on refusals.

Other details: `ModelInfo(...)` tells AutoGen the model supports function calling and vision,
since AutoGen's built-in list doesn't know newer model names. `tiktoken` is installed only
because `autogen-ext`'s Anthropic module imports it. The test suite's fake client checks on
*every request* that temperature is gone and thinking blocks are restored (§11).

---

## 7. Data contracts and configuration: Pydantic and pydantic-settings

### 7.1 What Pydantic is

**Pydantic v2** defines data classes with type hints and **validates** data against them,
converting JSON to objects and back. Its core is written in Rust, so it is fast.

### 7.2 Where it is used, and why it is central

Pydantic is the **language agents speak** (`models.py`):

| Model | Produced by | Consumed by |
|---|---|---|
| `PlannedSubTask`, `ResearchPlan` | Planner (Claude structured output) | Ledger, critic, writer |
| `SubTask` (+ `TaskStatus` enum) | `SharedMemory.add_tasks` | Researchers, dispatch, selector |
| `SourceRef`, `FindingDraft` | Researcher's `submit_findings` tool input | `Finding` |
| `Finding` | Researcher | Blackboard, critic, writer, report |
| `CritiqueIssue`, `Critique` | Critic (structured output) | Review policy, routing, writer |
| `ReportSection`, `ReportDraft` | Writer (structured output) | `ResearchReport` |
| `UsageStats`, `ResearchReport` | Orchestrator | CLI, API, JSON export |
| `Source`, `PageRecord` | Tools / memory | Cache, report |
| `Event` | `EventBus` | CLI, API SSE |
| `ResearchRequest`, `RunInfo` | API | FastAPI validation and responses |

**Why this matters:** each model is used *three* ways at once. (1) It becomes the **JSON
Schema** Claude must follow. (2) It **validates** Claude's output. (3) It serves as the
**typed Python object** the next agent uses. One definition, no drift between them.

**Pydantic features used:**

- `Field(description=...)`: the description is copied into the JSON Schema, so it doubles
  as **instructions to Claude** for that field (for example, *"Atomic, verifiable facts with
  numbers/dates where possible."*).
- `Field(default_factory=...)` for IDs (`task_ab12cd34`) and UTC timestamps.
- `@field_validator`: clamps `confidence` and `score` into [0, 1]. Strict mode can't enforce
  numeric ranges (see 7.3), so Python enforces them instead.
- `model_copy(update=...)`: immutable-style updates, used when changing a task's status,
  adjusting a finding after source verification, or applying settings overrides.
- `model_validate_json`, `model_dump(mode="json")`, `model_dump_json`: parsing Claude output,
  writing cache and run files, and returning JSON from the API.
- **Inheritance:** `Finding(FindingDraft)` adds bookkeeping fields (IDs, task, revision) to
  exactly what the model submitted, and `ResearchReport(ReportDraft)` does the same for the
  report.
- `ResearchReport.to_markdown()`: the model knows how to render itself.

### 7.3 `schema.py`: from a Pydantic model to a strict schema

Claude's strict mode requires a specific schema shape. Pydantic's default output doesn't match
it, so `strict_json_schema` transforms it:

1. **Inline every `$ref`.** Pydantic puts nested models under `$defs` and references them; we
   substitute the definitions in place.
2. **Every object gets `"additionalProperties": false`**, and all of its properties are listed
   in `"required"`.
3. **Drop keywords strict mode doesn't enforce** (`minimum`, `maximum`, `maxLength`,
   `pattern`, `format`, `default`, `title`, `minItems > 1`). Pydantic validators re-apply
   the rules after parsing.

The tests (`tests/test_schema.py`) walk every generated schema and assert these rules for all
the contract models.

### 7.4 pydantic-settings (`config.py`)

`Settings(BaseSettings)` loads configuration with the prefix `RA_` from the environment and
from a `.env` file. Each field is typed and validated: `max_subtasks` must be between 1 and 20,
and effort must be one of `low | medium | high | xhigh | max` (a `Literal`), and so on. A typo
fails at start-up instead of halfway through a run.

- `TAVILY_API_KEY` is read without the prefix, using `validation_alias`.
- `get_settings()` is wrapped in `functools.lru_cache`, so settings are loaded once.
- CLI flags and API parameters never mutate the global settings. They create a modified copy
  with `model_copy(update=...)` (in `factory.py`), so two API jobs with different options
  can't interfere.

---

## 8. Tools: web search, HTTP, HTML and PDF

### 8.1 `ddgs` (DuckDuckGo search): default search provider

- **Where:** `tools/search.py → DuckDuckGoSearch`.
- **What:** a library that queries DuckDuckGo and returns title, URL and snippet.
- **Why:** **no API key and no cost**, so anyone can clone and run the project immediately.
- **How:** `DDGS().text(query, max_results=n)`. The rows become `Source` models, and any
  library error becomes a `SearchError`, which the toolbox turns into an `is_error` tool
  result.

### 8.2 Tavily: optional search provider

- **Where:** `tools/search.py → TavilySearch`. Enable with `RA_SEARCH_PROVIDER=tavily` and
  `TAVILY_API_KEY`.
- **Why:** a search API designed for LLM agents, with cleaner content and better reliability
  than scraping a search engine. It is optional so the default setup stays free.
- **How:** a direct `httpx.post` to its REST endpoint, with no extra SDK dependency.
- `build_search_provider` picks the provider. Both implement the `SearchProvider` protocol,
  so nothing else changes.

### 8.3 httpx: HTTP client

- **Where:** `tools/fetch.py → HttpPageFetcher.fetch`, `TavilySearch`, and
  `examples/api_client.py` (including SSE streaming).
- **Why:** a modern, well-typed HTTP client with timeouts, redirects and streaming.
- **How it is used:** a browser-like `User-Agent`, an `Accept` header for HTML, PDF and text,
  `follow_redirects=True`, a configurable timeout (`RA_FETCH_TIMEOUT`), and
  `raise_for_status()`. Any HTTP error becomes a `FetchError`. Only `http(s)://` URLs are
  allowed.
- **Note:** the Anthropic SDK 1.x uses its own `httpx2` internally. Our code never passes
  `httpx` objects into the Anthropic client, so the two don't conflict.

### 8.4 BeautifulSoup: HTML to readable text

- **Where:** `tools/fetch.py → html_to_text`.
- **Why:** robust parsing of messy real-world HTML; the standard choice in Python.
- **How:** extract the `<title>`. Remove noise tags (`script`, `style`, `nav`, `footer`,
  `header`, `aside`, `form`, `svg`, `noscript`). Prefer the `<article>` or `<main>` element if
  one exists, otherwise use `<body>`. Get the text with newline separators, then collapse
  runs of blank lines. This removes menus and boilerplate so the model reads content, not
  navigation.

### 8.5 pypdf: PDF to text

- **Where:** `tools/fetch.py → pdf_to_text`.
- **Why:** much authoritative research material (reports, papers, agency publications) is PDF
  only. While testing, the IRENA cost-report web page returned 403, but its PDF summary was
  readable.
- **How:** detected by `content-type` or a `.pdf` URL. Read the first 40 pages, take the
  `/Title` metadata if present, and normalize whitespace. Unreadable PDFs raise `FetchError`.

### 8.6 Page size limits and summarization

- `RA_MAX_PAGE_CHARS` (60,000) caps raw page text and marks truncation explicitly, so
  a giant page can't blow up memory.
- `RA_SUMMARIZE_THRESHOLD_CHARS` (6,000): longer pages are **not** sent to the researcher in
  full. `ResearchToolbox.fetch_page` calls the **`SummarizerAgent`** with the researcher's
  `focus` (*"what do you need from this page?"*) and returns a focused summary of at most
  ~400 words.
- **Why:** it keeps each researcher's context small and relevant, which lowers token cost and
  improves accuracy. Summaries are cached per (URL, focus), so asking the same thing twice is
  free.

### 8.7 `ResearchToolbox`: the single gateway for tools

**Where:** `tools/toolbox.py`. **Every** tool call from **both** backends passes through it.

- `web_search`: check the memory cache, then the provider, then store the result. Cache hits
  are labelled *"(from shared memory -- this query was already run)"*, so the model knows it
  is repeating itself.
- `fetch_page`: check the cache, then fetch, then store (under both the redirected URL and the
  requested URL), then return the text or a focused summary.
- `recall_memory`: search teammates' published findings (the blackboard).
- `TOOL_SPECS`: the strict tool schemas, generated from small Pydantic input models
  (`WebSearchInput`, `FetchPageInput`, `RecallMemoryInput`).

**Why a single gateway:** caching, de-duplication, event logging and error handling are
written once and apply to both orchestration backends.

---

## 9. Shared memory and the standard library

### 9.1 `SharedMemory` (`memory.py`): the three parts

1. **Task ledger** (`tasks: dict[str, SubTask]`): every sub-task with its status, attempts,
   origin (planner or critic), revision, error and timestamps. Methods: `add_tasks`,
   `find_duplicate`, `set_status`, `pending_tasks`, `progress`.
2. **Knowledge cache** (`_searches`, `_pages`): search results and pages with their
   summaries. Methods: `get_search`, `put_search`, `get_page`, `put_page`,
   `add_page_summary`, `known_sources`, `cache_size`, `save_cache`, `clear_cache`.
3. **Blackboard** (`findings`): published findings. Methods: `publish_finding`,
   `all_findings`, `recall`.

This is a classic **blackboard architecture**: independent specialists coordinate by reading
and writing a shared workspace instead of messaging each other directly.

### 9.2 `threading.RLock`: safe parallel access

Researchers run in parallel threads (LangGraph branches plus the tool thread pool). Every
public method of `SharedMemory` holds a **re-entrant lock** (`RLock`). Re-entrant is required
because `add_tasks` holds the lock and then calls `find_duplicate`, which takes the same lock
again. A plain `Lock` would deadlock there. `tests/test_memory.py` starts 50 threads adding
tasks at once and checks that none are lost.

`EventBus` and `UsageTracker` use their own locks for the same reason.

### 9.3 `difflib` plus token Jaccard: duplicate detection

**Where:** `memory.py → similarity`, `find_duplicate`, `recall`.

```
similarity(a, b) = max(jaccard, 0.5·jaccard + 0.5·sequence_ratio)
```

- **Normalization:** lowercase, keep only letters and digits, drop stop-words ("how", "the",
  "of" …).
- **Jaccard similarity** = shared words ÷ all words. This captures "same topic, words in a
  different order".
- **`difflib.SequenceMatcher` ratio** captures near-identical strings, such as the same
  question with different punctuation.
- Taking the `max` means either signal can flag a duplicate. The threshold
  (`RA_DUPLICATE_SIMILARITY`, 0.82) is configurable.
- Failed tasks are ignored during duplicate checks, so they can be retried.

**Why not embeddings?** Embeddings would catch deeper paraphrases but add a model call, an
extra dependency and latency to every comparison. Sub-questions from the same run are usually
close rephrasings, so standard-library similarity is fast, free and deterministic (easy to
test). The `similarity` function is the single place to upgrade later.

### 9.4 `urllib.parse`: URL normalization

`normalize_url` lowercases the scheme and host, removes the `#fragment` and the trailing
slash, and keeps the query string. So `HTTPS://A.org/x/#top` and `https://a.org/x` are one
cache entry and count as one citation.

### 9.5 Anti-hallucination source check

`known_sources()` returns every URL that appeared in any search result or fetched page.
`ResearcherAgent._verify_sources` (and `submit_finding` in the AutoGen backend) **drops any
cited URL not in that set**, adds a note to `gaps`, and lowers `confidence`. A model can't
invent a citation that ends up in the report.

### 9.6 `json` plus atomic writes: persistence

- `save_cache()` writes searches and pages to `.research_memory/cache.json`. It writes to a
  `.tmp` file first and then `replace()`s, which is **atomic**, so a crash mid-write can never
  leave a corrupted cache.
- `_load_cache()` restores the cache at start-up. If the file is corrupt, it logs a warning
  and continues with an empty cache (tested).
- `save_run()` writes the full ledger, findings, progress and stats to
  `.research_memory/runs/<run_id>.json`, an audit trail of every run.

### 9.7 Other standard-library pieces

| Module | Where | Why |
|---|---|---|
| `concurrent.futures.ThreadPoolExecutor` | `llm.py → _run_tools`; `api.py → RunManager` | Run parallel tool calls at the same time; run API jobs in the background without blocking the web server |
| `collections.Counter` | `SharedMemory.stats`, `progress()` | Count cache hits and misses, duplicates skipped, tasks per status |
| `uuid` | IDs in `models.py`, run IDs | Unique, unguessable task, finding and run identifiers |
| `datetime` (UTC) | timestamps in models and events | Time-zone-safe timestamps |
| `re` | text normalization, slugs, PDF and HTML cleanup | Simple, fast text processing |
| `logging` | every module (`logger = logging.getLogger(__name__)`) | Standard logs; `-v` enables debug |
| `functools.lru_cache` | `get_settings` | Load configuration once |
| `operator.add` | LangGraph reducers | Append-merge for parallel results |
| `pathlib` | memory directory, report paths | Cross-platform file paths |
| `io.BytesIO` | `pdf_to_text` | Give pypdf the downloaded bytes without a temp file |

### 9.8 `EventBus` (`events.py`): observability

A small thread-safe publish/subscribe bus. Agents call `emit(type, agent, message, **data)`
for `plan`, `task_started`, `tool_call`, `summarize`, `finding`, `recovery`, `critique`,
`revision`, `report`, `run_started`/`run_finished`, and others. Subscribers:

- the **CLI** prints them in color (Rich);
- the **API** stores them per run and streams them over SSE;
- tests assert on them.

A subscriber that throws is logged and ignored, so a display bug can never crash a research
run. Events are Pydantic `Event` models, so they serialize directly to JSON.

---

## 10. Interfaces: Typer, Rich, FastAPI, Uvicorn, SSE

### 10.1 Typer: the command-line interface

- **Where:** `cli.py`. Installed as the `research-assistant` command through
  `[project.scripts]` in `pyproject.toml`.
- **Why:** builds a full CLI from **type hints**, with help text, validation and subcommands,
  and almost no boilerplate.
- **Commands:**
  - `run QUESTION` with options `--backend`, `-o/--output`, `--json`, `--max-subtasks`,
    `--max-revisions`, `--model`, `--worker-model`, `--sequential`, `--no-memory`,
    `-v/--verbose`, `--show/--no-show`
  - `serve` with `--host` and `--port`
  - `memory stats` and `memory clear` (a sub-app created with `add_typer`)
  - `version`
- **Exit codes:** 0 on success, 1 for LLM errors, 2 for configuration or installation errors,
  130 for Ctrl-C. These are standard conventions, which makes the CLI scriptable.

### 10.2 Rich: terminal output

- **Where:** `cli.py`.
- **Why:** readable, colorful live output, which matters when watching five agents work at once.
- **Used for:** a color per agent in the event stream (`_AGENT_STYLES`), a `Panel` header,
  rendering the final report as **Markdown** in the terminal, a `Table` run summary (critic
  score, revisions, findings and sources, LLM and tool calls, tokens, memory hits, duplicates
  skipped), and `console.rule()` separators.

### 10.3 FastAPI: the REST API

- **Where:** `api.py`. `create_app(factory)` builds the app, and module-level `app` is what
  Uvicorn serves.
- **Why:** type-hint-driven request validation (it uses Pydantic models directly), automatic
  OpenAPI docs at `/docs`, async support for streaming, and an excellent test client.
- **Endpoints:**

| Method and path | Purpose |
|---|---|
| `GET /health` | Liveness check and version |
| `POST /research` | Submit a job: `{question, backend, max_subtasks?, max_revisions?}` → **202** `{run_id}` |
| `GET /research` | List runs |
| `GET /research/{id}` | Status, error and full report JSON (`?include_events=true` adds events) |
| `GET /research/{id}/report.md` | The rendered Markdown report (409 if not finished) |
| `GET /research/{id}/events` | **Server-Sent Events** stream of live agent events |

- **Validation:** `ResearchRequest` requires a question of 5–2000 characters, a backend from
  `langgraph | autogen`, and bounded integers. Bad input returns **422** automatically.
- **Background jobs:** a research run takes minutes, far longer than an HTTP request should
  stay open. `RunManager` therefore returns **202 Accepted** right away and runs the job in a
  `ThreadPoolExecutor` (2 workers). Status moves through `queued → running →
  completed/failed`, and any exception is captured into `error` instead of crashing the
  server.
- **Dependency injection for tests:** `create_app(factory=...)` takes the orchestrator factory
  as a parameter. Tests pass a factory that builds orchestrators with fakes, so the whole API
  is tested offline.

### 10.4 Server-Sent Events (SSE)

- **What:** a simple one-way stream from server to browser over plain HTTP
  (`Content-Type: text/event-stream`), with lines like `event: plan` / `data: {...}`.
- **Why SSE and not WebSockets:** we only need server-to-client updates. SSE is simpler, works
  through proxies, and browsers reconnect automatically.
- **How:** an `async` generator inside a `StreamingResponse` sends any new events every 0.25 s.
  When the run finishes it sends a final `event: end`. The `async` sleep keeps the server free
  to handle other requests while a stream is open.

### 10.5 Uvicorn

- **Where:** `cli.py → serve` and the Docker `CMD`.
- **What:** the ASGI server that actually runs the FastAPI app. `uvicorn[standard]` adds
  faster event-loop and HTTP-parsing libraries.

---

## 11. Quality: pytest, fakes, Ruff

### 11.1 pytest

- **Where:** `tests/`. Configured in `pyproject.toml` (`testpaths`, `pythonpath = ["src"]`,
  `asyncio_mode = "auto"`). `pytest-asyncio` is installed and configured for async tests. The
  current tests call `asyncio.run` through the orchestrator's sync `run()`.
- **Why:** the standard Python test framework; fixtures (`conftest.py`) give each test its
  own temporary memory directory (`tmp_path`).
- **The suite (41 tests, all offline, ~25 s):**

| File | What it proves |
|---|---|
| `test_schema.py` | Every contract model becomes a valid strict schema; nested models inlined; unsupported keywords removed; clamping still works |
| `test_memory.py` | Similarity, duplicate skipping, failed-task retry, status transitions, cache hits and misses, URL normalization, persistence round-trip, corrupt-cache tolerance, recall ranking, 50-thread safety |
| `test_toolbox.py` | Strict tool specs, search caching, verbatim short pages vs summarized long pages (once per focus), fetch errors returned not raised, unknown tool, HTML boilerplate stripping |
| `test_llm.py` | The **real `ClaudeLLM`** against a fake SDK: schema, effort, fallbacks and cache-control in the request; fallbacks can be disabled; invalid output raises; refusal raises; parallel tools run and return in one message; thinking block preserved; tool budget enforced; tool exceptions become `is_error`; nudge once, then stop |
| `test_graph.py` | Full LangGraph runs: approval in round 1, a revision round, revision cap, duplicate follow-ups don't loop, shared-memory cache stops repeat network calls, hallucinated sources removed, transcript recovery, Markdown and persistence, sequential mode, review policy rules |
| `test_autogen_backend.py` | Full AutoGen runs with a revision round and with first-round approval; source verification; per-agent usage; the adapter's content-key logic |
| `test_api.py` | Health; submit → poll → report → Markdown → SSE stream → list; 422 validation; 404 |

### 11.2 Test doubles: how we test LLM systems without an LLM

- **`tests/fakes.py → FakeLLM`** implements the `LLMClient` protocol with scripted,
  schema-aware answers. For example, `critic_scores=[0.4, 0.9]` means "reject, then approve".
  It also plants a **hallucinated URL** in every finding, to prove the verifier removes it.
  `FakeSearch` and `FakeFetcher` serve fixed pages and record every call, which proves the
  cache prevents repeat network calls.
- **`tests/fake_anthropic.py`** goes one level deeper. It returns **real
  `anthropic.types.Message` objects**, so the actual `ClaudeLLM` and the actual AutoGen client
  run their real parsing code. `ScriptedAsyncAnthropic` plays the planner, researcher and
  critic in a full AutoGen run, and **on every request asserts** that no `temperature` was
  sent, `effort` is present, no empty text blocks were replayed, and every assistant tool-call
  turn starts with its thinking block. This is how the `_ClientAdapter` fix is locked in.
- **Why this approach:** tests are fast, free, deterministic and need no API key. Real LLM
  calls would be slow, cost money and give different answers each time, so they're unsuitable
  for CI.

### 11.3 Ruff

- **Where:** `[tool.ruff]` in `pyproject.toml`; run in CI and with `make lint`.
- **What:** an extremely fast linter and formatter (it replaces flake8, isort and black).
- **Rules enabled:** `E`/`F` (errors, undefined names), `I` (import order), `B` (bugbear:
  likely bugs), `UP` (modern syntax), `SIM` (simplifications). Ignored: `E501` (line length is
  left to the formatter) and `B008` (`typer.Option()` defaults are how Typer is meant to be
  used).
- Line length 100, target Python 3.10.

---

## 12. Packaging and delivery: hatchling, Docker, GitHub Actions, Make

### 12.1 `pyproject.toml` and hatchling

- **What:** the modern standard single file for project metadata, dependencies and tool
  configuration. **hatchling** is the build backend that turns it into an installable wheel.
- **Dependencies are split into extras:**
  - core: `anthropic`, `langgraph`, `pydantic`, `pydantic-settings`, `httpx`,
    `beautifulsoup4`, `pypdf`, `ddgs`, `typer`, `rich`
  - `[autogen]`: `autogen-agentchat`, `autogen-ext[anthropic]`, `tiktoken`
  - `[api]`: `fastapi`, `uvicorn[standard]`
  - `[dev]`: `pytest`, `pytest-asyncio`, `ruff`
- **Why extras:** a user who only wants the CLI doesn't install a web server or AutoGen. The
  code imports optional pieces lazily, with helpful error messages.
- `[project.scripts]` creates the `research-assistant` command on install.
- Editable install for development: `pip install -e ".[dev,api,autogen]"`.

### 12.2 Docker

- **`Dockerfile`:** based on `python:3.12-slim`. Installs the package with the `api` and
  `autogen` extras, runs as a **non-root user** (`app`), stores memory in a **volume** at
  `/data` (`RA_MEMORY_DIR=/data/memory`) so the cache survives container restarts, exposes
  port 8000, and starts Uvicorn.
- **`.dockerignore`:** keeps `.venv`, `.env` (secrets), caches, reports and tests out of the
  image, which keeps it smaller and safer.
- **Why:** one command gives an identical environment anywhere. The API key is passed at run
  time (`-e ANTHROPIC_API_KEY=...`), never baked into the image.

### 12.3 GitHub Actions CI (`.github/workflows/ci.yml`)

On every push to `main` and every pull request it runs, on **Python 3.10, 3.11 and 3.12**:
install with all extras → `ruff check` → `ruff format --check` → `pytest`. Because the suite
is fully offline, CI needs **no secrets** and costs nothing per run.

### 12.4 Makefile and supporting files

- `make install | test | lint | format | run Q="..." | serve | docker`: short, memorable
  commands.
- `.env.example`: every setting with its default, documented; copy it to `.env`.
- `.gitignore`: excludes `.env` (secrets), `.venv`, caches, `.research_memory/` and `reports/`.
- `LICENSE` (MIT), `README.md`, `docs/architecture.md`, and this file.
- `examples/quickstart.py` (library use) and `examples/api_client.py` (API plus live SSE).

---

## 13. Design patterns used across the code

| Pattern | Where | Benefit |
|---|---|---|
| **Planner–executor–critic** | The overall agent roles | Separates deciding *what* to do, *doing* it, and *checking* it, with a feedback loop |
| **Map-reduce** | LangGraph `Send` fan-out + `operator.add` reducer + critic join | Parallel research, safe merge |
| **Blackboard** | `SharedMemory` findings + `recall_memory` | Agents build on each other's work without direct messaging |
| **Finish tool** | `submit_findings`, `submit_review`, `register_plan` | Agentic loops end with validated structured data |
| **Typed contracts** | Pydantic models → strict schemas | No parsing errors between agents |
| **Policy in code, judgment in the model** | `apply_review_policy` | The model scores; Python decides approval and loop termination, so the loop always ends |
| **Bounded loops everywhere** | tool budget, nudge-once, `max_revisions`, `recursion_limit`, `MaxMessageTermination`, max model turns | No runaway cost or infinite loops |
| **Dependency injection + Protocols** | `LLMClient`, `SearchProvider`, `PageFetcher`, API `factory`, `anthropic_client` | Swap implementations; test with fakes |
| **Factory** | `factory.py → build_orchestrator`, `ResearchOrchestrator.from_settings` | One place builds a correctly wired system |
| **Adapter** | `_ClientAdapter` (AutoGen ↔ current Claude API) | Use a framework despite incompatibilities, without forking it |
| **Gateway / single choke point** | `ResearchToolbox` | Caching, logging and errors written once for both backends |
| **Publish–subscribe** | `EventBus` | CLI, API and tests observe runs without coupling to agents |
| **Cache-aside** | Toolbox checks memory, then fetches, then stores | Less repeated work and fewer network calls |
| **Graceful degradation** | failed tasks, `is_error` tool results, transcript recovery | One failure doesn't sink the run |
| **Strategy** | Two orchestrators sharing one interface (`run(question, on_event, run_id)`) | Same inputs and outputs, different orchestration |

---

## 14. Mapping to the resume and interview questions

### 14.1 Resume bullets → where each one is implemented

**"Built a multi-agent system in which planner, researcher, and critic agents collaborate
autonomously to complete multistep research tasks."**

- Agents: `agents/planner.py`, `agents/researcher.py`, `agents/critic.py` (plus
  `writer.py`, `summarizer.py`).
- Autonomous collaboration: `graph.py` (LangGraph) and `autogen_backend.py` (AutoGen). After
  the question is given, no human input is needed until the report.
- Multistep: plan → parallel research → critique → follow-up research → report.

**"Designed agent communication and tool-calling logic enabling web search, summarization,
and self-correction loops."**

- Communication: Pydantic contracts and strict structured outputs (`models.py`,
  `schema.py`, `ClaudeLLM.structured`); state reducers and `Send` routing; AutoGen tools over
  shared memory.
- Tool calling: `ClaudeLLM.tool_loop` (strict tools, parallel execution, budget, finish tool,
  error results) and `tools/toolbox.py`.
- Web search: `tools/search.py` (DuckDuckGo / Tavily), `tools/fetch.py` (HTML + PDF).
- Summarization: `SummarizerAgent` triggered by page length, focused and cached.
- Self-correction: critic follow-up loop, hallucinated-source removal, nudge and transcript
  recovery, `is_error` tool recovery.

**"Implemented shared memory/state across agents to track task progress and reduce redundant
iteration."**

- Shared memory: `memory.py → SharedMemory` (ledger, cache, blackboard), thread-safe,
  persisted to disk.
- Task progress: `TaskStatus` transitions, `progress()`, per-run ledger files, events.
- Reducing redundant iteration: duplicate-task rejection at the ledger; search, page and
  summary caching; `recall_memory`; and the loop ends when follow-ups add nothing new.
  Measured in each report: `duplicate_tasks_skipped`, `search_cache_hits`, `page_cache_hits`.

### 14.2 Likely interview questions, with short answers

**Why LangGraph *and* AutoGen?** LangGraph is the primary engine: explicit graph, native
parallel fan-out, cycles, reducers. AutoGen shows the same agents as a conversation-based team.
Both share tools, memory and the writer, so it shows the orchestration layer is replaceable.

**How do agents communicate without parsing each other's text?** Through Pydantic models that
become strict JSON Schemas for Claude's structured outputs, and through shared memory.
Output is guaranteed to match the schema and is validated again on return.

**How do you stop infinite loops and runaway cost?** Every loop is bounded: per-task tool
budget, nudge-once, capped model turns, `max_revisions`, LangGraph `recursion_limit`, AutoGen
`MaxMessageTermination`. The approval rule is enforced in code, and the loop ends as soon as
follow-ups add nothing new to the ledger.

**How do you prevent hallucinated citations?** Only URLs that a tool actually returned can be
cited. Anything else is removed, recorded as a gap, and confidence is lowered. The writer only
sees verified sources, numbered `[n]`.

**How does shared memory reduce redundant work?** Duplicate sub-tasks are rejected with token
Jaccard + `difflib` similarity, and searches, pages and focused summaries are cached and
persisted across runs. Researchers can also `recall_memory` teammates' findings instead of
searching again. The tests show two researchers issuing the same query cause one real search.

**How is it tested without paying for API calls?** A scripted `FakeLLM` drives full graph
runs. A fake Anthropic client returns real SDK message objects, so the actual `ClaudeLLM` and
AutoGen code paths run; it also asserts the request shape on every call. 41 tests, offline,
in CI on three Python versions.

**What was the hardest integration problem?** AutoGen's Anthropic client didn't match current
Claude models: it sent `temperature`, lost thinking blocks, and replayed empty text blocks. A
small adapter fixed this by stripping sampling params, setting effort, and restoring each
turn's original content so the history stays append-only. Tests check this on every request.

**What would you improve next?** Embedding-based duplicate detection; a LangGraph
checkpointer so long runs can resume; a persistent job store for the API (it is in-memory
today); per-domain rate limiting for fetches; and an evaluation set to tune effort levels and
model choice against report quality.
