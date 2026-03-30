# Streamlit UI context persistence for LangChain agent app

**Session ID:** ses_2d2c8a3ccffeu1ihVjwFXiy3MC
**Created:** 3/26/2026, 8:54:56 PM
**Updated:** 3/26/2026, 9:01:41 PM

---

## User

In this directory, on main.py, I'm working on a streamlit UI for a langchain agentic application found in ./graph package. I'm having issues with context persistence while using the Streamlit UI. Propose a way to address this issue

---

## Assistant (Plan · big-pickle · 86.8s)

_Thinking:_

The user is working on a Streamlit UI for a LangChain agentic application and is having issues with context persistence. They want me to propose a way to address this issue.

Let me first explore the codebase to understand:
1. The structure of the project
2. The main.py file and how Streamlit is being used
3. The ./graph package and how the LangChain agent is structured
4. How context/state is currently being managed

Since I'm in plan mode, I should only read and analyze - no edits allowed.

Let me start by exploring the codebase structure and reading the relevant files.

I'll explore the codebase to understand the current implementation and identify the context persistence issue.

**Tool: task**

**Input:**
```json
{
  "description": "Explore codebase structure",
  "prompt": "Explore the codebase at /home/jose/Development/Python/job-hunt-prep/src/job_hunt_prep to understand:\n\n1. The overall project structure - list all directories and key files\n2. Read main.py completely to understand how Streamlit is being used\n3. Explore the ./graph package - read all files to understand the LangChain agent structure\n4. Identify how context/state is currently being managed (look for session state usage, thread IDs, checkpointer patterns, etc.)\n\nReturn a comprehensive summary of:\n- Project structure\n- How main.py sets up the Streamlit UI\n- How the graph/agent works (nodes, state, edges)\n- Current state management approach\n- Any obvious issues with context persistence",
  "subagent_type": "explore"
}
```

**Output:**
```
task_id: ses_2d2c8829dffecr9I4iyfwhKv4b (for resuming to continue this task if needed)

<task_result>
Now I have a comprehensive understanding of the codebase. Let me provide a detailed summary.

---

## Comprehensive Codebase Summary

---

### 1. Project Structure

```
/home/jose/Development/Python/job-hunt-prep/
├── .env                          # LangSmith API keys for tracing
├── .python-version               # Python 3.11.5
├── pyproject.toml                # Dependencies: langchain, streamlit, chroma, ollama, selenium, etc.
├── draft.py                      # Scratch file for testing checkpointer patterns
├── data/                         # User documents + vector store
├── src/job_hunt_prep/
│   ├── __init__.py
│   ├── main.py                  # Streamlit frontend (THIS IS THE ENTRY POINT)
│   ├── ingestion.py             # Document ingestion pipeline (PDF/TXT/OTHER -> Chroma)
│   ├── frontend/__init__.py     # Empty package marker
│   ├── graph/
│   │   ├── __init__.py          # Exports: Agent, LLM, Retriever
│   │   ├── agent.py             # Agent class (main orchestrator)
│   │   ├── state.py             # State schema definitions
│   │   ├── nodes.py             # Workflow node implementations
│   │   ├── prompts.py           # All ChatPromptTemplate definitions
│   │   ├── tools.py             # Tool definitions (search_user_db, scrap_job_posting)
│   │   ├── llm_setup.py         # LLM and Retriever instantiation
│   │   ├── main.py              # CLI test runner for the agent
│   │   ├── README.md            # Architecture notes
│   │   └── __pycache__/         # Compiled bytecode
│   └── workflow.png             # Generated Mermaid diagram
└── tests/
```

---

### 2. How `main.py` Sets Up the Streamlit UI

**File: `/home/jose/Development/Python/job-hunt-prep/src/job_hunt_prep/main.py`**

The Streamlit app has a simple two-phase structure:

**Initialization (module-level, runs once):**
- Line 43: Creates a **single global `Agent` instance** at module load time:
  ```python
  agent = Agent(llm=LLM().llm, retriever=Retriever().retriever)
  ```
  This agent instance is shared across all Streamlit sessions/requests.

**Phase 1 -- Initial state (no messages yet):**
- Lines 88-99: Shows a `chat_input` widget with placeholder "Ask a question..."
- Displays a disclaimer button that opens a dialog

**Phase 2 -- Active conversation:**
- Line 103: A follow-up chat input field
- Lines 125-130: Renders all messages from `st.session_state.messages` as chat bubbles
- Lines 147-149: Sets up the Streamlit callback handler for streaming output:
  ```python
  st_callback = StreamlitCallbackHandler(answer_container)
  cfg = RunnableConfig()
  cfg["callbacks"] = [st_callback]
  cfg["configurable"]: {"thread_id": "1"}  # HARDCODED
  ```
- Lines 166-167: Invokes the agent:
  ```python
  answer = agent.run_agent(user_message, cfg)
  ```
- Lines 172-173: Appends messages to session state for display persistence

**Rate limiting:** Lines 121-161 enforce a minimum 5-second gap between requests.

---

### 3. How the Graph/Agent Works

**Architecture: Two competing approaches**

There are **two different LangGraph architectures** in this codebase:

#### Approach A: `langchain.agents.create_agent` (currently active in `agent.py`)

The `Agent` class in `agent.py` uses `create_agent` from `langchain.agents`:

```python
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware, SummarizationMiddleware

self._app = create_agent(
    model=self._llm,
    tools=tools,
    middleware=[TodoListMiddleware(), SummarizationMiddleware(model=self._llm)],
    checkpointer=InMemorySaver(),
    system_prompt=prompts.agent_prompt,
    state_schema=JobPrepState
)
```

This is the **legacy LangChain agent API**. The agent uses **tool-calling** as the primary interaction pattern -- the LLM decides when to call `search_user_db` or `scrap_job_posting` tools.

The `tools.py` file defines two tools decorated with `@tool`:

**`search_user_db(query, runtime)`** -- Searches the Chroma vector store for relevant user documents. Returns a `Command` object with `ToolMessage`.

**`scrap_job_posting(job_post_url, runtime)`** -- Scrapes a job posting URL using `SeleniumURLLoader`. Returns serialized content.

The agent prompt instructs the LLM to:
1. Always scrape the job post first
2. Then create a semantic search query for the vector store
3. Draft an answer in first person

#### Approach B: StateGraph (referenced in session notes, `nodes.py` has dead code)

The `nodes.py` file contains node implementations (`process_user_input`, `scrap_job_posting`, `distill_search_query`, `search_user_db`, `draft_answer`) that follow the traditional `StateGraph` pattern. However, this code is **not actively used** -- the `create_agent` approach above is what's wired up. The `nodes.py` code appears to be legacy/dead code from an earlier architecture.

#### State Schema (`state.py`)

```python
class JobPrepState(AgentState):
    raw_query: str
    user_query: str | None
    user_instructions: str | None
    job_post_url: str | None
    retrieved_documents: list[Document] | None
    serialized_documents: str | None
    job_post_documents: list[Document] | None
    serialized_job_post: str | None
    distilled_query: str | None
    draft_response: str | None
```

#### LLM Configuration (`llm_setup.py`)

- **Model:** `qwen3.5:9b` via Ollama (local)
- **Context window:** 8192 tokens
- **Embeddings:** `nomic-embed-text-v2-moe`
- **Vector store:** Chroma (persisted locally at `./data/embeddings/`)
- **Reasoning mode:** Disabled

---

### 4. Current State Management Approach

#### Streamlit Session State (`main.py`)

```python
st.session_state.messages              # List of {role, content} dicts for UI display
st.session_state.initial_question      # The very first question (from chat_input key)
st.session_state.prev_question_timestamp  # For rate limiting
```

That's it. **Only 3 session state variables.** All conversation history is stored as plain message dictionaries in `messages`.

#### LangGraph Checkpointer (`agent.py`)

```python
checkpointer = InMemorySaver()  # Module-level singleton

self._app = create_agent(
    checkpointer=InMemorySaver(),  # A SECOND InMemorySaver instance is created
    ...
)
```

The checkpointer is configured but **not actually used meaningfully**:

1. **Thread ID is hardcoded to `"1"`** everywhere:
   - Line 150-152 in `main.py`
   - Lines 67, 79, 95, 107 in `agent.py`
   - Line 41 in `draft.py`

2. **The config is created but the checkpointer's state is never read back** -- each `run_agent()` call passes the same config, but `create_agent` with `InMemorySaver` would maintain conversation history per thread_id. However:
   - `main.py` doesn't use the thread_id-aware features
   - `agent.run_agent()` always sends `{"messages": [{"role": "user", "content": query}]}` -- fresh messages only
   - No code ever reads or resumes from checkpoint state

3. **Two `InMemorySaver` instances exist**:
   - `checkpointer = InMemorySaver()` at module level (line 24)
   - `checkpointer=InMemorySaver()` in `create_agent()` (line 50) -- creates a **different instance**
   
   The module-level `checkpointer` variable is never used at all.

---

### 5. Obvious Issues with Context Persistence

#### Critical Issues

1. **Hardcoded `thread_id = "1"` everywhere**: All users share the same thread. In a multi-user Streamlit deployment, all users would overwrite each other's state. Even for a single user, this means the checkpointer provides no real benefit since it's never read from.

2. **`InMemorySaver` is in-memory only**: Every time the Streamlit server restarts (or the Python process restarts), the checkpointer state is **completely lost**. There's no persistence to disk.

3. **Message state is NOT synced with graph state**: 
   - `main.py` stores messages in `st.session_state.messages` for UI display
   - `agent.py` manages messages internally in the graph's `JobPrepState`
   - These two message stores are **completely independent**
   - The graph's message history is never read back; only `answer['messages'][-1].content` is extracted
   - So the graph's `checkpointer` (even if working) is storing messages that Streamlit never retrieves

4. **Duplicate checkpointer creation**: Two separate `InMemorySaver()` instances are created, making the module-level one useless.

5. **Fresh messages every call**: `agent.run_agent()` always sends `{"messages": [{"role": "user", "content": query}]}`. There's no conversation context passed from `st.session_state.messages` into the agent. The LLM sees only the current query, not prior conversation turns.

6. **Restart button only clears Streamlit state**: Clicking "Restart" (line 112) clears `st.session_state.messages`, but:
   - It does NOT clear the checkpointer state
   - The next call to `agent.run_agent()` will still have the checkpointer state available (if it were being used)
   - But the LLM won't see it anyway because of issue #5

7. **State mismatch between old and current architecture**: `nodes.py` defines `process_user_input` that references `state["messages"][0].content` (old StateGraph pattern), but the current `create_agent` approach uses tool-calling. The `process_user_input` node function in `nodes.py` is **never called** in the current workflow.

8. **No user instructions field in agent tool calls**: The `JobPrepState` has a `user_instructions` field, but the agent tool `scrap_job_posting` and `search_user_db` don't handle it -- they just search. The instructions get lost unless `process_user_input` (dead code) handles them.

#### Recommended Fixes

To make context persistence actually work:

1. Generate a unique `thread_id` per Streamlit session and pass it consistently
2. Use a **persistent checkpointer** (e.g., `PostgresSaver`, `SQLiteSaver`, or file-based) instead of `InMemorySaver`
3. Pass existing conversation history into `run_agent()` so the LLM sees prior messages
4. Consolidate to a single checkpointer instance
5. Remove or reconcile the dead `nodes.py` code with the current `create_agent` approach
</task_result>
```

---

## Assistant (Plan · big-pickle · 14.3s)

_Thinking:_

Now I have a comprehensive understanding of the codebase. The context persistence issues are clear:

1. **Hardcoded thread_id = "1"** - All users share the same thread
2. **InMemorySaver loses state on restart** - No persistence to disk
3. **Messages not synced** - Streamlit session state and graph checkpointer are independent
4. **Fresh messages every call** - No conversation context is passed to the agent
5. **Duplicate checkpointer instances** - Module-level one is unused

Let me propose a solution plan to the user.

Based on my analysis, I found several interconnected issues causing context persistence failures:

## Root Causes

1. **Hardcoded `thread_id = "1"`** - All users share the same checkpointer state
2. **`InMemorySaver` is volatile** - State is lost on server restart
3. **Messages never flow back from graph** - `main.py` only reads `answer['messages'][-1].content`, never syncs with checkpointer
4. **Fresh messages every call** - `run_agent()` sends only the current query, not conversation history
5. **Duplicate checkpointer instances** - Module-level one is unused

## Proposed Solution

| Issue | Fix |
|-------|-----|
| Per-user threads | Generate `thread_id = st.session_state.session_id` on first load |
| Volatile memory | Replace `InMemorySaver` with `FileCheckpointer` (simple, local) or `SQLiteSaver` |
| Message sync | Store full message history in session state, pass to agent on each call |
| Fresh messages | Build message list from session state before each `run_agent()` call |

Would you like me to proceed with this plan, or do you have preferences on:
- **Checkpointer type**: `FileCheckpointer` (simpler) vs `SQLiteSaver` (more robust)?
- **Scope**: Single-user first, or design for multi-user from the start?

---

## User

How can we do message syncing?

---

## Assistant (Plan · big-pickle · 16.1s)

_Thinking:_

The user is asking about how to do message syncing between the Streamlit session state and the LangGraph checkpointer. Let me explain the options based on my understanding of the codebase.

Looking at the current code:
- `main.py` stores messages in `st.session_state.messages` for UI display
- `agent.py` uses `create_agent` with `InMemorySaver` checkpointer
- `agent.run_agent()` always sends fresh messages only: `{"messages": [{"role": "user", "content": query}]`

There are a few approaches to sync messages:

**Approach 1: Pass history to agent on each call**
- Before calling `agent.run_agent()`, build a message list from `st.session_state.messages`
- Pass this history in the config or directly to the agent
- This way the LLM sees the full conversation context

**Approach 2: Read from checkpointer after each call**
- After `agent.run_agent()`, read the updated state from the checkpointer
- Sync the messages back to `st.session_state.messages`
- This ensures UI matches the graph's internal state

**Approach 3: Hybrid**
- Do both - pass history in, read state back out

Let me think about how LangGraph's `create_agent` works vs a standard `StateGraph`:
- `create_agent` from `langchain.agents` has a different API than the standard `StateGraph`
- The `run_agent` method would need to accept or be modified to accept message history

I should explain the options to the user and ask which approach they prefer.

Good question. Here are the options for message syncing:

## Option 1: Pass History INTO Agent (Read from Session State)
```python
# Before calling agent.run_agent()
history = st.session_state.messages
messages = [HumanMessage(content=m["content"]) for m in history]
messages.append(HumanMessage(content=user_message))

# Pass to agent
answer = agent.run_agent(user_message, config, messages=messages)
```

**Pros**: LLM sees full context | Simple to implement  
**Cons**: Relies on session state (loses context on server restart unless using persistent checkpointer)

---

## Option 2: Read FROM Checkpointer After Each Call
```python
# After agent.run_agent()
from langchain_core.messages import messages_to_dict

state = agent._app.get_state({"configurable": {"thread_id": thread_id}})
st.session_state.messages = messages_to_dict(state.values["messages"])
```

**Pros**: Source of truth is the checkpointer | Works across server restarts  
**Cons**: Requires correct checkpointer setup | Adds latency on each call

---

## Option 3: Hybrid (Recommended)
1. Pass history in on each call (so LLM has context)
2. Optionally read/checkpointer state back after to verify sync

---

## Key Question

Which pattern fits your use case?

- **Option 1** is simplest if users won't expect context after server restart
- **Option 2** is better if you want the checkpointer as single source of truth
- **Option 3** gives most robustness

Also: are you okay modifying `agent.run_agent()` signature to accept optional message history, or would you prefer a separate method like `run_agent_with_history()`?

---

## User

Option 1 with signature change

---

## Assistant (Plan · big-pickle · 11.4s)

_Thinking:_

The user wants Option 1 (pass history into agent) with a signature change to `run_agent()`. I'm in plan mode so I should outline the implementation plan without making any changes.

Let me outline what needs to be done:

1. **Modify `agent.run_agent()`** - Add optional `messages` parameter that defaults to None (backward compatible)
2. **Modify `main.py`**:
   - Generate unique session_id on first load
   - Build message list from session_state.messages before each call
   - Pass messages to run_agent()
   - Ensure thread_id uses session_id

Let me think about what files need to change and in what order.

I should present a clear, actionable plan.

Here's the implementation plan for Option 1 with signature change:

## Plan

### 1. Modify `agent.py` - Update `run_agent()` signature

```python
# Current
def run_agent(self, query: str, config: dict) -> dict:

# Proposed (backward compatible)
def run_agent(self, query: str, config: dict, messages: list[BaseMessage] | None = None) -> dict:
    if messages is None:
        messages = [HumanMessage(content=query)]
    else:
        messages = messages + [HumanMessage(content=query)]
    
    return self._app.invoke({"messages": messages}, config)
```

**File**: `src/job_hunt_prep/graph/agent.py:62`

---

### 2. Modify `main.py` - Session ID + Message Building

**Add to top of app (after `st.set_page_config`):**
```python
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
```

**File**: `src/job_hunt_prep/main.py:45` (after agent initialization)

---

**Update the config creation block (~line 147-152):**
```python
# Build message history from session state
messages = [HumanMessage(content=m["content"]) for m in st.session_state.messages]

cfg = RunnableConfig()
cfg["callbacks"] = [st_callback]
cfg["configurable"] = {"thread_id": st.session_state.session_id}

answer = agent.run_agent(user_message, cfg, messages=messages)
```

**File**: `src/job_hunt_prep/main.py:147-167`

---

### 3. Add Import to `main.py`

```python
from langchain_core.messages import HumanMessage
```

**File**: `src/job_hunt_prep/main.py:9`

---

## Files to Modify

| File | Change |
|------|--------|
| `graph/agent.py` | Add `messages` param to `run_agent()`, build message list internally |
| `main.py` | Add session_id init, import HumanMessage, build messages list before call |

---

## Verification

After changes, test:
1. Send first message → LLM responds
2. Send second message → LLM references context from first message
3. Server restart → Context resets (expected with this approach)

---

**Questions before implementation:**
1. Do you want the `messages` param added to `run_agent()` or a new method like `run_agent_with_history()`?
2. Should the `session_id` be shown anywhere in the UI (for debugging)?

---

