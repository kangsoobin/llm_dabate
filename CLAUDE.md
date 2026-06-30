# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

All commands must run inside the `debate` conda environment:
```bash
conda activate debate
```

## Running the System

**Terminal UI (streaming debate in terminal):**
```bash
python debate.py
```

**Streamlit web UI (ChatGPT-style chat interface):**
```bash
streamlit run app.py
# For remote server access:
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

**Server feasibility check:**
```bash
python check_server.py
```

**Environment setup (first time):**
```bash
bash install_env.sh
```

## Architecture Overview

The system loads **Qwen2.5-14B-Instruct** twice (4-bit NF4 quantization), pinning one instance per GPU. Two personas debate policy topics with the user acting as moderator.

### Data flow

1. User (moderator) enters a question via terminal or Streamlit chat input
2. `DebateSession._build_message()` constructs a user message that injects the opponent's last response into the next agent's context
3. LEFT agent responds first (RIGHT's previous reply injected as context)
4. RIGHT agent responds second (LEFT's just-completed reply injected as context)
5. Responses are streamed token-by-token via `TextIteratorStreamer` + background `Thread`

### Key design decisions

**Single GPU pinning:** `device_map={"": gpu_id}` forces each model onto exactly one GPU. Do not use `"auto"` — it spreads weights across both GPUs.

**transformers 5.x compatibility:** `apply_chat_template` returns `BatchEncoding` in transformers 5.x (vs. a raw `LongTensor` in 4.x). `base_agent._prepare_gen_kwargs()` detects this with `hasattr(encoded, "input_ids")` and handles both cases.

**Streamlit streaming:** `generate_iter()` is a generator that yields tokens. It appends the completed response to `self.history` only after the generator is exhausted. `app.py` calls `session._build_message()` between LEFT and RIGHT to inject the freshly completed LEFT response into RIGHT's context — this works because `st.write_stream()` exhausts the generator before returning.

**Streamlit message coloring:** Streamlit's `chat_message` container doesn't expose style hooks. The workaround injects a hidden `<span class="msg-left|msg-right|msg-moderator">` inside each container, then uses CSS `:has()` selectors to style the parent container. This is in the `CSS` constant in `app.py`.

**Shared tokenization logic:** Both `generate()` (terminal, callback-based) and `generate_iter()` (Streamlit, generator-based) call `_prepare_gen_kwargs()` to avoid duplication.

### Module responsibilities

| Module | Role |
|---|---|
| `agents/base_agent.py` | Model loading, tokenization, generation (streaming + non-streaming) |
| `agents/left_agent.py` / `right_agent.py` | Factory functions — set GPU, name, side |
| `core/session.py` | Round orchestration, opponent-injection message builder, JSON log saving |
| `core/display.py` | Terminal color constants and print helpers |
| `app.py` | Streamlit UI — `@st.cache_resource` model loading, chat rendering, state management |
| `debate.py` | Terminal entry point — interactive loop with `l`/`r`/`q`/`vram`/`reset` commands |
| `config/model.yaml` | Model ID, GPU assignment, quantization mode, generation params |
| `config/prompts.yaml` | Korean-language system prompts for both personas |

### Personas

- **LEFT — 이진영 교수** (gpu 0): 42세, 서울대 사회학과, Berkeley PhD, 진보 시민단체 공동대표. Core beliefs: 불평등 해소, 복지국가, 재벌 개혁.
- **RIGHT — 박민준 교수** (gpu 1): 55세, 연세대 경제학과, Chicago PhD, 시장경제연구원장. Core beliefs: 자유시장, 재정건전성, 작은 정부.

Both agents are configured to **never give neutral responses** and to maintain their position throughout the debate.

## Configuration

Edit `config/model.yaml` to change model, GPU assignment, quantization (`4bit`/`8bit`/`fp16`), or generation parameters. Edit `config/prompts.yaml` to tune personas without touching code.

## Known Issues / Gotchas

- **Shell version specifiers:** Never write unquoted `>=X.Y.Z` in bash scripts — bash interprets `>` as stdout redirection and creates artifact files named `=X.Y.Z`. Always quote: `"transformers>=4.45.0"`.
- **Debate logs** are saved to `logs/debate_YYYYMMDD_HHMMSS.json` when `q` is entered in terminal mode.
- First run downloads ~29 GB of model weights from HuggingFace.
