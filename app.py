"""
Unit 4 Final Assignment - GAIA-style agent (free-tier friendly).

The agent tries several LLM backends in order and skips any that hit a quota/rate limit,
so free quotas from different providers add up. Set whichever of these you have:

  GEMINI_API_KEY      Google AI Studio key (free). Each Gemini model has its own free quota.
  GEMINI_MODELS       comma-separated LiteLLM ids, tried in order
                      (default: gemini/gemini-3.1-flash-lite,gemini/gemini-3.5-flash-lite,gemini/gemini-3.5-flash)
  GEMINI_REASONING    reasoning effort for Gemini models (default "low"; set to "none" to not send it)
  GROQ_API_KEY        optional, free. GROQ_MODELS default: groq/llama-3.3-70b-versatile
  OPENROUTER_API_KEY  optional, free models only. OPENROUTER_MODELS default:
                      openrouter/meta-llama/llama-3.3-70b-instruct:free
  OLLAMA_MODEL        optional local model, e.g. qwen2.5-coder:14b (needs `ollama serve`)
  HF_TOKEN            HF Inference Providers (small free credit). MODEL_ID / HF_PROVIDER optional.
  QUESTION_TIMEOUT    seconds per question (default 360)
  SPACE_ID            set when running locally so the submission links to your Space.

Wrong or retired model ids are fine: they just fail once and get skipped.
"""

import os
import re
import sys
import json
import time
import threading
import subprocess

import gradio as gr
import pandas as pd
import requests

from smolagents.models import Model
from smolagents import (
    CodeAgent,
    InferenceClientModel,
    DuckDuckGoSearchTool,
    VisitWebpageTool,
    tool,
)

# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"
DOWNLOAD_DIR = "downloaded_files"
CACHE_PATH = "answers_cache.json"
QUESTION_TIMEOUT = int(os.getenv("QUESTION_TIMEOUT", "360"))  # seconds per question
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

WIKI_HEADERS = {"User-Agent": "GAIA-course-agent/1.0 (educational project)"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
SHEET_EXT = {".xlsx", ".xls", ".csv"}


# --- Helpers (not exposed to the agent) ---

def _df_to_text(df: pd.DataFrame, max_rows: int = 300) -> str:
    note = ""
    if len(df) > max_rows:
        note = f"\n[showing first {max_rows} of {len(df)} rows]"
        df = df.head(max_rows)
    try:
        return df.to_markdown(index=False) + note
    except Exception:  # tabulate missing
        return df.to_string(index=False) + note


def _load_sheets(path: str) -> dict:
    if path.lower().endswith(".csv"):
        return {"csv": pd.read_csv(path)}
    return pd.read_excel(path, sheet_name=None)


def _run_python_file(path: str, timeout: int = 30) -> str:
    try:
        p = subprocess.run(
            [sys.executable, path], capture_output=True, text=True, timeout=timeout
        )
        return (p.stdout + ("\n[stderr]\n" + p.stderr if p.stderr else "")).strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"(timed out after {timeout}s)"
    except Exception as e:
        return f"(could not run: {e})"


def _transcribe_audio(path: str) -> str:
    """Try HF Inference first, then a local whisper pipeline if transformers is installed."""
    try:
        from huggingface_hub import InferenceClient

        client = InferenceClient(token=os.getenv("HF_TOKEN"), timeout=120)
        return client.automatic_speech_recognition(path, model="openai/whisper-large-v3").text
    except Exception as e1:
        try:
            from transformers import pipeline

            asr = pipeline("automatic-speech-recognition", model="openai/whisper-base", chunk_length_s=30)
            return asr(path)["text"]
        except Exception as e2:
            return f"[transcription failed: {e1} | {e2}]"


def _run_with_timeout(fn, timeout: int):
    """Run fn() in a daemon thread; raise TimeoutError if it doesn't finish in time."""
    box = {}

    def target():
        try:
            box["result"] = fn()
        except BaseException as e:
            box["error"] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError(f"no result after {timeout}s")
    if "error" in box:
        raise box["error"]
    return box["result"]


def clean_answer(text: str) -> str:
    a = str(text).strip()
    a = re.sub(r"^(final answer|answer)\s*[:\-]\s*", "", a, flags=re.I).strip()
    if len(a) >= 2 and a[0] == a[-1] and a[0] in "\"'":
        a = a[1:-1].strip()
    if re.fullmatch(r"-?\d+\.0+", a):
        a = a.split(".")[0]
    if a.endswith(".") and not re.search(r"\b[A-Z]\.$", a):
        a = a[:-1]
    return a


# --- Tools available to the agent ---

class RobustSearchTool(DuckDuckGoSearchTool):
    """DuckDuckGo search that retries when rate-limited instead of crashing the step."""

    def forward(self, query: str) -> str:
        last = None
        for attempt in range(4):
            try:
                return super().forward(query)
            except Exception as e:
                last = e
                time.sleep(3 * (attempt + 1))
        return (
            f"Search failed after retries ({last}). Try different wording, "
            "or use wikipedia_search / visit_webpage directly."
        )


@tool
def wikipedia_search(query: str) -> str:
    """Searches English Wikipedia and returns matching page titles, URLs and snippets.
    Follow up with visit_webpage on a URL to read the full page, including tables.

    Args:
        query: Search terms, e.g. an artist name, a topic, or an article title.
    """
    try:
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "list": "search", "srsearch": query, "srlimit": 8, "format": "json"},
            headers=WIKI_HEADERS,
            timeout=20,
        )
        r.raise_for_status()
        hits = r.json()["query"]["search"]
        if not hits:
            return "No Wikipedia results."
        out = []
        for h in hits:
            title = h["title"]
            snippet = re.sub(r"<[^>]+>", "", h["snippet"])
            url = "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")
            out.append(f"- {title} | {url} | {snippet}")
        return "\n".join(out)
    except Exception as e:
        return f"Wikipedia search failed: {e}"


@tool
def youtube_transcript(url: str) -> str:
    """Returns the spoken transcript (captions) of a YouTube video. It cannot see the
    video itself, so it only helps for questions about what is said.

    Args:
        url: A YouTube URL or the 11-character video id.
    """
    m = re.search(r"(?:v=|youtu\.be/|embed/)([\w-]{11})", url)
    vid = m.group(1) if m else url.strip()
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        if hasattr(YouTubeTranscriptApi, "get_transcript"):  # old API (<1.0)
            data = YouTubeTranscriptApi.get_transcript(vid)
            return " ".join(d["text"] for d in data)
        data = YouTubeTranscriptApi().fetch(vid)  # new API
        return " ".join(s.text for s in data)
    except Exception as e:
        return f"Could not get transcript: {e}"


@tool
def read_spreadsheet(file_path: str) -> str:
    """Reads a local CSV or Excel file and returns every sheet as a markdown table.
    For sums, counts or filtering, prefer loading the file with pandas in code.

    Args:
        file_path: Local path to a .csv, .xlsx, or .xls file.
    """
    try:
        sheets = _load_sheets(file_path)
        return "\n\n".join(f"### Sheet: {name}\n{_df_to_text(df)}" for name, df in sheets.items())
    except Exception as e:
        return f"Could not read spreadsheet: {e}"


@tool
def read_text_file(file_path: str) -> str:
    """Reads a local plain-text or code file and returns its contents.

    Args:
        file_path: Local path to a text-based file (.txt, .py, .json, .md, etc.).
    """
    try:
        with open(file_path, "r", errors="replace") as f:
            return f.read()
    except Exception as e:
        return f"Could not read file: {e}"


# --- The agent ---

INSTRUCTIONS = """You answer GAIA benchmark questions. Answers are graded by exact match.

Final answer format: call final_answer ONCE with only the answer: a number, a few words, or a
comma-separated list. No sentence, no "The answer is", no explanation. No units or currency/percent
symbols unless the question asks for them. No thousands separators. No articles (a/an/the) and no
abbreviations in strings unless the question asks for them. Follow any format the question specifies
(e.g. "alphabetical order", "comma separated", "IOC code", "two decimal places").

How to work:
- Never answer factual questions from memory. Search, then open the actual pages with visit_webpage
  and read them (tables included). Search snippets are not enough.
- For anything on Wikipedia use wikipedia_search, then visit_webpage on the article URL.
  Page histories, talk pages and archived pages can also be opened with visit_webpage.
- If a search returns nothing useful, reword the query and try a different source. Try at least
  three distinct approaches before giving up.
- Respect every constraint in the question exactly (date ranges and whether they are inclusive,
  which type of item to count, which year, which unit). Re-read the question before answering.
- Puzzles (reversed text, logic tables, counting, arithmetic, sorting) must be solved with python
  code, not by eye. If the question text looks reversed or scrambled, decode it first.
- Information about an attached file is included in the task text; use it.
- If you cannot fully verify the answer, still give your best-supported guess via final_answer.
  Never return an empty answer."""


class FallbackModel(Model):
    """Tries several models in order. A model that hits a quota / rate / config error is put
    on cooldown so later steps skip it straight away. Empty replies are retried, then skipped."""

    DAILY_HINTS = ("perday", "per day", "per-day", "daily", "402", "credits", "monthly")
    CONFIG_HINTS = ("404", "not found", "api key", "401", "403", "permission", "unauthorized", "is not supported")

    def __init__(self, models: list):
        super().__init__(model_id="fallback[" + " > ".join(str(m.model_id) for m in models) + "]")
        self.models = models
        self.cooldown_until = {}

    def _cool(self, i: int, err: Exception) -> int:
        msg = str(err).lower()
        if any(h in msg for h in self.DAILY_HINTS):
            secs = 6 * 3600  # daily quota / credits: don't bother again this session
        elif any(h in msg for h in self.CONFIG_HINTS):
            secs = 24 * 3600  # bad model id / key
        else:
            secs = 65  # per-minute limit or transient error
        self.cooldown_until[i] = time.time() + secs
        return secs

    @staticmethod
    def _is_empty(msg) -> bool:
        content = getattr(msg, "content", None)
        has_tools = bool(getattr(msg, "tool_calls", None))
        return not has_tools and (content is None or not str(content).strip())

    def generate(self, messages, **kwargs):
        last = None
        for _ in range(4):
            tried = False
            for i, m in enumerate(self.models):
                if self.cooldown_until.get(i, 0) > time.time():
                    continue
                tried = True
                # empty replies are usually transient: retry the same model before moving on
                for empty_try in range(3):
                    try:
                        msg = m.generate(messages, **kwargs)
                    except Exception as e:
                        last = e
                        secs = self._cool(i, e)
                        print(f"[fallback] {m.model_id} failed, skipping for {secs}s: {str(e)[:200]}")
                        break
                    if not self._is_empty(msg):
                        return msg
                    last = RuntimeError("empty response")
                    raw = getattr(msg, "raw", None)
                    print(f"[fallback] {m.model_id} returned empty content (try {empty_try + 1}/3) raw={str(raw)[:300]}")
                    time.sleep(2)
                else:
                    self.cooldown_until[i] = time.time() + 30  # short rest, try next model
            if not tried:
                wait = min(self.cooldown_until.values()) - time.time()
                if wait > 90:
                    break
                time.sleep(max(wait, 1))
        raise RuntimeError(f"all models unavailable: {last}")


def _lite(model_id: str, **kw):
    from smolagents import LiteLLMModel

    try:
        import litellm

        litellm.drop_params = True  # silently drop params a provider doesn't support
    except Exception:
        pass
    return LiteLLMModel(model_id=model_id.strip(), retry=False, timeout=90, **kw)


def build_model():
    """Returns (model, has_vision). Order: Gemini > Groq > OpenRouter > Ollama > HF."""
    models = []
    vision = False

    if os.getenv("GEMINI_API_KEY"):
        ids = os.getenv(
            "GEMINI_MODELS",
            "gemini/gemini-3.1-flash-lite,gemini/gemini-3.5-flash-lite,gemini/gemini-3.5-flash",
        )
        effort = os.getenv("GEMINI_REASONING", "low").strip().lower()
        extra = {} if effort in ("", "none", "off") else {"reasoning_effort": effort}
        for mid in ids.split(","):
            models.append(_lite(mid, api_key=os.getenv("GEMINI_API_KEY"), requests_per_minute=8, **extra))
        vision = True

    if os.getenv("GROQ_API_KEY"):
        for mid in os.getenv("GROQ_MODELS", "groq/llama-3.3-70b-versatile").split(","):
            models.append(_lite(mid, api_key=os.getenv("GROQ_API_KEY"), requests_per_minute=20))

    if os.getenv("OPENROUTER_API_KEY"):
        ids = os.getenv("OPENROUTER_MODELS", "openrouter/meta-llama/llama-3.3-70b-instruct:free")
        for mid in ids.split(","):
            models.append(_lite(mid, api_key=os.getenv("OPENROUTER_API_KEY"), requests_per_minute=15))

    if os.getenv("OLLAMA_MODEL"):
        models.append(
            _lite(
                "ollama_chat/" + os.getenv("OLLAMA_MODEL"),
                api_base=os.getenv("OLLAMA_BASE", "http://localhost:11434"),
                api_key="ollama",
                num_ctx=16384,  # Ollama's default context is too small for agent prompts
            )
        )

    if os.getenv("HF_TOKEN"):
        models.append(
            InferenceClientModel(
                model_id=os.getenv("MODEL_ID", "Qwen/Qwen2.5-Coder-32B-Instruct"),
                provider=os.getenv("HF_PROVIDER") or None,
                retry=False,
            )
        )

    if not models:
        raise RuntimeError("No LLM backend configured: set GEMINI_API_KEY, GROQ_API_KEY, OLLAMA_MODEL, ... ")
    # Always wrap, even a single model, so empty replies get retried.
    return FallbackModel(models), vision


class BasicAgent:
    def __init__(self):
        self.model, self.vision = build_model()

    # Fresh agent per question: a timed-out run can't leak state into the next question.
    def _build_agent(self) -> CodeAgent:
        return CodeAgent(
            model=self.model,
            tools=[
                RobustSearchTool(),
                VisitWebpageTool(),
                wikipedia_search,
                youtube_transcript,
                read_spreadsheet,
                read_text_file,
            ],
            additional_authorized_imports=[
                "pandas", "numpy", "json", "re", "math", "statistics", "datetime",
                "itertools", "collections", "csv", "string", "unicodedata", "fractions", "decimal",
            ],
            max_steps=12,
            instructions=INSTRUCTIONS,
        )

    # Download the attachment up front and describe it in the prompt, so the agent
    # doesn't have to figure out when/how to fetch it.
    def _prepare_file(self, task_id: str, file_name: str, api_url: str):
        if not file_name:
            return "", None
        try:
            r = requests.get(f"{api_url}/files/{task_id}", timeout=60)
            r.raise_for_status()
        except Exception as e:
            return f"\nAn attachment ({file_name}) exists but could not be downloaded: {e}\n", None

        path = os.path.join(DOWNLOAD_DIR, os.path.basename(file_name))
        with open(path, "wb") as f:
            f.write(r.content)
        ext = os.path.splitext(path)[1].lower()

        if ext in AUDIO_EXT:
            return f"\nAttached audio file ({path}). Transcript:\n\"\"\"\n{_transcribe_audio(path)}\n\"\"\"\n", None

        if ext in IMAGE_EXT:
            if self.vision:
                from PIL import Image

                return f"\nAn image is attached (also shown to you) at {path}.\n", [Image.open(path).convert("RGB")]
            return (
                f"\nAn image is attached at {path}, but you cannot view images. "
                "If the answer depends on it, give your best guess.\n",
                None,
            )

        if ext == ".py":
            code = open(path, errors="replace").read()
            return (
                f"\nAttached Python file ({path}):\n```python\n{code}\n```\n"
                f"Output when this file was actually executed:\n{_run_python_file(path)}\n",
                None,
            )

        if ext in SHEET_EXT:
            try:
                preview = "\n\n".join(
                    f"### Sheet: {n}\n{_df_to_text(df.head(15))}" for n, df in _load_sheets(path).items()
                )
            except Exception as e:
                preview = f"(preview failed: {e})"
            return (
                f"\nAttached spreadsheet at {path}. Load it with pandas in your code "
                f"(pd.read_excel / pd.read_csv) and compute the answer with code. First rows:\n{preview}\n",
                None,
            )

        return f"\nAttached file at {path}. Use read_text_file to inspect it.\n", None

    def __call__(self, question: str, task_id: str, api_url: str, file_name: str = "") -> str:
        file_ctx, images = self._prepare_file(task_id, file_name, api_url)
        prompt = f"Question: {question}\n{file_ctx}"

        for attempt in range(2):
            agent = self._build_agent()
            try:
                result = _run_with_timeout(lambda: agent.run(prompt, images=images), QUESTION_TIMEOUT)
                return clean_answer(result)
            except TimeoutError as e:
                agent.interrupt()  # stop after the current step
                print(f"Task {task_id} timed out: {e}")
                return f"AGENT ERROR: timed out ({e})"
            except Exception as e:
                print(f"Agent error on task {task_id} (attempt {attempt + 1}): {e}")
                if attempt == 0:
                    time.sleep(20)  # usually a rate limit; wait and retry once
                else:
                    return f"AGENT ERROR: {e}"


# --- Cache so you can re-run only what failed ---

def load_cache() -> dict:
    try:
        with open(CACHE_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def save_cache(cache: dict) -> None:
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)


def _is_good(ans) -> bool:
    return bool(ans) and not str(ans).startswith("AGENT ERROR")


# --- Run + submit flow ---

def run_and_submit_all(reuse_cache: bool, profile: gr.OAuthProfile | None):
    space_id = os.getenv("SPACE_ID")

    if profile:
        username = profile.username
        print(f"User logged in: {username}")
    else:
        return "Please log in to Hugging Face using the button above first.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main" if space_id else "local-run"

    try:
        agent = BasicAgent()
    except Exception as e:
        return f"Error initializing agent: {e}", None

    try:
        response = requests.get(questions_url, timeout=30)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
            return "Fetched questions list is empty.", None
    except Exception as e:
        return f"Error fetching questions: {e}", None

    # Always load the file so we never wipe earlier answers; only *read* from it if asked.
    cache = load_cache()
    results_log = []
    answers_payload = []

    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or question_text is None:
            continue

        cached = cache.get(task_id)
        if reuse_cache and _is_good(cached):
            submitted_answer = cached
            print(f"[{task_id}] using cached answer: {submitted_answer}")
        else:
            try:
                submitted_answer = agent(question_text, task_id, api_url, item.get("file_name") or "")
            except Exception as e:
                submitted_answer = f"AGENT ERROR: {e}"
            print(f"[{task_id}] Q: {question_text[:100]}...\n[{task_id}] A: {submitted_answer}")
            # don't overwrite a previously good answer with an error
            if _is_good(submitted_answer) or not _is_good(cached):
                cache[task_id] = submitted_answer
                save_cache(cache)
            elif _is_good(cached):
                submitted_answer = cached

        answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
        results_log.append({
            "Task ID": task_id,
            "Question": question_text,
            "Submitted Answer": submitted_answer,
        })

    if not answers_payload:
        return "Agent produced no answers to submit.", pd.DataFrame(results_log)

    submission_data = {
        "username": username.strip(),
        "agent_code": agent_code,
        "answers": answers_payload,
    }

    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        return final_status, pd.DataFrame(results_log)
    except Exception as e:
        return f"Submission Failed: {e}", pd.DataFrame(results_log)


def submit_cached_only(profile: gr.OAuthProfile | None):
    """Submit whatever is in answers_cache.json, without running the agent."""
    if not profile:
        return "Please log in to Hugging Face using the button above first.", None

    cache = load_cache()
    answers = [
        {"task_id": tid, "submitted_answer": ans}
        for tid, ans in cache.items()
        if _is_good(ans)
    ]
    if not answers:
        return "Cache is empty (or only contains errors). Nothing to submit.", None

    space_id = os.getenv("SPACE_ID")
    payload = {
        "username": profile.username.strip(),
        "agent_code": f"https://huggingface.co/spaces/{space_id}/tree/main" if space_id else "local-run",
        "answers": answers,
    }
    table = pd.DataFrame(answers)
    try:
        r = requests.post(f"{DEFAULT_API_URL}/submit", json=payload, timeout=60)
        r.raise_for_status()
        d = r.json()
        return (
            f"Submitted {len(answers)} cached answers.\n"
            f"Score: {d.get('score', 'N/A')}% "
            f"({d.get('correct_count', '?')}/{d.get('total_attempted', '?')} correct)\n"
            f"Message: {d.get('message', '')}"
        ), table
    except Exception as e:
        return f"Submission failed: {e}", table


# --- Gradio UI ---

with gr.Blocks() as demo:
    gr.Markdown("# GAIA Unit 4 Agent")
    gr.Markdown(
        """
        1. Log in with your Hugging Face account below.
        2. Click "Run Evaluation & Submit All Answers".

        Tick the cache box to keep answers from earlier runs that didn't error and only
        re-run the failed ones. Leave it off after you change the agent code or model.
        "Submit cached answers only" sends whatever is already in `answers_cache.json`
        without running the agent (handy after a crash).
        """
    )

    gr.LoginButton()
    reuse_cache = gr.Checkbox(label="Reuse cached answers from previous runs", value=False)
    run_button = gr.Button("Run Evaluation & Submit All Answers")
    submit_cached_button = gr.Button("Submit cached answers only (no agent run)")

    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(fn=run_and_submit_all, inputs=[reuse_cache], outputs=[status_output, results_table])
    submit_cached_button.click(fn=submit_cached_only, inputs=None, outputs=[status_output, results_table])


if __name__ == "__main__":
    demo.launch(debug=True, share=False)