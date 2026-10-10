# GAIA Unit 4 Agent

My final assignment for the [Hugging Face AI Agents Course](https://huggingface.co/learn/agents-course). The agent answers GAIA-style benchmark questions from the course's scoring API and submits its answers for grading. It is built to run on free resources only.

## How it works

1. The app fetches the course questions and downloads any file attached to each one.
2. A [smolagents](https://github.com/huggingface/smolagents) `CodeAgent` solves each question using the tools below, with a 12-step limit.
3. The answer is cleaned up for exact-match grading and submitted to the scoring API. The score is shown in the Gradio interface.

## Tools

- Web search (DuckDuckGo, with automatic retries when rate-limited)
- Webpage reader
- Wikipedia search
- YouTube transcript lookup
- Spreadsheet and text file readers
- Python code execution, with pandas, numpy and other standard libraries allowed

Attachments are handled before the agent starts: audio is transcribed with Whisper, images are passed to the model when a vision-capable model is configured, Python files are run and their output is included, and spreadsheets are previewed.

## Free-tier design

- **Model fallback:** tries several LLM providers in order (Gemini, Groq, OpenRouter free models, a local Ollama model, Hugging Face Inference). A provider that hits a quota or rate limit is skipped for a cooldown period, so free quotas from different providers add up.
- **Isolation per question:** each question gets a fresh agent and a timeout, so one failure cannot affect the next.
- **Answer cache:** answers are saved to `answers_cache.json`, so only failed questions need re-running. A "Submit cached answers only" button sends saved answers without re-running the agent.

## Setup

```bash
pip install -r requirements.txt
python app.py
```

Set at least one model key as an environment variable (never commit keys). Then log in with Hugging Face in the app and click **Run Evaluation & Submit All Answers**.

| Variable | Purpose |
| --- | --- |
| `GEMINI_API_KEY` | Google AI Studio key (free tier) |
| `GROQ_API_KEY` | Groq key (free tier), optional |
| `OPENROUTER_API_KEY` | OpenRouter key (free models), optional |
| `OLLAMA_MODEL` | Local model name, optional (needs `ollama serve`) |
| `HF_TOKEN` | Hugging Face token, used for Inference and audio transcription |
| `QUESTION_TIMEOUT` | Seconds allowed per question (default 360) |
| `SPACE_ID` | Set when running locally so the submission links to your Space |

## Built with

Python, smolagents, LiteLLM, Gradio, pandas

## Result

Reached the course's pass grade using only free resources.

---
title: Template Final Assignment
emoji: 🕵🏻‍♂️
colorFrom: indigo
colorTo: indigo
sdk: static
sdk_version: 5.25.2
app_file: app.py
pinned: false
hf_oauth: true
# optional, default duration is 8 hours/480 minutes. Max duration is 30 days/43200 minutes.
hf_oauth_expiration_minutes: 480
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
