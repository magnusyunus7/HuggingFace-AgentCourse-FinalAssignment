"""
Unit 4 Final Assignment — GAIA-style agent.

Flow:
  1. Log in with your Hugging Face account (identifies you to the scoring API).
  2. Click "Run Evaluation & Submit All Answers".
  3. The app fetches the 20 questions, runs BasicAgent on each one,
     and POSTs your answers + your Space's code link for scoring.

Requires a Space secret named HF_TOKEN (Settings -> Variables and secrets)
with a Hugging Face token that has Inference API access. This is separate
from the OAuth login below: OAuth tells the scoring API who you are,
HF_TOKEN is what your agent uses to actually call the model.
"""

import os
import json
import mimetypes

import gradio as gr
import pandas as pd
import requests

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
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


# --- Custom tools available to the agent ---

@tool
def download_task_file(task_id: str, api_url: str) -> str:
    """Downloads the file attached to a task (if any) and returns the local path.

    Call this first for any question that seems to reference an attachment,
    a spreadsheet, an image, an audio clip, or a code file. If there is no
    file for this task, it returns a short message saying so instead.

    Args:
        task_id: The task_id of the current question.
        api_url: Base URL of the scoring API, e.g. "https://agents-course-unit4-scoring.hf.space".
    """
    url = f"{api_url}/files/{task_id}"
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        return f"No file available for task {task_id} (status {resp.status_code})."

    content_disp = resp.headers.get("content-disposition", "")
    if "filename=" in content_disp:
        filename = content_disp.split("filename=")[-1].strip('"; ')
    else:
        ext = mimetypes.guess_extension(resp.headers.get("content-type", "")) or ""
        filename = f"{task_id}{ext}"

    path = os.path.join(DOWNLOAD_DIR, filename)
    with open(path, "wb") as f:
        f.write(resp.content)
    return f"Downloaded to local path: {path}"


@tool
def read_spreadsheet(file_path: str) -> str:
    """Reads a local CSV or Excel file and returns its contents as a markdown table.

    Args:
        file_path: Local path to a .csv, .xlsx, or .xls file (e.g. from download_task_file).
    """
    try:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        return df.to_markdown(index=False)
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

class BasicAgent:
    def __init__(self):
        # Uses HF_TOKEN from the Space's environment automatically.
        self.model = InferenceClientModel()
        self.agent = CodeAgent(
            model=self.model,
            tools=[
                DuckDuckGoSearchTool(),
                VisitWebpageTool(),
                download_task_file,
                read_spreadsheet,
                read_text_file,
            ],
            additional_authorized_imports=[
                "pandas", "numpy", "json", "re", "math",
                "statistics", "datetime", "itertools", "collections",
            ],
            max_steps=8,
        )
        self.instructions = (
            "You are a careful research assistant answering short-answer questions. "
            "Work through the problem step by step using your tools, then call "
            "final_answer with ONLY the answer itself: a number, a short string, or a "
            "comma-separated list. Do not add extra words, a leading phrase like "
            "'The answer is', articles such as 'a' or 'the', or abbreviations. Do not "
            "add units unless the question explicitly asks for them. Write numbers "
            "plainly, without thousands separators."
        )

    def __call__(self, question: str, task_id: str, api_url: str) -> str:
        prompt = (
            f"{self.instructions}\n\n"
            f"task_id: {task_id}\n"
            f"api_url (pass this to download_task_file if you need the attached file): {api_url}\n\n"
            f"Question: {question}"
        )
        try:
            result = self.agent.run(prompt)
            return str(result).strip()
        except Exception as e:
            print(f"Agent error on task {task_id}: {e}")
            return f"AGENT ERROR: {e}"


# --- Run + submit flow ---

def run_and_submit_all(profile: gr.OAuthProfile | None):
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

    results_log = []
    answers_payload = []
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or question_text is None:
            continue
        submitted_answer = agent(question_text, task_id, api_url)
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


# --- Gradio UI ---

with gr.Blocks() as demo:
    gr.Markdown("# GAIA Unit 4 Agent")
    gr.Markdown(
        """
        **Steps:**
        1. Log in with your Hugging Face account below.
        2. Click "Run Evaluation & Submit All Answers".

        Running all 20 questions can take a few minutes since each one is a
        multi-step agent run.
        """
    )

    gr.LoginButton()
    run_button = gr.Button("Run Evaluation & Submit All Answers")

    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(fn=run_and_submit_all, outputs=[status_output, results_table])


if __name__ == "__main__":
    demo.launch(debug=True, share=False)
