"""
Sanity check: run BasicAgent on ONE random question before committing
to the full 20-question evaluation.

Run this from the same folder as app.py, in the same venv:
    python test_one_question.py

No need to set SPACE_ID for this - that only matters for the real
submission's agent_code link, which this script never touches.
"""

import requests
from app import BasicAgent, DEFAULT_API_URL

# 1. Fetch one random question
resp = requests.get(f"{DEFAULT_API_URL}/random-question", timeout=30)
resp.raise_for_status()
question_data = resp.json()

task_id = question_data.get("task_id")
question_text = question_data.get("question")

print("=" * 60)
print(f"Task ID:  {task_id}")
print(f"Question: {question_text}")
print("=" * 60)

# 2. Run the agent on it (this may take anywhere from a few seconds
#    to a couple minutes depending on how many steps it needs)
agent = BasicAgent()
answer = agent(question_text, task_id, DEFAULT_API_URL)

print("=" * 60)
print(f"Agent's answer: {answer!r}")
print("=" * 60)