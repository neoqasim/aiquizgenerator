import os
import re
import json
import requests
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

API_KEY = os.getenv("GROQ_API_KEY")
API_URL = "https://api.groq.com/openai/v1/chat/completions"

# ─────────────────────────────────────────────
# SYSTEM PROMPT  — strict JSON output, no markdown
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert quiz designer and educator.

Your task is to generate a high-quality multiple-choice quiz based on the topic provided by the user.

RULES:
1. Generate exactly 5 questions.
2. Each question must have exactly 4 options labeled A, B, C, D.
3. Only one option is correct.
4. Questions should range from straightforward recall to applied/analytical thinking.
5. Keep language clear, precise, and unambiguous.
6. Do NOT repeat questions or options.
7. Make wrong options (distractors) plausible — not obviously silly.

OUTPUT FORMAT:
Respond ONLY with a valid JSON array. No markdown, no backticks, no explanation.

[
  {
    "question": "Full question text here?",
    "options": {
      "A": "First option",
      "B": "Second option",
      "C": "Third option",
      "D": "Fourth option"
    },
    "answer": "B",
    "explanation": "Brief one-sentence explanation of why B is correct."
  }
]

Output ONLY the JSON array, nothing else."""


# ─────────────────────────────────────────────
# Parse the model response → list of dicts
# ─────────────────────────────────────────────
def parse_quiz(raw: str):
    """
    Try JSON parse first.  Fall back to regex extraction if the model
    wraps output in markdown fences despite instructions.
    """
    raw = raw.strip()

    # Strip optional markdown fences
    raw = re.sub(r"^```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
    raw = re.sub(r"```$", "", raw).strip()

    try:
        questions = json.loads(raw)
        if isinstance(questions, list) and len(questions) > 0:
            return questions, None
    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {e}"

    return None, "Unexpected response format."


# ─────────────────────────────────────────────
# Call Groq API
# ─────────────────────────────────────────────
def generate_quiz(topic: str, num_questions: int = 5, difficulty: str = "mixed"):
    if not API_KEY:
        return None, "GROQ_API_KEY is not set. Please add it to your .env file."

    difficulty_instruction = {
        "easy":   "Focus on basic recall and definitions. Keep questions simple.",
        "medium": "Include a mix of recall and comprehension questions.",
        "hard":   "Prioritise application, analysis, and evaluation questions.",
        "mixed":  "Include a balanced range from easy recall to hard analytical questions.",
    }.get(difficulty, "")

    user_message = (
        f"Topic: {topic}\n"
        f"Number of questions: {num_questions}\n"
        f"Difficulty: {difficulty_instruction}"
    )

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        "temperature": 0.7,
        "max_tokens": 2048,
    }

    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        return None, "Request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        return None, f"Network error: {e}"

    data = resp.json()

    if "error" in data:
        return None, f"API Error: {data['error'].get('message', 'Unknown error')}"

    if "choices" not in data or not data["choices"]:
        return None, f"Unexpected API response: {data}"

    raw_content = data["choices"][0]["message"]["content"]
    return parse_quiz(raw_content)


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    """AJAX endpoint — returns JSON."""
    body = request.get_json(silent=True) or {}
    topic      = (body.get("topic") or "").strip()
    difficulty = body.get("difficulty", "mixed")
    num_q      = int(body.get("num_questions", 5))

    if not topic:
        return jsonify({"error": "Please enter a topic."}), 400

    num_q = max(3, min(num_q, 10))   # clamp between 3–10

    questions, error = generate_quiz(topic, num_questions=num_q, difficulty=difficulty)
    if error:
        return jsonify({"error": error}), 500

    return jsonify({"questions": questions, "topic": topic})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)