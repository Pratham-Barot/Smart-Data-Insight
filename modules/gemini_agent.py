import os
import json
from google import genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_MODEL = "models/gemini-2.5-flash"

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ================================
# SYSTEM PROMPT — Data Scientist
# ================================
SYSTEM_PROMPT = """You are an expert senior data scientist AI assistant with 15 years of experience.
You are analyzing a dataset and helping a user understand it deeply.

Your behavior rules:
1. Always be specific — reference actual column names, numbers, and percentages from the data
2. If generating Python code, wrap it in ```python code blocks
3. Use pandas (df variable) and plotly express (px variable) in your code
4. Always end your response with a one-line "Key Insight:" summary
5. Be concise but thorough — no unnecessary filler text
6. If you detect a potential business problem in the data, highlight it
7. Format numbers clearly — use commas for thousands, 2 decimal places for floats

When writing code:
- Always assign plots to variable named 'fig'
- Always assign tabular results to variable named 'result'
- Never use plt (matplotlib) — only use plotly express as px
- Always handle null values before operations
"""


# ================================
# Step 5.3 — Context Builder
# ================================
def build_context(profile: dict, df_sample_json: str) -> str:
    """
    Converts data profile + sample rows into
    a clean prompt-ready context string for Gemini.
    """
    context = f"""
=== DATASET PROFILE ===
Rows: {profile['shape']['rows']}
Columns: {profile['shape']['columns']}
Memory: {profile.get('memory_usage_kb', 'N/A')} KB
Duplicate rows: {profile['duplicate_rows']}

=== COLUMN ROLES ===
{json.dumps(profile['column_roles'], indent=2)}

=== NULL ANALYSIS ===
{json.dumps(profile['null_percent'], indent=2)}

=== NUMERIC STATISTICS ===
{json.dumps(profile.get('numeric_stats', {}), indent=2, default=str)}

=== TOP VALUES (CATEGORICAL) ===
{json.dumps(profile.get('top_values', {}), indent=2, default=str)}

=== SAMPLE ROWS (first 5) ===
{df_sample_json}
"""
    return context


# ================================
# Step 5.4 — Send to Gemini
# ================================
def ask_gemini(
    question: str,
    profile: dict,
    df_sample_json: str,
    chat_history: list = []
) -> str:
    """
    Sends user question + full data context to Gemini.
    Returns Gemini's response as a string.
    """
    context = build_context(profile, df_sample_json)

    full_prompt = f"""
{SYSTEM_PROMPT}

{context}

=== USER QUESTION ===
{question}
"""

    try:
        contents = []
        for msg in chat_history:
            contents.append(msg)
        contents.append({
            "role": "user",
            "parts": [{"text": full_prompt}]
        })

        # Try primary model first, fallback if unavailable
        models_to_try = [
            "models/gemini-2.5-flash",
            "models/gemini-2.0-flash",
            "models/gemini-2.0-flash-lite",
        ]

        last_error = ""
        for model_name in models_to_try:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents
                )
                return response.text
            except Exception as model_error:
                last_error = str(model_error)
                if "503" in last_error or "UNAVAILABLE" in last_error:
                    continue  # try next model
                elif "429" in last_error:
                    return (
                        "Rate limit reached. "
                        "Please wait 30 seconds and try again."
                    )
                else:
                    return f"Gemini error: {last_error}"

        return (
            "All models are currently busy. "
            "Please wait 1-2 minutes and try again. "
            f"Last error: {last_error[:100]}"
        )

    except Exception as e:
        return f"Gemini error: {str(e)}"

# ================================
# Step 5.1 — Auto First Look
# ================================
def get_auto_summary(profile: dict, df_sample_json: str) -> str:
    import time
    auto_question = """
Analyze this dataset and provide a first-look summary covering:
1. What kind of dataset is this? What domain does it appear to be from?
2. What are the 3 most important columns and why?
3. What data quality issues exist? (nulls, duplicates, suspicious values)
4. What are 3 interesting patterns or insights you can already see?
5. What are the top 3 questions a data analyst should investigate?

Be specific with numbers and column names.
Key Insight: (one line summary)
"""
    # Retry up to 3 times with 10 second wait
    for attempt in range(3):
        result = ask_gemini(auto_question, profile, df_sample_json)
        if "503" in result or "UNAVAILABLE" in result or "busy" in result:
            if attempt < 2:
                time.sleep(10)
                continue
        return result
    return "Auto summary unavailable — model busy. Try refreshing the page."

# ================================
# Step 5.5 — Extract Code Blocks
# ================================
def extract_code(response_text: str) -> str | None:
    """
    Extracts Python code block from Gemini's response.
    Returns the code string or None if no code found.
    """
    import re
    pattern = r"```python(.*?)```"
    match = re.search(pattern, response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None