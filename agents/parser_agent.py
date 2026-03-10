import json
import re
import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

PARSER_SYSTEM = """You are a math problem parser for JEE-level problems.
Your job is to clean raw input (possibly from OCR or speech-to-text) and
output a structured JSON representation of the math problem.

Output ONLY valid JSON, no markdown, no explanation. Use this schema:
{
  "problem_text": "cleaned, clear problem statement",
  "topic": "algebra|probability|calculus|linear_algebra|unknown",
  "subtopic": "e.g. quadratic_equations, determinants, limits, ...",
  "variables": ["list", "of", "variables"],
  "constraints": ["e.g. x > 0", "n is a positive integer"],
  "given_values": {"key": "value"},
  "what_to_find": "what the problem is asking for",
  "needs_clarification": false,
  "clarification_reason": ""
}

Set needs_clarification=true if:
- The problem is ambiguous or incomplete
- Critical information is missing
- OCR artifacts make the problem unclear
"""


def parse_problem(raw_input: str) -> dict:
    """
    Convert raw text (from OCR/ASR/user) into structured problem JSON.
    Returns parsed dict with possible needs_clarification flag.
    """
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": PARSER_SYSTEM},
                {"role": "user", "content": f"Parse this math problem:\n\n{raw_input}"}
            ],
            temperature=0.1,
            max_tokens=600
        )
        raw_json = response.choices[0].message.content.strip()

        # Strip markdown code fences if present
        raw_json = re.sub(r"```(?:json)?", "", raw_json).strip("` \n")

        parsed = json.loads(raw_json)
        return parsed

    except json.JSONDecodeError as e:
        return {
            "problem_text": raw_input,
            "topic": "unknown",
            "subtopic": "",
            "variables": [],
            "constraints": [],
            "given_values": {},
            "what_to_find": "unknown",
            "needs_clarification": True,
            "clarification_reason": f"Could not parse problem structure: {str(e)}"
        }
    except Exception as e:
        return {
            "problem_text": raw_input,
            "topic": "unknown",
            "subtopic": "",
            "variables": [],
            "constraints": [],
            "given_values": {},
            "what_to_find": "unknown",
            "needs_clarification": True,
            "clarification_reason": f"Parser error: {str(e)}"
        }