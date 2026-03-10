import os
import re
import json
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

VERIFIER_SYSTEM = """You are a strict math solution verifier for JEE-level problems.
Your job is to critically review a proposed solution for correctness.

Check for:
1. Mathematical correctness (right formulas, no arithmetic errors)
2. Logical flow (each step follows from previous)
3. Domain/constraint violations (e.g., sqrt of negative, log of 0)
4. Edge cases missed
5. Units and dimensions (if applicable)
6. Whether the answer actually addresses the question

Respond in this exact JSON format (no markdown):
{
  "is_correct": true,
  "confidence": 0.85,
  "issues_found": ["list of specific issues if any"],
  "corrections": ["specific corrections needed"],
  "verification_steps": ["what you checked"],
  "needs_human_review": false,
  "review_reason": ""
}

Set needs_human_review=true if confidence < 0.7 or you find critical errors.
"""


def verify_solution(parsed_problem: dict, solution: dict) -> dict:
    """
    Critically verify the proposed solution.
    Returns verification report with confidence and any issues.
    """
    try:
        user_message = f"""ORIGINAL PROBLEM:
{parsed_problem.get('problem_text', '')}

Topic: {parsed_problem.get('topic', '')}
What to find: {parsed_problem.get('what_to_find', '')}
Constraints: {parsed_problem.get('constraints', [])}

PROPOSED SOLUTION:
{solution.get('solution_text', '')}

CLAIMED FINAL ANSWER: {solution.get('final_answer', '')}

Verify this solution thoroughly."""

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": VERIFIER_SYSTEM},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1,
            max_tokens=600
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"```(?:json)?", "", raw).strip("` \n")
        verification = json.loads(raw)

        # Cross-check with solver confidence
        solver_conf = solution.get("confidence", 0.75)
        verifier_conf = verification.get("confidence", 0.75)
        combined_conf = (solver_conf + verifier_conf) / 2

        # Trigger HITL if combined confidence is low
        if combined_conf < 0.65:
            verification["needs_human_review"] = True
            verification["review_reason"] = (
                verification.get("review_reason") or
                f"Low combined confidence: {combined_conf:.2f}"
            )

        verification["combined_confidence"] = round(combined_conf, 3)
        return verification

    except Exception as e:
        return {
            "is_correct": False,
            "confidence": 0.5,
            "issues_found": [f"Verifier error: {str(e)}"],
            "corrections": [],
            "verification_steps": [],
            "needs_human_review": True,
            "review_reason": "Verifier encountered an error",
            "combined_confidence": 0.5
        }