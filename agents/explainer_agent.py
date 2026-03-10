import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

EXPLAINER_SYSTEM = """You are a friendly, patient JEE math tutor explaining solutions to students.
Your explanations should be:
- Clear and easy to understand (like talking to a 17-year-old student)
- Step-by-step with WHY each step is taken (not just HOW)
- Include the key formula/concept used at each step
- Point out common mistakes to avoid
- Use simple language, avoid jargon unless necessary (and explain it when used)
- Encouraging and motivating

Structure your explanation as:
## 🎯 What This Problem is About
[brief context - 1-2 sentences]

## 📚 Key Concepts Needed
[bullet list of formulas/concepts]

## 🔍 Step-by-Step Solution
[numbered steps, each with WHY + HOW]

## ✅ Final Answer
[clearly stated]

## 💡 Key Takeaway
[one important insight to remember for similar problems]

## ⚠️ Common Mistake to Avoid
[one typical error students make on this type of problem]
"""


def explain_solution(parsed_problem: dict, solution: dict, verification: dict) -> str:
    """
    Generate a student-friendly explanation of the solution.
    Incorporates any corrections from the verifier.
    """
    corrections_note = ""
    if verification.get("corrections"):
        corrections_note = f"\n\nNOTE - The verifier found these issues to address:\n"
        corrections_note += "\n".join(f"- {c}" for c in verification["corrections"])

    user_message = f"""PROBLEM: {parsed_problem.get('problem_text', '')}

TOPIC: {parsed_problem.get('topic', '')} - {parsed_problem.get('subtopic', '')}

SOLUTION WORKED OUT:
{solution.get('solution_text', '')}

FINAL ANSWER: {solution.get('final_answer', '')}
{corrections_note}

Now create a clear, student-friendly explanation of this solution."""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": EXPLAINER_SYSTEM},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3,
            max_tokens=1200
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"""## Solution Explanation

**Problem:** {parsed_problem.get('problem_text', '')}

**Solution:** {solution.get('solution_text', '')}

**Final Answer:** {solution.get('final_answer', '')}

*(Explainer encountered an error: {str(e)})*"""