import os
import json
import re
import math
from groq import Groq
from rag.retriever import retrieve_relevant_chunks, retrieve_similar_problems
from dotenv import load_dotenv
load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

SOLVER_SYSTEM = """You are an expert JEE mathematics solver (IIT-JEE level).
You solve problems step-by-step using retrieved formulas and context.

IMPORTANT RULES:
1. Use the retrieved knowledge context to guide your solution
2. Show every step clearly with mathematical notation
3. State which formula/theorem you are using at each step
4. If you use a calculation, show the arithmetic explicitly
5. Box or clearly mark the FINAL ANSWER
6. Do NOT fabricate formulas not in the context — say "I'll use first principles" if needed
7. Be precise and concise

Format your response as:
STRATEGY: [one line strategy]
SOLUTION:
[step-by-step solution]
FINAL ANSWER: [clearly stated answer with units if applicable]
CONFIDENCE: [0.0-1.0] based on how certain you are
"""


def safe_calculate(expression: str) -> str:
    """Safe Python calculator tool for numerical computations."""
    try:
        # Whitelist safe operations only
        allowed = set("0123456789+-*/().,^ %eEijπ")
        expr = expression.replace("^", "**").replace("π", str(math.pi))

        # Only allow safe characters and math functions
        safe_globals = {
            "__builtins__": {},
            "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
            "tan": math.tan, "log": math.log, "log10": math.log10,
            "exp": math.exp, "abs": abs, "round": round,
            "factorial": math.factorial, "pi": math.pi, "e": math.e,
            "pow": pow, "comb": math.comb, "perm": math.perm
        }
        result = eval(expr, safe_globals, {})
        return str(result)
    except Exception as ex:
        return f"Calculation error: {ex}"


def solve_problem(parsed_problem: dict, routing: dict, retrieved_chunks: list) -> dict:
    """
    Core solver using RAG context + LLM reasoning.
    Returns solution dict with steps, answer, confidence, and sources used.
    """
    # Check memory for similar problems
    similar_problems = retrieve_similar_problems(
        parsed_problem.get("problem_text", ""), top_k=2
    )

    # Build context from retrieved chunks
    context_text = "\n\n---\n\n".join([
        f"[Source: {c['source']}]\n{c['content']}"
        for c in retrieved_chunks
    ])

    # Build memory context
    memory_context = ""
    if similar_problems:
        memory_context = "\n\nSIMILAR PREVIOUSLY SOLVED PROBLEMS:\n"
        for sp in similar_problems:
            memory_context += f"- Problem: {sp['problem'][:200]}\n"
            memory_context += f"  Solution summary: {sp['solution_summary'][:300]}\n"
            memory_context += f"  Similarity: {sp['similarity']}\n\n"

    user_message = f"""PROBLEM TO SOLVE:
{parsed_problem.get('problem_text', '')}

Topic: {parsed_problem.get('topic', '')} | Subtopic: {parsed_problem.get('subtopic', '')}
Variables: {parsed_problem.get('variables', [])}
Constraints: {parsed_problem.get('constraints', [])}
What to find: {parsed_problem.get('what_to_find', '')}

SOLUTION STRATEGY: {routing.get('solution_strategy', '')}

RETRIEVED KNOWLEDGE (use these formulas):
{context_text if context_text else "No specific context retrieved - use general knowledge."}
{memory_context}

Solve this step-by-step."""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SOLVER_SYSTEM},
                {"role": "user", "content": user_message}
            ],
            temperature=0.2,
            max_tokens=1500
        )
        solution_text = response.choices[0].message.content.strip()

        # Extract confidence from response
        conf_match = re.search(r"CONFIDENCE:\s*([\d.]+)", solution_text)
        confidence = float(conf_match.group(1)) if conf_match else 0.75

        # Extract final answer
        answer_match = re.search(r"FINAL ANSWER:\s*(.+?)(?:\n|CONFIDENCE|$)", solution_text, re.DOTALL)
        final_answer = answer_match.group(1).strip() if answer_match else "See solution above"

        return {
            "solution_text": solution_text,
            "final_answer": final_answer,
            "confidence": min(max(confidence, 0.0), 1.0),
            "retrieved_chunks": retrieved_chunks,
            "similar_problems_used": similar_problems,
            "memory_reused": len(similar_problems) > 0,
            "error": None
        }

    except Exception as e:
        return {
            "solution_text": f"Solver error: {str(e)}",
            "final_answer": "Unable to solve",
            "confidence": 0.0,
            "retrieved_chunks": retrieved_chunks,
            "similar_problems_used": [],
            "memory_reused": False,
            "error": str(e)
        }