import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

ROUTER_SYSTEM = """You are a math problem router for a JEE tutoring system.
Given a parsed problem, decide the solution strategy and workflow.

Respond in this exact JSON format (no markdown):
{
  "topic": "algebra|probability|calculus|linear_algebra",
  "difficulty": "easy|medium|hard",
  "solution_strategy": "brief strategy description",
  "tools_needed": ["calculator", "symbolic_math"],
  "rag_query": "optimized search query to retrieve relevant formulas",
  "confidence": 0.95,
  "workflow_steps": ["step1", "step2", "step3"]
}

tools_needed options: "calculator" (for numerical computation), "symbolic_math" (for algebraic manipulation)
"""


def route_problem(parsed_problem: dict) -> dict:
    """
    Classify problem and determine optimal solution strategy.
    Returns routing decision with workflow plan.
    """
    try:
        problem_summary = f"""
Topic: {parsed_problem.get('topic', 'unknown')}
Subtopic: {parsed_problem.get('subtopic', '')}
Problem: {parsed_problem.get('problem_text', '')}
What to find: {parsed_problem.get('what_to_find', '')}
Variables: {parsed_problem.get('variables', [])}
Constraints: {parsed_problem.get('constraints', [])}
"""
        from groq import Groq
        import json, re
        c = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        response = c.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM},
                {"role": "user", "content": f"Route this problem:{problem_summary}"}
            ],
            temperature=0.1,
            max_tokens=400
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"```(?:json)?", "", raw).strip("` \n")
        return json.loads(raw)

    except Exception as e:
        # Fallback routing based on parsed topic
        topic = parsed_problem.get("topic", "algebra")
        return {
            "topic": topic,
            "difficulty": "medium",
            "solution_strategy": f"Direct solution using {topic} techniques",
            "tools_needed": ["calculator"],
            "rag_query": f"{topic} {parsed_problem.get('subtopic', '')} formulas solution",
            "confidence": 0.7,
            "workflow_steps": ["Identify formula", "Substitute values", "Compute answer", "Verify"]
        }