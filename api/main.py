import os
import sys
import uuid
import json
import logging
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from agents.parser_agent import parse_problem
from agents.router_agent import route_problem
from agents.solver_agent import solve_problem
from agents.verifier_agent import verify_solution
from agents.explainer_agent import explain_solution
from rag.retriever import (
    retrieve_relevant_chunks,
    ingest_knowledge_base,
    store_solved_problem,
    retrieve_similar_problems
)
from utils.ocr import extract_text_from_image
from utils.asr import transcribe_audio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Math Mentor API",
    description="JEE Math Solver with RAG + Multi-Agent System",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ────────────────────────────────────────────────────────────────────

class TextProblemRequest(BaseModel):
    problem_text: str
    session_id: Optional[str] = None


class HITLFeedbackRequest(BaseModel):
    session_id: str
    corrected_problem: Optional[str] = None
    approved: bool
    feedback_text: Optional[str] = None
    correct_answer: Optional[str] = None


class FeedbackRequest(BaseModel):
    session_id: str
    is_correct: bool
    feedback_comment: Optional[str] = None


# In-memory session store (use Redis for production)
sessions: dict = {}


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    logger.info("Ingesting knowledge base into ChromaDB...")
    count = ingest_knowledge_base()
    logger.info(f"Knowledge base ready: {count} chunks indexed.")


# ── Core Pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(raw_text: str, session_id: str, input_type: str = "text") -> dict:
    """
    Full 5-agent pipeline: Parse → Route → Retrieve → Solve → Verify → Explain
    """
    trace = []

    # 1. Parser Agent
    trace.append({"agent": "ParserAgent", "status": "running", "input": raw_text[:100]})
    parsed = parse_problem(raw_text)
    trace[-1]["status"] = "done"
    trace[-1]["output"] = parsed

    # Trigger HITL if parser needs clarification
    if parsed.get("needs_clarification"):
        sessions[session_id] = {
            "state": "awaiting_hitl",
            "raw_text": raw_text,
            "parsed": parsed,
            "input_type": input_type,
            "trace": trace
        }
        return {
            "session_id": session_id,
            "hitl_required": True,
            "hitl_reason": parsed.get("clarification_reason", "Ambiguous problem"),
            "parsed_problem": parsed,
            "trace": trace
        }

    # 2. Intent Router Agent
    trace.append({"agent": "RouterAgent", "status": "running"})
    routing = route_problem(parsed)
    trace[-1]["status"] = "done"
    trace[-1]["output"] = routing

    # 3. RAG Retrieval
    trace.append({"agent": "RAGRetriever", "status": "running",
                  "query": routing.get("rag_query", "")})
    chunks = retrieve_relevant_chunks(routing.get("rag_query", parsed["problem_text"]), top_k=4)
    trace[-1]["status"] = "done"
    trace[-1]["output"] = {"chunks_retrieved": len(chunks)}

    # 4. Solver Agent
    trace.append({"agent": "SolverAgent", "status": "running"})
    solution = solve_problem(parsed, routing, chunks)
    trace[-1]["status"] = "done"
    trace[-1]["output"] = {
        "confidence": solution.get("confidence"),
        "memory_reused": solution.get("memory_reused")
    }

    # 5. Verifier Agent
    trace.append({"agent": "VerifierAgent", "status": "running"})
    verification = verify_solution(parsed, solution)
    trace[-1]["status"] = "done"
    trace[-1]["output"] = {
        "is_correct": verification.get("is_correct"),
        "combined_confidence": verification.get("combined_confidence")
    }

    # Trigger HITL if verifier not confident
    if verification.get("needs_human_review"):
        sessions[session_id] = {
            "state": "awaiting_hitl",
            "parsed": parsed,
            "routing": routing,
            "solution": solution,
            "verification": verification,
            "trace": trace,
            "input_type": input_type
        }
        # Still run explainer for display
        explanation = explain_solution(parsed, solution, verification)
        trace.append({"agent": "ExplainerAgent", "status": "done"})

        return {
            "session_id": session_id,
            "hitl_required": True,
            "hitl_reason": verification.get("review_reason", "Low confidence"),
            "parsed_problem": parsed,
            "routing": routing,
            "solution": solution,
            "verification": verification,
            "explanation": explanation,
            "retrieved_chunks": chunks,
            "trace": trace
        }

    # 6. Explainer Agent
    trace.append({"agent": "ExplainerAgent", "status": "running"})
    explanation = explain_solution(parsed, solution, verification)
    trace[-1]["status"] = "done"

    # Store in memory
    problem_id = f"prob_{session_id}"
    store_solved_problem(
        problem_id=problem_id,
        problem_text=parsed["problem_text"],
        solution=solution["solution_text"],
        topic=parsed["topic"],
        feedback="pending"
    )

    result = {
        "session_id": session_id,
        "hitl_required": False,
        "parsed_problem": parsed,
        "routing": routing,
        "solution": solution,
        "verification": verification,
        "explanation": explanation,
        "retrieved_chunks": chunks,
        "trace": trace
    }
    sessions[session_id] = {"state": "complete", "result": result}
    return result


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.post("/solve/text")
async def solve_text(request: TextProblemRequest):
    session_id = request.session_id or str(uuid.uuid4())
    try:
        result = run_pipeline(request.problem_text, session_id, "text")
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/solve/image")
async def solve_image(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None)
):
    session_id = session_id or str(uuid.uuid4())
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    
    image_bytes = await file.read()
    
    # OCR extraction
    ocr_result = extract_text_from_image(image_bytes, file.filename or "image.jpg")
    
    # If OCR confidence is low → return for HITL
    if ocr_result.get("needs_review") or ocr_result.get("confidence", 0) < 0.75:
        sessions[session_id] = {
            "state": "awaiting_ocr_review",
            "ocr_result": ocr_result,
            "input_type": "image"
        }
        return JSONResponse(content={
            "session_id": session_id,
            "hitl_required": True,
            "hitl_reason": f"Low OCR confidence ({ocr_result.get('confidence', 0):.0%}). Please verify extracted text.",
            "extracted_text": ocr_result.get("extracted_text", ""),
            "ocr_confidence": ocr_result.get("confidence", 0),
            "ocr_method": ocr_result.get("method", "unknown")
        })
    
    # OCR was confident, proceed with pipeline
    result = run_pipeline(ocr_result["extracted_text"], session_id, "image")
    result["ocr_info"] = {
        "extracted_text": ocr_result["extracted_text"],
        "confidence": ocr_result["confidence"],
        "method": ocr_result["method"]
    }
    return JSONResponse(content=result)


@app.post("/solve/audio")
async def solve_audio(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None)
):
    session_id = session_id or str(uuid.uuid4())
    audio_bytes = await file.read()
    
    # ASR transcription
    asr_result = transcribe_audio(audio_bytes, file.filename or "audio.wav")
    
    # If transcription unclear → return for confirmation
    if asr_result.get("needs_confirmation") or not asr_result.get("transcript"):
        sessions[session_id] = {
            "state": "awaiting_asr_review",
            "asr_result": asr_result,
            "input_type": "audio"
        }
        return JSONResponse(content={
            "session_id": session_id,
            "hitl_required": True,
            "hitl_reason": f"Transcription may be unclear ({asr_result.get('confidence', 0):.0%} confidence). Please verify.",
            "transcript": asr_result.get("transcript", ""),
            "asr_confidence": asr_result.get("confidence", 0)
        })
    
    result = run_pipeline(asr_result["transcript"], session_id, "audio")
    result["asr_info"] = {
        "transcript": asr_result["transcript"],
        "confidence": asr_result["confidence"]
    }
    return JSONResponse(content=result)


@app.post("/hitl/confirm")
async def hitl_confirm(request: HITLFeedbackRequest):
    """
    Human-in-the-loop confirmation endpoint.
    Called when user approves/edits/rejects a solution or OCR/ASR output.
    """
    session = sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    state = session.get("state")

    # Handle OCR / ASR review
    if state in ("awaiting_ocr_review", "awaiting_asr_review"):
        confirmed_text = request.corrected_problem
        if not confirmed_text:
            raise HTTPException(status_code=400, detail="corrected_problem required")
        result = run_pipeline(confirmed_text, request.session_id,
                              session.get("input_type", "text"))
        return JSONResponse(content=result)

    # Handle solution review
    if state == "awaiting_hitl":
        if not request.approved:
            # User rejected — return with their correction
            return JSONResponse(content={
                "session_id": request.session_id,
                "status": "rejected",
                "message": "Solution rejected by reviewer.",
                "user_feedback": request.feedback_text,
                "correct_answer": request.correct_answer
            })
        # User approved — re-run if correction provided
        if request.corrected_problem:
            result = run_pipeline(request.corrected_problem, request.session_id,
                                  session.get("input_type", "text"))
        else:
            result = session.get("result", {})

        # Store feedback in memory
        if session.get("parsed"):
            store_solved_problem(
                problem_id=f"prob_{request.session_id}_hitl",
                problem_text=session["parsed"]["problem_text"],
                solution=session.get("solution", {}).get("solution_text", ""),
                topic=session["parsed"]["topic"],
                feedback="approved_hitl"
            )
        return JSONResponse(content=result)

    raise HTTPException(status_code=400, detail=f"Session in unexpected state: {state}")


@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Store user feedback (✅/❌) and update memory."""
    session = sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    result = session.get("result", {})
    parsed = result.get("parsed_problem", {})
    solution = result.get("solution", {})

    feedback_label = "correct" if request.is_correct else "incorrect"
    if parsed and solution:
        store_solved_problem(
            problem_id=f"prob_{request.session_id}_feedback",
            problem_text=parsed.get("problem_text", ""),
            solution=solution.get("solution_text", ""),
            topic=parsed.get("topic", "unknown"),
            feedback=feedback_label
        )

    return {"status": "feedback_stored", "feedback": feedback_label}


@app.post("/ingest")
async def reingest_knowledge():
    """Manually trigger knowledge base re-ingestion."""
    count = ingest_knowledge_base()
    return {"status": "ok", "chunks_ingested": count}


@app.get("/memory/similar")
async def find_similar(query: str, top_k: int = 3):
    """Search memory for similar past problems."""
    similar = retrieve_similar_problems(query, top_k=top_k)
    return {"similar_problems": similar, "count": len(similar)}