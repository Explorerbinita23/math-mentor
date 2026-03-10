from dotenv import load_dotenv
load_dotenv()
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import streamlit as st
import time
import uuid
from agents.parser_agent import parse_problem
from agents.router_agent import route_problem
from agents.solver_agent import solve_problem
from agents.verifier_agent import verify_solution
from agents.explainer_agent import explain_solution
from rag.retriever import (
    retrieve_relevant_chunks, ingest_knowledge_base,
    store_solved_problem, retrieve_similar_problems
)
from utils.ocr import extract_text_from_image
from utils.asr import transcribe_audio

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🧮 Math Mentor",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Light Theme CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global background & text ── */
    .stApp {
        background-color: #f5f7fa;
        color: #1a1a2e;
    }
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e4ed;
    }
    .block-container {
        padding: 1.5rem 2rem;
        max-width: 1400px;
    }

    /* ── Header ── */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px 28px;
        border-radius: 16px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(102,126,234,0.3);
    }
    .main-header h1 { color: white; margin: 0; font-size: 2rem; }
    .main-header p  { color: rgba(255,255,255,0.85); margin: 4px 0 0 0; font-size: 0.95rem; }

    /* ── Cards ── */
    .card {
        background: #ffffff;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        border: 1px solid #e8ecf4;
    }

    /* ── Agent trace ── */
    .agent-done    { border-left: 4px solid #22c55e !important; }
    .agent-running { border-left: 4px solid #f59e0b !important; }

    /* ── HITL banner ── */
    .hitl-banner {
        background: #fff7ed;
        border: 2px solid #f97316;
        border-radius: 12px;
        padding: 18px;
        margin: 14px 0;
    }

    /* ── Answer box ── */
    .answer-box {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border: 2px solid #10b981;
        border-radius: 14px;
        padding: 20px 24px;
        margin: 16px 0;
    }
    .answer-box h3 { color: #065f46; margin: 0 0 8px 0; }
    .answer-box p  { color: #064e3b; font-size: 1.2rem; font-weight: 600; margin: 0; }

    /* ── Metric cards ── */
    div[data-testid="metric-container"] {
        background: #ffffff;
        border: 1px solid #e8ecf4;
        border-radius: 10px;
        padding: 12px 16px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }

    /* ── Confidence badge ── */
    .conf-high { color: #16a34a; font-weight: 700; }
    .conf-mid  { color: #d97706; font-weight: 700; }
    .conf-low  { color: #dc2626; font-weight: 700; }

    /* ── Source chips ── */
    .source-chip {
        display: inline-block;
        background: #ede9fe;
        color: #5b21b6;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        margin: 2px;
        font-weight: 500;
    }

    /* ── OCR / ASR info bar ── */
    .info-bar {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 8px;
        padding: 8px 14px;
        font-size: 0.85rem;
        color: #1e40af;
        margin: 6px 0;
    }

    /* ── Feedback buttons ── */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background: #f8fafc !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
    }

    /* ── Sidebar items ── */
    .sidebar-topic {
        background: #f0f4ff;
        border-radius: 8px;
        padding: 6px 10px;
        margin: 3px 0;
        font-size: 0.85rem;
        color: #3730a3;
    }

    /* ── Progress bar colour ── */
    .stProgress > div > div { background-color: #667eea !important; }

    /* ── Divider ── */
    hr { border-color: #e8ecf4 !important; }
</style>
""", unsafe_allow_html=True)


# ── Session State Init ────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "kb_ready": False,
        "result": None,
        "hitl_pending": False,
        "hitl_context": {},
        "extracted_text": "",
        "ocr_confidence": 0.0,
        "ocr_lines": [],
        "asr_confidence": 0.0,
        "session_id": None,
        "feedback_given": False,
        "show_feedback_form": False,
        "trace": []
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── Knowledge Base Init ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="📚 Loading knowledge base...")
def load_knowledge_base():
    return ingest_knowledge_base()

if not st.session_state.kb_ready:
    with st.spinner("🔄 Initialising knowledge base..."):
        load_knowledge_base()
        st.session_state.kb_ready = True


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧮 Math Mentor")
    st.caption("JEE-level AI Math Tutor")
    st.divider()

    input_mode = st.radio(
        "📥 **Input Mode**",
        options=["✏️ Text", "🖼️ Image", "🎙️ Audio"],
        index=0
    )
    st.divider()

    st.markdown("**📐 Topics Covered**")
    for topic in ["Algebra", "Probability", "Calculus", "Linear Algebra"]:
        st.markdown(f'<div class="sidebar-topic">• {topic}</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("**🤖 Agents**")
    for agent in ["Parser", "Router", "Solver", "Verifier", "Explainer"]:
        st.caption(f"✅ {agent} Agent")

    st.divider()
    if st.button("🔄 Reset Session", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# ── Helpers ───────────────────────────────────────────────────────────────────
def confidence_color(conf: float) -> str:
    if conf >= 0.80: return "🟢"
    if conf >= 0.60: return "🟡"
    return "🔴"

def confidence_label(conf: float) -> str:
    if conf >= 0.80: return "High"
    if conf >= 0.60: return "Medium"
    return "Low"


# ── Pipeline ──────────────────────────────────────────────────────────────────
def run_full_pipeline(raw_text: str, input_type: str = "text"):
    session_id = str(uuid.uuid4())[:8]
    st.session_state.session_id = session_id
    trace = []

    progress = st.progress(0, text="🚀 Starting pipeline...")

    # 1. Parser
    progress.progress(15, text="🔍 Parser Agent: Structuring problem...")
    trace.append({"agent": "ParserAgent", "status": "running"})
    parsed = parse_problem(raw_text)
    trace[-1]["status"] = "done"
    trace[-1]["output"] = parsed

    if parsed.get("needs_clarification"):
        progress.empty()
        st.session_state.hitl_pending = True
        st.session_state.hitl_context = {
            "type": "parser_ambiguity",
            "parsed": parsed,
            "raw_text": raw_text,
            "reason": parsed.get("clarification_reason", "")
        }
        st.session_state.trace = trace
        return None

    # 2. Router
    progress.progress(30, text="🧭 Router Agent: Planning strategy...")
    trace.append({"agent": "RouterAgent", "status": "running"})
    routing = route_problem(parsed)
    trace[-1]["status"] = "done"
    trace[-1]["output"] = routing

    # 3. RAG
    progress.progress(45, text="📚 Retrieving relevant formulas...")
    trace.append({"agent": "RAGRetriever", "status": "running"})
    chunks = retrieve_relevant_chunks(routing.get("rag_query", raw_text), top_k=4)
    trace[-1]["status"] = "done"
    trace[-1]["chunks"] = chunks

    # 4. Solver
    progress.progress(60, text="🔢 Solver Agent: Working through the problem...")
    trace.append({"agent": "SolverAgent", "status": "running"})
    solution = solve_problem(parsed, routing, chunks)
    trace[-1]["status"] = "done"
    trace[-1]["output"] = solution

    # 5. Verifier
    progress.progress(80, text="✔️ Verifier Agent: Checking correctness...")
    trace.append({"agent": "VerifierAgent", "status": "running"})
    verification = verify_solution(parsed, solution)
    trace[-1]["status"] = "done"
    trace[-1]["output"] = verification

    needs_hitl = verification.get("needs_human_review", False)

    # 6. Explainer
    progress.progress(95, text="💬 Explainer Agent: Preparing explanation...")
    trace.append({"agent": "ExplainerAgent", "status": "running"})
    explanation = explain_solution(parsed, solution, verification)
    trace[-1]["status"] = "done"

    progress.progress(100, text="✅ Done!")
    time.sleep(0.3)
    progress.empty()

    store_solved_problem(
        problem_id=f"prob_{session_id}",
        problem_text=parsed["problem_text"],
        solution=solution["solution_text"],
        topic=parsed["topic"],
        feedback="pending"
    )

    result = {
        "session_id": session_id,
        "parsed": parsed,
        "routing": routing,
        "chunks": chunks,
        "solution": solution,
        "verification": verification,
        "explanation": explanation,
        "hitl_required": needs_hitl
    }

    if needs_hitl:
        st.session_state.hitl_pending = True
        st.session_state.hitl_context = {
            "type": "low_confidence",
            "result": result,
            "reason": verification.get("review_reason", "Low confidence")
        }

    st.session_state.trace = trace
    return result


# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🧮 Math Mentor</h1>
    <p>JEE-level AI Tutor · Multimodal · RAG-powered · Multi-agent · Human-in-the-loop</p>
</div>
""", unsafe_allow_html=True)

# ── Two Column Layout ─────────────────────────────────────────────────────────
col_input, col_right = st.columns([1.4, 1], gap="large")

with col_input:
    st.markdown("### 📥 Problem Input")
    raw_input = None
    input_type = "text"

    # ── TEXT ──────────────────────────────────────────────────────────────────
    if input_mode == "✏️ Text":
        raw_input = st.text_area(
            "Type your math problem below",
            placeholder=(
                "e.g. Find the roots of x² + 5x + 6 = 0\n"
                "e.g. If P(A)=0.4, P(B)=0.3, A and B independent, find P(A∪B)\n"
                "e.g. Find the derivative of f(x) = x³ + 2x² - 5x + 1"
            ),
            height=150
        )
        input_type = "text"

    # ── IMAGE ─────────────────────────────────────────────────────────────────
    elif input_mode == "🖼️ Image":
        uploaded_img = st.file_uploader(
            "Upload image (JPG/PNG)", type=["jpg", "jpeg", "png"]
        )
        if uploaded_img:
            st.image(uploaded_img, caption="Uploaded image", use_container_width=True)
            if st.button("🔍 Extract Text from Image", use_container_width=True):
                with st.spinner("Running EasyOCR..."):
                    ocr_result = extract_text_from_image(
                        uploaded_img.getvalue(), uploaded_img.name
                    )
                st.session_state.extracted_text  = ocr_result.get("extracted_text", "")
                st.session_state.ocr_confidence  = ocr_result.get("confidence", 0)
                st.session_state.ocr_lines       = ocr_result.get("lines", [])
                st.rerun()

        if st.session_state.extracted_text:
            conf = st.session_state.ocr_confidence
            quality = "✅ Good" if conf >= 0.75 else "⚠️ Low — please verify math symbols"
            st.markdown(
                f'<div class="info-bar">'
                f'OCR Confidence: {confidence_color(conf)} <strong>{conf:.0%}</strong> '
                f'({confidence_label(conf)}) — {quality}'
                f'</div>',
                unsafe_allow_html=True
            )

            edited = st.text_area(
                "✏️ Extracted Text (edit if needed — check superscripts like x²)",
                value=st.session_state.extracted_text,
                height=120
            )
            raw_input  = edited
            input_type = "image"

            # OCR line details
            if st.session_state.ocr_lines:
                with st.expander("🔍 OCR Line Details", expanded=False):
                    for line in st.session_state.ocr_lines:
                        c = line.get("confidence", 0)
                        icon = "🟢" if c > 0.8 else "🟡" if c > 0.6 else "🔴"
                        st.write(f"{icon} `{line['text']}` — {c:.0%}")

    # ── AUDIO ─────────────────────────────────────────────────────────────────
    elif input_mode == "🎙️ Audio":
        mic_tab, upload_tab = st.tabs(["🎤 Record via Mic", "📁 Upload Audio File"])

        with mic_tab:
            st.markdown("Record your math problem directly in the browser:")
            mic_html = """
            <style>
                body { font-family: sans-serif; background: transparent; }
                .mic-btn {
                    background:#4f46e5; color:white; border:none;
                    padding:12px 26px; font-size:15px; border-radius:50px;
                    cursor:pointer; margin:5px; transition:all 0.2s;
                }
                .mic-btn.recording { background:#ef4444; animation:pulse 1.2s infinite; }
                .mic-btn:disabled  { background:#9ca3af; cursor:not-allowed; }
                @keyframes pulse {
                    0%  { box-shadow:0 0 0 0 rgba(239,68,68,0.5); }
                    70% { box-shadow:0 0 0 12px rgba(239,68,68,0); }
                    100%{ box-shadow:0 0 0 0 rgba(239,68,68,0); }
                }
                #timer { font-size:20px; font-weight:bold; color:#4f46e5; margin:8px 0; }
                #rec-status {
                    background:#f0f4ff; border:1px solid #c7d2fe;
                    border-radius:8px; padding:8px 14px;
                    font-size:13px; color:#3730a3; margin-top:10px;
                }
            </style>
            <button class="mic-btn" id="startBtn" onclick="startRec()">🎤 Start Recording</button>
            <button class="mic-btn" id="stopBtn" onclick="stopRec()" disabled>⏹ Stop</button>
            <div id="timer"></div>
            <div id="rec-status">Press Start and speak your math problem clearly.</div>
            <div id="audio-preview" style="margin-top:10px"></div>
            <script>
            let mr, chunks=[], timerInterval, seconds=0;
            function updateTimer(){
                seconds++;
                document.getElementById('timer').innerText =
                    '⏱ '+String(Math.floor(seconds/60)).padStart(2,'0')+':'+String(seconds%60).padStart(2,'0');
            }
            async function startRec(){
                chunks=[]; seconds=0;
                document.getElementById('audio-preview').innerHTML='';
                const stream = await navigator.mediaDevices.getUserMedia({audio:true});
                mr = new MediaRecorder(stream,{mimeType:'audio/webm'});
                mr.ondataavailable = e => { if(e.data.size>0) chunks.push(e.data); };
                mr.onstop = () => {
                    const blob = new Blob(chunks,{type:'audio/webm'});
                    const url = URL.createObjectURL(blob);
                    document.getElementById('audio-preview').innerHTML =
                        '<audio controls src="'+url+'" style="width:100%;margin-top:8px"></audio>'+
                        '<br><a href="'+url+'" download="recording.wav" '+
                        'style="color:#4f46e5;font-weight:500">⬇ Download recording.wav</a>';
                    document.getElementById('rec-status').innerText =
                        '✅ Done! Download above then upload below to transcribe.';
                    clearInterval(timerInterval);
                };
                mr.start(100);
                timerInterval = setInterval(updateTimer, 1000);
                document.getElementById('startBtn').classList.add('recording');
                document.getElementById('startBtn').disabled=true;
                document.getElementById('stopBtn').disabled=false;
                document.getElementById('rec-status').innerText='🔴 Recording in progress...';
            }
            function stopRec(){
                mr.stop(); mr.stream.getTracks().forEach(t=>t.stop());
                document.getElementById('startBtn').classList.remove('recording');
                document.getElementById('startBtn').disabled=false;
                document.getElementById('stopBtn').disabled=true;
            }
            </script>
            """
            st.components.v1.html(mic_html, height=240)
            st.info("⬇️ Download your recording above, then upload it here:")
            recorded_file = st.file_uploader(
                "Upload recorded file",
                type=["wav","mp3","m4a","ogg","webm"],
                key="mic_recorded"
            )
            if recorded_file:
                st.audio(recorded_file)
                if st.button("📝 Transcribe Recording", use_container_width=True, key="transcribe_mic"):
                    with st.spinner("Transcribing with Whisper large-v3-turbo..."):
                        asr_result = transcribe_audio(recorded_file.getvalue(), recorded_file.name)
                    st.session_state.extracted_text = asr_result.get("transcript", "")
                    st.session_state.asr_confidence = asr_result.get("confidence", 0)
                    st.rerun()

        with upload_tab:
            uploaded_audio = st.file_uploader(
                "Upload audio file",
                type=["wav","mp3","m4a","ogg","webm"],
                key="audio_upload"
            )
            if uploaded_audio:
                st.audio(uploaded_audio)
                if st.button("🎙️ Transcribe Audio", use_container_width=True, key="transcribe_upload"):
                    with st.spinner("Transcribing with Whisper large-v3-turbo..."):
                        asr_result = transcribe_audio(uploaded_audio.getvalue(), uploaded_audio.name)
                    st.session_state.extracted_text = asr_result.get("transcript", "")
                    st.session_state.asr_confidence = asr_result.get("confidence", 0)
                    st.rerun()

        if st.session_state.extracted_text:
            conf = st.session_state.asr_confidence
            quality = "✅ Clear" if conf >= 0.80 else "⚠️ Unclear — please verify"
            st.markdown(
                f'<div class="info-bar">'
                f'Transcription Confidence: {confidence_color(conf)} <strong>{conf:.0%}</strong> — {quality}'
                f'</div>',
                unsafe_allow_html=True
            )
            edited = st.text_area(
                "✏️ Transcript (edit if needed)",
                value=st.session_state.extracted_text,
                height=120
            )
            raw_input  = edited
            input_type = "audio"

    # ── SOLVE BUTTON ──────────────────────────────────────────────────────────
    st.markdown("")
    solve_btn = st.button(
        "🚀 Solve Problem",
        disabled=not raw_input or not raw_input.strip(),
        use_container_width=True,
        type="primary"
    )

    if solve_btn and raw_input and raw_input.strip():
        st.session_state.result           = None
        st.session_state.hitl_pending     = False
        st.session_state.feedback_given   = False
        st.session_state.show_feedback_form = False
        result = run_full_pipeline(raw_input.strip(), input_type)
        if result:
            st.session_state.result = result
        st.rerun()


# ── RIGHT PANEL: Agent Trace + RAG Context ────────────────────────────────────
with col_right:
    if st.session_state.trace:
        st.markdown("### 🔄 Agent Trace")
        for step in st.session_state.trace:
            agent  = step["agent"]
            status = step["status"]
            icon   = "✅" if status == "done" else "⏳"
            with st.expander(f"{icon} {agent}", expanded=False):
                if "output" in step:
                    out = step["output"]
                    if isinstance(out, dict):
                        display = {k: v for k, v in out.items()
                                   if k not in ("solution_text","retrieved_chunks","similar_problems_used")}
                        st.json(display)
                    else:
                        st.write(out)
                if "chunks" in step:
                    st.caption(f"📦 Retrieved {len(step['chunks'])} chunks")

    if st.session_state.result and st.session_state.result.get("chunks"):
        st.markdown("### 📚 Retrieved Context")
        for i, chunk in enumerate(st.session_state.result["chunks"]):
            rel = chunk['relevance_score']
            rel_icon = "🟢" if rel > 0.8 else "🟡" if rel > 0.6 else "🔴"
            with st.expander(
                f"{rel_icon} [{i+1}] {chunk['source']} — relevance: {rel}",
                expanded=False
            ):
                st.markdown(
                    f'<span class="source-chip">{chunk["topic"]}</span>',
                    unsafe_allow_html=True
                )
                st.text(chunk["content"][:400])


# ── HITL SECTION ──────────────────────────────────────────────────────────────
if st.session_state.hitl_pending:
    ctx = st.session_state.hitl_context
    st.markdown('<div class="hitl-banner">', unsafe_allow_html=True)
    st.warning(f"⚠️ **Human Review Required** — {ctx.get('reason', '')}")

    if ctx.get("type") == "parser_ambiguity":
        st.write("The parser found ambiguity. Please clarify the problem:")
        clarified = st.text_area(
            "Clarified problem:", value=ctx.get("raw_text", ""), height=100
        )
        if st.button("✅ Submit Clarification", type="primary"):
            result = run_full_pipeline(clarified, "text")
            if result:
                st.session_state.result = result
            st.session_state.hitl_pending = False
            st.rerun()

    elif ctx.get("type") == "low_confidence":
        result = ctx.get("result", {})
        st.write("**Proposed answer:**",
                 result.get("solution", {}).get("final_answer", ""))
        issues = result.get("verification", {}).get("issues_found", [])
        if issues:
            st.write("**Issues found:**")
            for issue in issues:
                st.write(f"  - {issue}")

        hcol1, hcol2 = st.columns(2)
        with hcol1:
            if st.button("✅ Approve Solution", type="primary", use_container_width=True):
                st.session_state.result = result
                st.session_state.hitl_pending = False
                store_solved_problem(
                    problem_id=f"hitl_{result.get('session_id')}",
                    problem_text=result["parsed"]["problem_text"],
                    solution=result["solution"]["solution_text"],
                    topic=result["parsed"]["topic"],
                    feedback="approved_hitl"
                )
                st.rerun()
        with hcol2:
            correct_ans = st.text_input("Enter correct answer (optional):")
            if st.button("❌ Reject & Override", use_container_width=True):
                st.session_state.hitl_pending = False
                if correct_ans:
                    st.success(f"Correct answer recorded: **{correct_ans}**")
                    store_solved_problem(
                        problem_id=f"hitl_corrected_{result.get('session_id')}",
                        problem_text=result["parsed"]["problem_text"],
                        solution=f"Correct answer: {correct_ans}",
                        topic=result["parsed"]["topic"],
                        feedback="corrected"
                    )
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# ── RESULTS SECTION ───────────────────────────────────────────────────────────
if st.session_state.result and not st.session_state.hitl_pending:
    result       = st.session_state.result
    parsed       = result.get("parsed", {})
    verification = result.get("verification", {})
    solution     = result.get("solution", {})
    conf         = verification.get("combined_confidence", solution.get("confidence", 0.75))
    is_correct   = verification.get("is_correct", False)

    st.divider()

    # ── Metrics row ───────────────────────────────────────────────────────────
    rcol1, rcol2, rcol3, rcol4 = st.columns([2, 1, 1, 1])

    with rcol1:
        st.markdown("#### 📋 Problem Analysis")
        st.write(f"**Topic:** `{parsed.get('topic','N/A')}` — `{parsed.get('subtopic','')}`")
        st.write(f"**Variables:** {', '.join(parsed.get('variables',[])) or 'N/A'}")
        st.write(f"**Finding:** {parsed.get('what_to_find','N/A')}")

    with rcol2:
        st.metric(
            label=f"{confidence_color(conf)} Confidence",
            value=f"{conf:.0%}",
            delta=confidence_label(conf)
        )

    with rcol3:
        st.metric(
            label="✔️ Verified",
            value="✅ OK" if is_correct else "⚠️ Review"
        )

    with rcol4:
        mem_reused = solution.get("memory_reused", False)
        st.metric(
            label="🧠 Memory",
            value="✅ Reused" if mem_reused else "🆕 New"
        )

    st.divider()

    # ── Final Answer ──────────────────────────────────────────────────────────
    final_ans = solution.get("final_answer", "See explanation below")
    st.markdown(
        f'<div class="answer-box">'
        f'<h3>🎯 Final Answer</h3>'
        f'<p>{final_ans}</p>'
        f'</div>',
        unsafe_allow_html=True
    )

    # ── Verifier notes ────────────────────────────────────────────────────────
    if verification.get("issues_found"):
        with st.expander("⚠️ Verifier Notes", expanded=False):
            for issue in verification["issues_found"]:
                st.write(f"- {issue}")

    # ── Step-by-step explanation ──────────────────────────────────────────────
    st.markdown("#### 📖 Step-by-Step Explanation")
    explanation = result.get("explanation", "")
    st.markdown(explanation)

    # ── Memory reuse details ──────────────────────────────────────────────────
    if solution.get("memory_reused") and solution.get("similar_problems_used"):
        with st.expander("🧠 Memory Reuse — Similar Past Problems", expanded=False):
            for sp in solution["similar_problems_used"]:
                st.markdown(f"**Similarity:** {sp['similarity']:.0%}")
                st.markdown(f"**Past Problem:** {sp['problem'][:200]}")
                st.markdown(f"**Past Solution:** {sp['solution_summary'][:300]}")
                st.divider()

    # ── Feedback ──────────────────────────────────────────────────────────────
    st.divider()
    if not st.session_state.feedback_given:
        st.markdown("#### 💬 Was this solution helpful?")
        fb1, fb2, fb3 = st.columns([1, 1, 4])
        with fb1:
            if st.button("👍 Correct", use_container_width=True, type="primary"):
                store_solved_problem(
                    problem_id=f"feedback_{result.get('session_id')}",
                    problem_text=parsed.get("problem_text", ""),
                    solution=solution.get("solution_text", ""),
                    topic=parsed.get("topic", "unknown"),
                    feedback="correct"
                )
                st.session_state.feedback_given = True
                st.rerun()
        with fb2:
            if st.button("👎 Incorrect", use_container_width=True):
                st.session_state.show_feedback_form = True
                st.rerun()

        if st.session_state.get("show_feedback_form"):
            comment     = st.text_input("What was wrong?")
            correct_ans = st.text_input("Correct answer (optional):")
            if st.button("📨 Submit Feedback", type="primary"):
                sol_text = solution.get("solution_text", "")
                if correct_ans:
                    sol_text = f"User correction: {correct_ans}\n\n" + sol_text
                store_solved_problem(
                    problem_id=f"feedback_corrected_{result.get('session_id')}",
                    problem_text=parsed.get("problem_text", ""),
                    solution=sol_text,
                    topic=parsed.get("topic", "unknown"),
                    feedback=f"incorrect: {comment}"
                )
                st.session_state.feedback_given     = True
                st.session_state.show_feedback_form = False
                st.rerun()
    else:
        st.success("✅ Feedback recorded — thank you for helping the system improve!")