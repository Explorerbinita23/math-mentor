import os
import re
import tempfile
import logging
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
logger = logging.getLogger(__name__)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Math-specific normalization rules
MATH_PHRASES = {
    r"\bsquare root of\b": "√",
    r"\bcube root of\b": "∛",
    r"\braised to the power\b": "^",
    r"\bto the power of\b": "^",
    r"\bsquared\b": "²",
    r"\bcubed\b": "³",
    r"\bplus or minus\b": "±",
    r"\bgreater than or equal to\b": "≥",
    r"\bless than or equal to\b": "≤",
    r"\bgreater than\b": ">",
    r"\bless than\b": "<",
    r"\btimes\b": "×",
    r"\bdivided by\b": "/",
    r"\bpi\b": "π",
    r"\binfinity\b": "∞",
    r"\bsigma\b": "Σ",
    r"\bdelta\b": "Δ",
    r"\balpha\b": "α",
    r"\bbeta\b": "β",
    r"\bgamma\b": "γ",
    r"\btheta\b": "θ",
    r"\blambda\b": "λ",
    r"\bepsilon\b": "ε",
}


def transcribe_audio(audio_bytes: bytes, filename: str = "audio.wav") -> dict:
    """
    Transcribe audio using Whisper large-v3-turbo via Groq API.
    
    Returns:
        {
            "transcript": str,
            "confidence": float,
            "needs_confirmation": bool,
            "method": str
        }
    """
    try:
        # Write bytes to temp file
        suffix = "." + filename.split(".")[-1] if "." in filename else ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        with open(tmp_path, "rb") as f:
            transcription = client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=(filename, f, _get_audio_mime(filename)),
                response_format="verbose_json",  # includes segments with confidence
                language="en",
                prompt="JEE mathematics problem. The audio may contain math terms like: quadratic, derivative, matrix, probability, integral, equation, logarithm, determinant, eigenvalue."
            )

        raw_transcript = transcription.text.strip()
        
        # Try to get avg_logprob for confidence estimation
        confidence = 0.85  # default
        try:
            if hasattr(transcription, "segments") and transcription.segments:
                avg_logprob = sum(s.avg_logprob for s in transcription.segments) / len(transcription.segments)
                # avg_logprob is typically in [-1, 0], convert to [0,1] confidence
                confidence = min(1.0, max(0.0, 1.0 + avg_logprob))
        except Exception:
            pass

        # Apply math-specific normalizations
        cleaned_transcript = _normalize_math_speech(raw_transcript)

        needs_confirmation = (
            confidence < 0.80 or
            len(cleaned_transcript.strip()) < 10 or
            _contains_ambiguous_math(cleaned_transcript)
        )

        # Cleanup temp file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

        return {
            "transcript": cleaned_transcript,
            "raw_transcript": raw_transcript,
            "confidence": round(confidence, 3),
            "needs_confirmation": needs_confirmation,
            "method": "whisper-large-v3-turbo"
        }

    except Exception as e:
        logger.error(f"ASR error: {e}")
        return {
            "transcript": "",
            "raw_transcript": "",
            "confidence": 0.0,
            "needs_confirmation": True,
            "method": "error",
            "error": str(e)
        }


def _normalize_math_speech(text: str) -> str:
    """Replace spoken math phrases with proper symbols."""
    result = text
    for pattern, replacement in MATH_PHRASES.items():
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result


def _contains_ambiguous_math(text: str) -> bool:
    """Check for phrases that often get transcribed incorrectly."""
    ambiguous_patterns = [
        r"\b(x|y|z)\s+(equals?|is)\s+\d",  # variable assignments
        r"\bto the\b",                        # exponents
        r"\bover\b",                          # fractions
        r"(\d+)\s+([a-z])\b",               # coefficient-variable pairs
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in ambiguous_patterns)


def _get_audio_mime(filename: str) -> str:
    ext = filename.lower().split(".")[-1] if "." in filename else "wav"
    return {
        "wav": "audio/wav", "mp3": "audio/mpeg", "mp4": "audio/mp4",
        "m4a": "audio/mp4", "ogg": "audio/ogg", "webm": "audio/webm",
        "flac": "audio/flac"
    }.get(ext, "audio/wav")