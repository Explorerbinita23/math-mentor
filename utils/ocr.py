import os
import logging
import io
import numpy as np
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

# EasyOCR reader — loaded once and reused
_reader = None


def get_reader():
    """Lazy load EasyOCR reader (downloads model on first run ~500MB)."""
    global _reader
    if _reader is None:
        logger.info("Loading EasyOCR model (first time may take a minute)...")
        import easyocr
        _reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        logger.info("EasyOCR model loaded.")
    return _reader


def preprocess_for_math(image: Image.Image) -> Image.Image:
    """
    Enhance image contrast and sharpness for better math OCR accuracy.
    Works even without opencv — uses PIL only.
    """
    try:
        import cv2
        img = np.array(image)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Boost contrast
        gray = cv2.convertScaleAbs(gray, alpha=1.8, beta=30)
        # Sharpen
        gray = cv2.GaussianBlur(gray, (1, 1), 0)
        # Otsu threshold → clean black/white
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Scale up 2x — helps EasyOCR read small fonts
        h, w = thresh.shape
        thresh = cv2.resize(thresh, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        return Image.fromarray(thresh)
    except ImportError:
        # opencv not available — fallback to PIL enhancement
        from PIL import ImageEnhance, ImageFilter
        image = ImageEnhance.Contrast(image).enhance(2.0)
        image = ImageEnhance.Sharpness(image).enhance(2.0)
        # Scale up 2x
        w, h = image.size
        image = image.resize((w * 2, h * 2), Image.LANCZOS)
        return image


def extract_text_from_image(image_bytes: bytes, filename: str = "image.jpg") -> dict:
    """
    Extract text from image using EasyOCR with math preprocessing.

    Returns:
        {
            "extracted_text": str,
            "confidence": float,
            "lines": list,
            "method": str,
            "needs_review": bool
        }
    """
    try:
        # Load and preprocess image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = preprocess_for_math(image)
        img_array = np.array(image)

        reader = get_reader()

        # Run OCR — paragraph=True groups nearby text better
        results = reader.readtext(
            img_array,
            detail=1,
            paragraph=False,
            contrast_ths=0.1,
            adjust_contrast=0.5
        )

        if not results:
            return {
                "extracted_text": "",
                "confidence": 0.0,
                "lines": [],
                "method": "easyocr",
                "needs_review": True,
                "error": "No text detected in image. Please try a clearer image or type the problem."
            }

        # Build lines list
        lines = []
        for (bbox, text, conf) in results:
            cleaned = text.strip()
            if cleaned:
                lines.append({
                    "text": cleaned,
                    "confidence": round(float(conf), 3)
                })

        if not lines:
            return {
                "extracted_text": "",
                "confidence": 0.0,
                "lines": [],
                "method": "easyocr",
                "needs_review": True,
                "error": "No readable text found."
            }

        # Join text in reading order (top to bottom)
        extracted_text = " ".join([l["text"] for l in lines]).strip()
        avg_confidence = sum(l["confidence"] for l in lines) / len(lines)

        # Always flag math images for review — OCR is imperfect for math
        needs_review = avg_confidence < 0.80 or len(extracted_text) < 5

        return {
            "extracted_text": extracted_text,
            "confidence": round(avg_confidence, 3),
            "lines": lines,
            "method": "easyocr",
            "needs_review": needs_review
        }

    except ImportError:
        return {
            "extracted_text": "",
            "confidence": 0.0,
            "lines": [],
            "method": "error",
            "needs_review": True,
            "error": "EasyOCR not installed. Run: pip install easyocr"
        }
    except Exception as e:
        logger.error(f"EasyOCR error: {e}")
        return {
            "extracted_text": "",
            "confidence": 0.0,
            "lines": [],
            "method": "error",
            "needs_review": True,
            "error": str(e)
        }


def check_ocr_health() -> dict:
    """Check if EasyOCR is available."""
    try:
        import easyocr
        return {"status": "ok", "method": "easyocr"}
    except ImportError:
        return {"status": "error", "error": "easyocr not installed"}