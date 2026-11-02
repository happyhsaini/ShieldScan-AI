"""OCR helper for extracting text from uploaded images using Tesseract."""
import os
import shutil

try:
    import pytesseract
    from PIL import Image, ImageEnhance, ImageFilter

    configured_cmd = os.environ.get("TESSERACT_CMD")
    candidates = [configured_cmd, shutil.which("tesseract")]
    if os.name == "nt":
        candidates.extend([
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
        ])

    TESSERACT_CMD = next(
        (path for path in candidates if path and os.path.isfile(path)), None
    )
    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        # Verify the executable at startup so deployment logs expose a broken install.
        pytesseract.get_tesseract_version()
        OCR_AVAILABLE = True
    else:
        OCR_AVAILABLE = False
except (ImportError, OSError, RuntimeError):
    OCR_AVAILABLE = False
    TESSERACT_CMD = None


def extract_text_from_image(file_storage) -> str:
    """Extract text from an uploaded image or raise a user-facing error."""
    if not OCR_AVAILABLE:
        raise RuntimeError(
            "OCR is unavailable: Tesseract or its Python package is missing. "
            "Check the Render build logs for the Tesseract installation step."
        )

    try:
        img = Image.open(file_storage.stream).convert("RGB")
        img = img.filter(ImageFilter.SHARPEN)
        img = ImageEnhance.Contrast(img).enhance(2.0)
        text = pytesseract.image_to_string(img, config="--oem 3 --psm 6").strip()
        if not text:
            raise RuntimeError(
                "No text could be extracted from the image. "
                "Try a clearer, higher-contrast screenshot."
            )
        return text
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"OCR failed: {exc}") from exc
