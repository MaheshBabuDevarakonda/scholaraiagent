# --- ScholarScan Colab Notebook Code ---

# === CODE CELL 0 ===
import subprocess, sys, os

def _sh(cmd):
    subprocess.run(cmd, shell=True, check=False)

_sh("sudo apt-get -qq update")
_sh("sudo apt-get -qq install -y tesseract-ocr tesseract-ocr-eng espeak-ng ffmpeg libespeak1")
_sh("pip install -q rapidocr-onnxruntime opencv-python-headless Pillow numpy matplotlib")
_sh("pip install -q nltk spacy symspellpy wordninja")
_sh("pip install -q sentencepiece sacremoses deep-translator gtts")
_sh("pip install -q 'transformers>=4.33' accelerate torch --index-url https://download.pytorch.org/whl/cpu")
_sh("python -m spacy download en_core_web_sm -q")

print("All packages installed.")

import cv2
import numpy as np
import time
import unicodedata
import re
import warnings

from base64    import b64decode
from pathlib   import Path

import torch
import nltk
import wordninja

from rapidocr_onnxruntime import RapidOCR
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MarianMTModel, MarianTokenizer

warnings.filterwarnings("ignore")
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

#environment
try:
    from IPython.display     import Audio, Javascript, display, clear_output
    from google.colab.output import eval_js
    import matplotlib.pyplot as plt
    USE_COLAB = True
except ImportError:
    USE_COLAB = False

print(f"Environment: {'Colab/Jupyter' if USE_COLAB else 'Local'}")

#NLP tools
try:
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    print("spaCy : OK")
except Exception as _e:
    nlp = None
    print(f"spaCy : not available ({_e})")

try:
    from symspellpy import SymSpell, Verbosity
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    _dict_url = (
        "https://raw.githubusercontent.com/mammothb/symspellpy/master/"
        "symspellpy/frequency_dictionary_en_82_765.txt"
    )
    _dict_path = "/tmp/freq_dict_en.txt"
    if not os.path.exists(_dict_path):
        import urllib.request
        urllib.request.urlretrieve(_dict_url, _dict_path)
    sym_spell.load_dictionary(_dict_path, term_index=0, count_index=1)
    print("SymSpell : OK")
except Exception as _e:
    sym_spell = None
    print(f"SymSpell : not available ({_e})")

# === CODE CELL 1 ===
#RapidOCR
try:
    ocr_engine = RapidOCR(
        det_db_box_thresh=0.4,
        det_db_unclip_ratio=1.8,
        use_angle_cls=True
    )
except Exception:
    ocr_engine = RapidOCR()
print("RapidOCR : OK")

#TRANSLATION MODELS
def _is_online() -> bool:
    try:
        import urllib.request
        urllib.request.urlopen("https://translate.google.com", timeout=4)
        return True
    except Exception:
        return False

_ONLINE = _is_online()
print("Network :", "ONLINE" if _ONLINE else "OFFLINE")

TELUGU_RE = re.compile(r"[\u0C00-\u0C7F]")
ENGLISH_RE = re.compile(r"[a-zA-Z]")

#Primary offline- Helsinki opus-mt-en-mul
_MARIAN_MODEL = "Helsinki-NLP/opus-mt-en-mul"
_MARIAN_CACHE = "/tmp/marian_en_te"
_marian_tok   = None
_marian_model = None
_MARIAN_OK    = False

print("Loading opus-mt-en-mul (offline primary) …")
try:
    _marian_tok   = MarianTokenizer.from_pretrained(_MARIAN_MODEL, cache_dir=_MARIAN_CACHE)
    _marian_model = MarianMTModel.from_pretrained(_MARIAN_MODEL, cache_dir=_MARIAN_CACHE, torch_dtype=torch.float32)
    _marian_model.eval()
    torch.set_num_threads(min(4, torch.get_num_threads()))  # cap threads to avoid overhead
    _MARIAN_OK = True
    print("opus-mt-en-mul : OK")
except Exception as _e:
    print(f"opus-mt-en-mul : FAILED ({_e})")

#Secondary offline- NLLB-200
_NLLB_MODEL     = "facebook/nllb-200-distilled-600M"
_NLLB_CACHE     = "/tmp/nllb_en_te"
_nllb_tokenizer = None
_nllb_model     = None
_NLLB_OK        = False

print("Loading NLLB-200-distilled-600M (offline secondary) …")
try:
    _nllb_tokenizer = AutoTokenizer.from_pretrained(
        _NLLB_MODEL, cache_dir=_NLLB_CACHE, src_lang="eng_Latn"
    )
    _nllb_model = AutoModelForSeq2SeqLM.from_pretrained(_NLLB_MODEL, cache_dir=_NLLB_CACHE, torch_dtype=torch.float32)
    _nllb_model.eval()
    _tgt_id = _nllb_tokenizer.convert_tokens_to_ids("tel_Telu")
    assert _tgt_id != _nllb_tokenizer.unk_token_id, "tel_Telu token not found"
    _NLLB_OK = True
    print("NLLB-200 : OK")
except Exception as _e:
    print(f"NLLB-200 : FAILED ({_e})")

#Tertiary online- GoogleTranslator
_GOOGLE_TRANS_OK = False
try:
    from deep_translator import GoogleTranslator as _GoogleTranslator
    _google_translator = _GoogleTranslator(source="en", target="te")
    _GOOGLE_TRANS_OK = True
    print("GoogleTranslator : OK (online fallback)")
except Exception as _e:
    print(f"GoogleTranslator : not available ({_e})")

if not _MARIAN_OK and not _NLLB_OK and not _GOOGLE_TRANS_OK:
    print("WARNING: ALL translation backends unavailable. Output stays English.")

#TEXT UTILITIES
OCR_SUBS = [
    (re.compile(r"\bl\b"),                     "I"),
    (re.compile(r"\b0\b"),                     "O"),
    (re.compile(r"(?<=[a-zA-Z])0(?=[a-zA-Z])"), "o"),
    (re.compile(r"(?<=[a-zA-Z])1(?=[a-zA-Z])"), "l"),
]

def clean_ocr_text(text: str) -> str:
    for pat, rep in OCR_SUBS:
        text = pat.sub(rep, text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _clean_telugu_output(text: str) -> str:
    text = re.sub(r"[a-zA-Z]+", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.strip(" .,")
    return text.strip()


def _chunk_text(text: str, max_chars: int = 400) -> list:
    raw = [s.strip() for s in re.split(r"[.!?\n]+", text) if s.strip()]
    if not raw:
        raw = [text]
    chunks, buf = [], ""
    for sent in raw:
        if len(buf) + len(sent) + 1 <= max_chars:
            buf = (buf + " " + sent).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = sent
    if buf:
        chunks.append(buf)
    return chunks if chunks else [text]


def _score_translation(text: str) -> float:
    if not text:
        return 0.0
    tel_chars = len(TELUGU_RE.findall(text))
    eng_chars = len(ENGLISH_RE.findall(text))
    return tel_chars - eng_chars * 2

# === CODE CELL 2 ===
#TRANSLATION  (english to telugu)
def _marian_translate(text: str) -> str:
    tagged = f">>te<< {text}"
    enc = _marian_tok([tagged], return_tensors="pt",
                       padding=True, truncation=True, max_length=128)
    with torch.inference_mode():
        out = _marian_model.generate(
            **enc, max_length=192, num_beams=2,
            early_stopping=True, repetition_penalty=1.2,
        )
    return _marian_tok.decode(out[0], skip_special_tokens=True).strip()


def _nllb_translate(text: str) -> str:
    inputs = _nllb_tokenizer(
        text, return_tensors="pt", padding=True,
        truncation=True, max_length=128
    )
    with torch.inference_mode():
        out = _nllb_model.generate(
            **inputs, forced_bos_token_id=_tgt_id,
            max_length=192, num_beams=2, early_stopping=True,
            repetition_penalty=1.2,
        )
    return _nllb_tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()


def _google_translate(text: str) -> str:
    if len(text) <= 4800:
        return _google_translator.translate(text).strip()
    parts, chunk = [], ""
    for w in text.split():
        if len(chunk) + len(w) + 1 > 4800:
            parts.append(_google_translator.translate(chunk).strip())
            chunk = w
        else:
            chunk = (chunk + " " + w).strip()
    if chunk:
        parts.append(_google_translator.translate(chunk).strip())
    return " ".join(parts)


# Cache: avoid re-translating identical OCR text seen recently
_translation_cache: dict = {}
_CACHE_MAX = 50

def translate_to_telugu(english: str) -> str:
    english = clean_ocr_text(english.strip())
    if not english:
        return ""

    # Return cached result instantly for repeated text
    _cache_key = english[:200]
    if _cache_key in _translation_cache:
        cached = _translation_cache[_cache_key]
        print(f"  [cache hit] {cached[:80]}")
        return cached

    chunks  = _chunk_text(english, max_chars=400)
    results = []

    for chunk in chunks:
        best_text  = ""
        best_score = -999.0
        source     = ""

        # 1. Google first: fastest + most accurate when online
        if _GOOGLE_TRANS_OK and _ONLINE:
            try:
                cand = _google_translate(chunk)
                sc   = _score_translation(cand)
                if sc > best_score:
                    best_text, best_score, source = cand, sc, "Google"
            except Exception as e:
                print(f"  [Google error] {e}")

        # 2. Marian: offline fallback (fast, ~300MB)
        if best_score < 3.0 and _MARIAN_OK:
            try:
                cand  = _marian_translate(chunk)
                cand  = _clean_telugu_output(cand)
                sc    = _score_translation(cand)
                if sc > best_score:
                    best_text, best_score, source = cand, sc, "MarianMT"
            except Exception as e:
                print(f"  [MarianMT error] {e}")

        # 3. NLLB: last resort only if both above failed/scored poorly
        if best_score < 3.0 and _NLLB_OK:
            try:
                cand  = _nllb_translate(chunk)
                cand  = _clean_telugu_output(cand)
                sc    = _score_translation(cand)
                if sc > best_score:
                    best_text, best_score, source = cand, sc, "NLLB"
            except Exception as e:
                print(f"  [NLLB error] {e}")

        if best_score < 1.0 or not best_text:
            best_text = "[అనువాదం అందుబాటులో లేదు]"
            source    = "none"

        print(f"  [{source}] {best_text[:100]}")
        results.append(best_text)

    final = (" ".join(r for r in results if r != "[అనువాదం అందుబాటులో లేదు]")
             or "[అనువాదం అందుబాటులో లేదు]")
    if len(_translation_cache) >= _CACHE_MAX:
        _translation_cache.pop(next(iter(_translation_cache)))
    _translation_cache[_cache_key] = final
    return final

#TTS(Telugu)
def _sanitise_for_speech(text: str) -> str:
    text = re.sub(r"\.\.\.", " ... ", text)
    text = re.sub(r"([.!?])\s*", r"\1 ", text)
    text = re.sub(r"[/\\|]", " ", text)
    text = re.sub(r"[-–—]{2,}", ", ", text)
    text = re.sub(r"[-–—]", " ", text)
    text = re.sub(r"[\"\'`\u201C\u201D\u2018\u2019]", "", text)
    text = re.sub(r"[(){}\[\]<>]", " ", text)
    text = re.sub(r"[#@^*~_=+%]", " ", text)
    text = re.sub(r"&", " and ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def speak_telugu(text: str) -> None:
    text = text.strip()
    if not text or text == "[అనువాదం అందుబాటులో లేదు]":
        return
    text = _sanitise_for_speech(text)

    # gTTS (online, natural voice)
    try:
        from gtts import gTTS
        mp3 = "/tmp/tel_tts_out.mp3"
        gTTS(text=text, lang="te", slow=False).save(mp3)
        print(f"  TTS: gTTS Telugu → {mp3}")
        if USE_COLAB:
            display(Audio(mp3, autoplay=True))
        else:
            subprocess.run(["ffplay", "-nodisp", "-autoexit",
                            "-af", "atempo=1.25", mp3],
                           capture_output=True)
        return
    except Exception as _e:
        print(f"  gTTS failed ({_e}), trying espeak-ng …")

    # espeak-ng (offline)
    wav = "/tmp/tel_tts_out.wav"
    for voice in ["te", "te+f3", "te+m3"]:
        r = subprocess.run(
            ["espeak-ng", "-v", voice, "-s", "160", "-p", "50",
             "-a", "175", "-g", "8", "-w", wav, text],
            capture_output=True, text=True
        )
        if r.returncode == 0:
            print(f"  TTS: espeak-ng voice={voice} → {wav}")
            if USE_COLAB:
                display(Audio(wav, autoplay=True))
            else:
                subprocess.run(["ffplay", "-nodisp", "-autoexit", wav],
                               capture_output=True)
            return
    print("  TTS: all voices failed")


# === CODE CELL 3 ===
#PREPROCESSING
def deskew(gray):
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                            threshold=100, minLineLength=100, maxLineGap=10)
    if lines is None:
        return gray
    angles = [np.degrees(np.arctan2(y2 - y1, x2 - x1))
              for x1, y1, x2, y2 in lines[:, 0] if x2 != x1]
    if not angles:
        return gray
    med = np.median(angles)
    if abs(med) > 15:
        return gray
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), med, 1.0)
    return cv2.warpAffine(gray, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def sharpen(gray):
    blurred = cv2.GaussianBlur(gray, (0, 0), 2)
    return cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)


def preprocess_for_ocr(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    orig_h, orig_w = gray.shape
    scale = max(1.0, 2000 / max(orig_h, orig_w))
    if scale > 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)
    gray  = cv2.GaussianBlur(gray, (3, 3), 0)
    gray  = deskew(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)
    gray  = sharpen(gray)
    pad   = 30
    gray  = cv2.copyMakeBorder(gray, pad, pad, pad, pad,
                                cv2.BORDER_CONSTANT, value=255)
    return gray, {"scale": scale, "pad": pad,
                  "orig_w": orig_w, "orig_h": orig_h}

#OCR MULTI-PASS + BOX UTILITIES
def run_single_pass(img):
    try:
        result, _ = ocr_engine(img)
        return result if result else []
    except Exception:
        return []


def box_overlap(b1, b2):
    x11, y11 = min(p[0] for p in b1), min(p[1] for p in b1)
    x12, y12 = max(p[0] for p in b1), max(p[1] for p in b1)
    x21, y21 = min(p[0] for p in b2), min(p[1] for p in b2)
    x22, y22 = max(p[0] for p in b2), max(p[1] for p in b2)
    ix1, iy1 = max(x11, x21), max(y11, y21)
    ix2, iy2 = min(x12, x22), min(y12, y22)
    if ix2 < ix1 or iy2 < iy1:
        return 0
    inter = (ix2 - ix1) * (iy2 - iy1)
    a1 = (x12 - x11) * (y12 - y11)
    a2 = (x22 - x21) * (y22 - y21)
    return inter / float(a1 + a2 - inter)


def merge_results(r1, r2):
    merged = list(r1)
    for item2 in r2:
        box2, word2, conf2 = item2
        matched = False
        for i, (box1, word1, conf1) in enumerate(merged):
            if box_overlap(box1, box2) > 0.5:
                if conf2 > conf1:
                    merged[i] = item2
                matched = True
                break
        if not matched:
            merged.append(item2)
    return merged


def run_ocr_multipass(gray):
    r1 = run_single_pass(gray)
    r2 = run_single_pass(cv2.bitwise_not(gray))
    return merge_results(r1, r2)


def remap_boxes(ocr_result, meta):
    scale, pad = meta["scale"], meta["pad"]
    corrected  = []
    for r in ocr_result:
        if len(r) < 3:
            continue
        box, word, conf = r
        new_box = [
            [max(0, min(meta["orig_w"] - 1, (x - pad) / scale)),
             max(0, min(meta["orig_h"] - 1, (y - pad) / scale))]
            for x, y in box
        ]
        corrected.append([new_box, word, conf])
    return corrected

#SYMBOL/LOGO FILTER  (identical to OCR_real_time.ipynb)
_ALPHA_RE  = re.compile(r"[A-Za-z]")
_ALNUM_RE  = re.compile(r"[A-Za-z0-9]")
_SYMBOL_RE = re.compile(
    r"^[\W_]+$"
    r"|^[©®™℠°•·▲▼◆★☆→←↑↓–—…±×÷]+$"
    r"|^\W{0,2}[0-9]{1,3}\W{0,2}$"
    r"|^[^A-Za-z0-9\s]{1,3}$",
    re.UNICODE
)
_BAD_CATS = {"So", "Sm", "Sk", "Po", "Ps", "Pe", "Pi", "Pf"}


def _unicode_categories(text):
    return {unicodedata.category(c) for c in text if not c.isspace()}


def _box_aspect(box):
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    w  = max(xs) - min(xs)
    h  = max(ys) - min(ys)
    return (w / h) if h > 0 else 999


def is_likely_symbol_or_logo(word, box, conf):
    stripped = word.strip()
    if not stripped:
        return True
    if _SYMBOL_RE.search(stripped):
        return True
    if len(_ALNUM_RE.findall(stripped)) / max(len(stripped), 1) < 0.30:
        return True
    cats = _unicode_categories(stripped)
    if cats and cats.issubset(_BAD_CATS):
        return True
    if len(stripped) <= 2 and _box_aspect(box) < 1.8:
        return True
    if len(stripped) == 1 and conf < 0.60:
        return True
    if not _ALPHA_RE.search(stripped) and len(stripped) <= 2:
        return True
    return False


def filter_symbols(ocr_result):
    kept, removed = [], []
    for r in ocr_result:
        if len(r) < 3:
            continue
        box, word, conf = r
        if is_likely_symbol_or_logo(word, box, conf):
            removed.append(word)
        else:
            kept.append(r)
    if removed:
        print(f"  [symbol-filter] removed: {removed}")
    return kept

#SMART READING ORDER
def _box_left(box):    return min(p[0] for p in box)
def _box_top(box):     return min(p[1] for p in box)
def _box_height(box):
    ys = [p[1] for p in box]
    return max(ys) - min(ys)
def _baseline_y(box):
    ys = sorted(p[1] for p in box)
    return sum(ys[-2:]) / 2


def smart_reading_order(ocr_result):
    if not ocr_result:
        return ocr_result

    heights = [_box_height(r[0]) for r in ocr_result]
    med_h   = sorted(heights)[len(heights) // 2] if heights else 20
    med_h   = max(med_h, 8)
    v_gap   = 0.9 * med_h

    tokens  = sorted(ocr_result, key=lambda r: (_baseline_y(r[0]), _box_left(r[0])))
    used    = [False] * len(tokens)
    lines   = []

    for i, tok in enumerate(tokens):
        if used[i]:
            continue
        line   = [tok]
        used[i] = True
        cy_i   = _baseline_y(tok[0])

        for j in range(i + 1, len(tokens)):
            if used[j]:
                continue
            cy_j = _baseline_y(tokens[j][0])
            if abs(cy_j - cy_i) > v_gap:
                break
            line.append(tokens[j])
            used[j] = True
            cy_i = sum(_baseline_y(r[0]) for r in line) / len(line)

        line.sort(key=lambda r: _box_left(r[0]))
        lines.append(line)

    lines.sort(key=lambda ln: (_box_top(ln[0][0]), _box_left(ln[0][0])))
    return [tok for line in lines for tok in line]

#ANNOTATION + DISPLAY
def annotate_image(img_bgr, ocr_result):
    out = img_bgr.copy()
    for r in ocr_result:
        if len(r) < 3:
            continue
        box, word, conf = r
        pts   = [(int(x), int(y)) for x, y in box]
        color = (0, 255, 0) if conf > 0.8 else (0, 200, 255)
        for i in range(4):
            cv2.line(out, pts[i], pts[(i + 1) % 4], color, 2)
        cv2.putText(out, f"{word} ({conf:.0%})",
                    (pts[0][0], max(20, pts[0][1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return out


def show_image(img, title="OCR Result"):
    if not USE_COLAB:
        cv2.imshow(title, img)
        cv2.waitKey(1500)
        return
    import matplotlib.pyplot as plt
    plt.figure(figsize=(16, 9))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title, fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

#FRAME PIPELINE
def process_frame(img_bgr: np.ndarray, frame_num: int) -> str:
    print(f"\n{'='*65}")
    print(f"  FRAME #{frame_num}  |  {img_bgr.shape[1]}x{img_bgr.shape[0]}  |  {time.strftime('%H:%M:%S')}")
    print(f"{'='*65}")
    t0 = time.time()

    #OCR
    print("Pre-processing and OCR …")
    ocr_ready, meta = preprocess_for_ocr(img_bgr)
    raw = run_ocr_multipass(ocr_ready)

    if not raw:
        print("  No text detected. Try again.")
        return ""

    filtered = [r for r in raw if len(r) >= 3 and r[2] > 0.35]
    filtered = remap_boxes(filtered, meta)
    filtered = filter_symbols(filtered)

    if not filtered:
        print("  Only symbols/logos detected – nothing to translate.")
        return ""

    filtered = smart_reading_order(filtered)

    english = clean_ocr_text(" ".join(r[1] for r in filtered))
    print(f"\n  English  ({time.time()-t0:.1f}s):\n  {english}\n")

    #Annotate
    annotated = annotate_image(img_bgr, filtered)
    show_image(annotated, f"Frame #{frame_num} – OCR Boxes")

    #Translate to Telugu
    print("Translating to Telugu …")
    t1     = time.time()
    telugu = translate_to_telugu(english)
    print(f"\n  తెలుగు  ({time.time()-t1:.1f}s):\n  {telugu}\n")

    #TTS
    print("Speaking Telugu …")
    speak_telugu(telugu)

    print(f"  Done in {time.time()-t0:.1f}s total")
    print("─" * 55)
    print("  SPACE = capture next  |  Q = quit")
    return telugu

# === CODE CELL 4 ===
#CAMERA JS
SPACEBAR_JS = """
async function spaceCapture() {
    const wrapper = document.createElement('div');
    wrapper.tabIndex = 0;
    wrapper.style.cssText =
        'position:fixed;top:0;left:0;width:100%;height:100%;'
        +'background:#111;display:flex;flex-direction:column;'
        +'align-items:center;justify-content:center;z-index:9999;outline:none;';

    const video = document.createElement('video');
    video.style.cssText =
        'width:640px;height:360px;object-fit:cover;'
        +'border:3px solid #0f0;border-radius:8px;background:#000;';
    video.autoplay   = true;
    video.playsInline = true;

    const hint = document.createElement('p');
    hint.style.cssText =
        'color:#0f0;font-size:22px;font-family:monospace;margin-top:12px;';
    hint.textContent = '⎵ SPACE = Capture   |   Q = Quit';

    const badge = document.createElement('p');
    badge.style.cssText =
        'color:#29b6f6;font-size:15px;font-family:monospace;margin-top:4px;';
    badge.textContent = 'ENGLISH OCR  →  TELUGU TRANSLATION  →  TTS';

    const status = document.createElement('p');
    status.style.cssText =
        'color:#ff0;font-size:18px;font-family:monospace;margin-top:6px;';
    status.textContent = 'Starting camera …';

    wrapper.appendChild(video);
    wrapper.appendChild(hint);
    wrapper.appendChild(badge);
    wrapper.appendChild(status);
    document.body.appendChild(wrapper);
    wrapper.focus();

    const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment',
                 width: {ideal:1920}, height: {ideal:1080} }
    });
    video.srcObject = stream;

    await new Promise(resolve => { video.onloadedmetadata = resolve; });
    await video.play();

    google.colab.output.setIframeHeight(document.body.scrollHeight, true);

    // warm-up
    const WARMUP_MS = 1500, TICK_MS = 100;
    let elapsed = 0;
    await new Promise(resolve => {
        const ticker = setInterval(() => {
            elapsed += TICK_MS;
            const rem = ((WARMUP_MS - elapsed) / 1000).toFixed(1);
            status.textContent =
                `Camera warming up … ${rem}s  (${video.videoWidth}×${video.videoHeight})`;
            if (elapsed >= WARMUP_MS) { clearInterval(ticker); resolve(); }
        }, TICK_MS);
    });

    status.textContent =
        `Ready (${video.videoWidth}×${video.videoHeight}) – press SPACE to capture`;

    const data = await new Promise(resolve => {
        const handler = (e) => {
            if (e.code === 'Space') {
                e.preventDefault();
                status.textContent = 'Capturing …';
                const canvas   = document.createElement('canvas');
                const settings = stream.getVideoTracks()[0].getSettings();
                canvas.width   = settings.width  || video.videoWidth;
                canvas.height  = settings.height || video.videoHeight;
                canvas.getContext('2d').drawImage(video, 0, 0,
                                                   canvas.width, canvas.height);
                stream.getTracks().forEach(t => t.stop());
                wrapper.removeEventListener('keydown', handler);
                resolve(canvas.toDataURL('image/jpeg', 0.97));
            } else if (e.key === 'q' || e.key === 'Q') {
                stream.getTracks().forEach(t => t.stop());
                wrapper.removeEventListener('keydown', handler);
                resolve('QUIT');
            }
        };
        wrapper.addEventListener('keydown', handler);
    });

    wrapper.remove();
    return data;
}
"""

#WEBCAM LOOP
def _local_webcam_loop(max_frames: int = 100) -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    print("Press SPACE to capture, Q to quit.")
    frame_num   = 0
    all_results = []

    while frame_num < max_frames:
        ret, frame = cap.read()
        if not ret:
            print("Frame read error.")
            break

        cv2.putText(frame,
                    "SPACE = capture  |  Q = quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 220, 255), 2)
        cv2.putText(frame,
                    "English OCR -> Telugu TTS",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 200, 100), 2)
        cv2.imshow("English OCR -> Telugu TTS", frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q") or key == 27:
            print("Session ended.")
            break
        if key == ord(" "):
            frame_num += 1
            result = process_frame(frame.copy(), frame_num)
            if result:
                all_results.append(result)

    cap.release()
    cv2.destroyAllWindows()

    if all_results:
        print("\n  SESSION TRANSLATIONS")
        for i, tel in enumerate(all_results, 1):
            print(f"  [{i:02d}]  {tel}")

#Main function

def realtime_english_to_telugu(max_frames: int = 100) -> None:
    print("=" * 65)
    print("  ENGLISH REAL-TIME OCR  →  TELUGU TRANSLATION  →  TTS")
    print("=" * 65)
    print("  OCR         : RapidOCR multi-pass (normal + inverted)")
    print("  Post-proc   : Symbol filter, smart reading order")

    trans_status = []
    if _MARIAN_OK:        trans_status.append("opus-mt-en-mul >>te<< (offline)")
    if _NLLB_OK:          trans_status.append("NLLB-200 eng→tel (offline)")
    if _GOOGLE_TRANS_OK:  trans_status.append("GoogleTranslator (online)")
    print(f"  Translation : {' → '.join(trans_status) if trans_status else 'NONE AVAILABLE'}")
    print("  TTS         : gTTS Telugu (online) / espeak-ng te (offline)")
    print("=" * 65)
    print("  SPACE = capture frame  |  Q = quit")
    print("=" * 65)

    if USE_COLAB:
        frame, all_results = 0, []
        try:
            while frame < max_frames:
                display(Javascript(SPACEBAR_JS))
                try:
                    data = eval_js("spaceCapture()")
                except KeyboardInterrupt:
                    print("\nCapture interrupted.")
                    break

                if data == "QUIT":
                    print("\nSession ended by user.")
                    break
                if not isinstance(data, str) or not data.startswith("data:image"):
                    print("Bad frame - skipping.")
                    continue

                frame += 1
                binary = b64decode(data.split(",")[1])
                img    = cv2.imdecode(np.frombuffer(binary, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    print("Decode failed - skipping.")
                    continue

                result = process_frame(img, frame)
                if result:
                    all_results.append(result)

                time.sleep(0.3)

        except KeyboardInterrupt:
            print("\nSession interrupted.")

        if all_results:
            print("\n  SESSION TRANSLATIONS")
            for i, tel in enumerate(all_results, 1):
                print(f"  [{i:02d}]  {tel}")
    else:
        _local_webcam_loop(max_frames)

# === CODE CELL 5 ===
realtime_english_to_telugu(max_frames = 100)

