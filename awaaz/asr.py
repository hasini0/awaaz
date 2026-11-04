"""Speech-to-text. The model is chosen per language via config/settings.yaml (asr.overrides).

Flow for the default `hf` backend:
1. decode any audio file to 16 kHz mono with ffmpeg
2. if the caller gave no language, identify it with Whisper, *restricted to the languages Awaaz serves*
   (unrestricted Whisper often hears Telugu as Tamil or Kannada)
3. transcribe with that language's model and the language token forced, so Telugu speech comes back
   in Telugu script instead of being spelled out in Devanagari
"""
from __future__ import annotations

import subprocess
import time
from functools import lru_cache

import numpy as np

from . import config

LANGUAGES = ("te", "hi", "ur", "en")
SAMPLE_RATE = 16000
# Older fine-tuned Whisper checkpoints ship a minimal generation config without the language table;
# they share the tokenizer of the base multilingual model, so its generation config is borrowed.
BASE_GENERATION_CONFIG = "openai/whisper-small"


class AudioError(ValueError):
    """The audio could not be decoded or transcribed (bad file, no ffmpeg, unsupported language)."""


def _load_audio(path: str) -> np.ndarray:
    """Decode any audio file to 16 kHz mono float32 with ffmpeg (no torchaudio needed)."""
    try:
        raw = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", path, "-f", "f32le", "-ac", "1",
                              "-ar", str(SAMPLE_RATE), "-"], capture_output=True, check=True).stdout
    except FileNotFoundError as e:
        raise AudioError("ffmpeg is not installed, so audio cannot be decoded.") from e
    except subprocess.CalledProcessError as e:
        raise AudioError("Could not read this audio file. Try wav, mp3, m4a or ogg.") from e
    audio = np.frombuffer(raw, dtype=np.float32)
    if len(audio) < SAMPLE_RATE * 0.3:
        raise AudioError("The recording is too short. Please describe the problem for a few seconds.")
    return audio


def _model_for(language: str | None) -> tuple[str, str]:
    cfg = config.settings()["asr"]
    over = (cfg.get("overrides") or {}).get(language or "", {})
    return over.get("backend", cfg["backend"]), over.get("model", cfg["model"])


# ------------------------------------------------------------------------- HF Whisper
@lru_cache
def _whisper(model_id: str):
    from transformers import GenerationConfig, WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id).eval()
    if not getattr(model.generation_config, "lang_to_id", None):
        model.generation_config = GenerationConfig.from_pretrained(BASE_GENERATION_CONFIG)
    return processor, model


def _features(processor, audio: np.ndarray):
    # truncation=False + attention mask lets Whisper transcribe recordings longer than 30 s
    return processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", truncation=False,
                     padding="longest", return_attention_mask=True)


def detect_language(audio: np.ndarray, model_id: str | None = None) -> tuple[str, float]:
    """Most likely of LANGUAGES and its probability (Whisper's language token, restricted to our set)."""
    import torch

    processor, model = _whisper(model_id or _model_for(None)[1])
    feats = processor(audio[:30 * SAMPLE_RATE], sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
    gc = model.generation_config
    token_ids = [gc.lang_to_id[f"<|{lang}|>"] for lang in LANGUAGES]
    with torch.no_grad():
        logits = model(input_features=feats, decoder_input_ids=torch.tensor([[gc.decoder_start_token_id]])).logits
    probs = logits[0, -1, token_ids].softmax(-1)
    best = int(probs.argmax())
    return LANGUAGES[best], float(probs[best])


def _transcribe_hf(audio: np.ndarray, model_id: str, language: str) -> str:
    import torch

    processor, model = _whisper(model_id)
    inputs = _features(processor, audio)
    with torch.no_grad():
        ids = model.generate(inputs.input_features, attention_mask=inputs.attention_mask, language=language,
                             task="transcribe")
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


# ------------------------------------------------------------------ faster-whisper
@lru_cache
def _faster_whisper(size: str):
    from faster_whisper import WhisperModel

    return WhisperModel(size, device="cpu", compute_type="int8")


def _transcribe_faster_whisper(path: str, model: str, language: str | None) -> tuple[str, str, float, float]:
    segments, info = _faster_whisper(model).transcribe(path, language=language, beam_size=5, vad_filter=True,
                                                       task="transcribe")
    text = " ".join(s.text.strip() for s in segments).strip()
    return text, info.language, info.language_probability, info.duration


# --------------------------------------------------------------------------- public
def preload():
    """Load every configured Whisper model now (call it in a background thread at start-up) so the first
    voice note doesn't wait ~30 s for model loading."""
    cfg = config.settings()["asr"]
    if cfg["backend"] != "hf":
        return
    for model_id in {cfg["model"], *(o.get("model", cfg["model"]) for o in (cfg.get("overrides") or {}).values())}:
        _whisper(model_id)


def transcribe(audio_path: str, language: str | None = None) -> dict:
    """Returns {text, language, language_prob, audio_seconds, latency_s, backend}.
    `language` is a hint (te/hi/ur/en); without it the language is detected from the audio."""
    if language is not None and language not in LANGUAGES:
        raise AudioError(f"Unsupported language {language!r}; use one of {', '.join(LANGUAGES)}.")
    t0 = time.perf_counter()
    backend, _ = _model_for(language)

    if backend == "faster_whisper":
        text, lang, prob, seconds = _transcribe_faster_whisper(audio_path, _model_for(language)[1], language)
        return {"text": text, "language": lang, "language_prob": round(prob, 3), "audio_seconds": round(seconds, 2),
                "latency_s": time.perf_counter() - t0, "backend": "faster-whisper"}

    audio = _load_audio(audio_path)
    prob = None
    if language is None:
        language, prob = detect_language(audio)
    model_id = _model_for(language)[1]
    return {"text": _transcribe_hf(audio, model_id, language), "language": language,
            "language_prob": round(prob, 3) if prob is not None else None,
            "audio_seconds": round(len(audio) / SAMPLE_RATE, 2), "latency_s": time.perf_counter() - t0,
            "backend": model_id.split("/")[-1]}
