"""
CosyVoice3 local TTS server with OpenAI-compatible /v1/audio/speech endpoint.

Wraps CosyVoice3 native inference behind the same API that LiveTalking's
omnitts plugin expects, so `--tts omnitts --TTS_SERVER http://127.0.0.1:8091`
works with local voice cloning.

Usage:
    python cosyvoice_api_server.py --port 8091 \
        --model_dir D:/model_cache/modelscope/FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
        --voices_dir voices/

Voice setup:
    Place reference audio files in the voices/ directory:
        voices/my_voice.wav    (10-20s clear speech, 16kHz+ mono)
        voices/my_voice.txt    (transcript of the audio)
    Then use  voice: "my_voice"  in config.yaml (REF_FILE).
"""

import argparse
import io
import os
import re
import sys
import json
import struct
import logging
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Generator

from fastapi import FastAPI, Request, File, Form, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

_SAFE_NAME_RE = re.compile(r'^[A-Za-z0-9_-]{1,64}$')

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "third_party", "Matcha-TTS"))

from cosyvoice.cli.cosyvoice import AutoModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cosyvoice_api")


def load_wav(wav_path: str, target_sr: int) -> torch.Tensor:
    """Load a WAV file and resample to target_sr. Returns [1, T] float32 tensor."""
    speech, sr = sf.read(wav_path, dtype="float32")
    if len(speech.shape) > 1:
        speech = speech[:, 0]
    if sr != target_sr:
        import librosa
        speech = librosa.resample(speech, orig_sr=sr, target_sr=target_sr)
    return torch.from_numpy(speech).unsqueeze(0)

app = FastAPI(title="CosyVoice3 API Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
voices: dict[str, dict] = {}
sample_rate = 24000
voices_dir_path: Path = None


def _register_voice_file(wav_path: Path) -> bool:
    """Load a single wav+txt pair into the live `voices` dict. Returns True on success."""
    name = wav_path.stem
    txt_path = wav_path.with_suffix(".txt")
    if not txt_path.exists():
        logger.warning(f"Skipping {wav_path.name}: no matching .txt file")
        return False

    prompt_text = txt_path.read_text(encoding="utf-8").strip()
    if not prompt_text:
        logger.warning(f"Skipping {wav_path.name}: .txt file is empty")
        return False
    if "<|endofprompt|>" not in prompt_text:
        prompt_text = prompt_text + "<|endofprompt|>"

    prompt_wav = load_wav(str(wav_path), 16000)
    voices[name] = {
        "prompt_text": prompt_text,
        "prompt_wav": prompt_wav,
    }
    logger.info(f"Loaded voice '{name}' from {wav_path.name} ({prompt_text[:30]}...)")
    return True


def load_voices(voices_dir: str):
    """Load all reference voices from the voices directory."""
    vdir = Path(voices_dir)
    if not vdir.exists():
        vdir.mkdir(parents=True)
        logger.info(f"Created voices directory: {vdir}")
        return

    for wav_path in vdir.glob("*.wav"):
        _register_voice_file(wav_path)

    if not voices:
        logger.warning(f"No voices loaded from {vdir}. Place .wav + .txt pairs there.")


def generate_pcm(model_output: Generator) -> Generator[bytes, None, None]:
    """Convert CosyVoice output to streaming PCM16 bytes."""
    for chunk in model_output:
        audio = chunk["tts_speech"].numpy()
        pcm = (audio * 32767).astype(np.int16).tobytes()
        yield pcm


def generate_wav(model_output: Generator) -> Generator[bytes, None, None]:
    """Convert CosyVoice output to WAV (header + PCM16 data)."""
    buf = io.BytesIO()
    first = True
    total_frames = 0

    for chunk in model_output:
        audio = chunk["tts_speech"].numpy().flatten()
        pcm = (audio * 32767).astype(np.int16)
        total_frames += len(pcm)

        if first:
            sf.write(buf, pcm, sample_rate, subtype="PCM_16", format="WAV")
            buf.seek(0)
            yield buf.read()
            buf = io.BytesIO()
            first = False
        else:
            yield pcm.tobytes()


@app.post("/v1/audio/speech")
async def speech(request: Request):
    """OpenAI-compatible TTS endpoint."""
    body = await request.json()

    text = body.get("input", "")
    voice_name = body.get("voice", "default")
    speed = float(body.get("speed", 1.0))
    response_format = body.get("response_format", "wav")
    stream = body.get("stream", True)

    if not text:
        return {"error": "input text is required"}

    if voice_name not in voices:
        available = list(voices.keys())
        return {
            "error": f"Voice '{voice_name}' not found. Available: {available}"
        }

    voice = voices[voice_name]
    logger.info(f"TTS request: voice={voice_name}, text={text[:60]}...")

    model_output = model.inference_zero_shot(
        tts_text=text,
        prompt_text=voice["prompt_text"],
        prompt_wav=voice["prompt_wav"],
        stream=True,
        speed=speed,
    )

    media_type = "audio/wav" if response_format == "wav" else "audio/pcm"
    gen = generate_wav(model_output) if response_format == "wav" else generate_pcm(model_output)

    return StreamingResponse(gen, media_type=media_type)


@app.get("/v1/voices")
async def list_voices():
    """List available voices."""
    return {
        "voices": [
            {"name": k, "prompt_text": v["prompt_text"][:50]}
            for k, v in voices.items()
        ]
    }


@app.post("/v1/voices")
async def register_voice(
    audio_file: UploadFile = File(...),
    name: str = Form(...),
    transcript: str = Form(...),
):
    """Register a new reference voice at runtime — no model reload needed."""
    if not _SAFE_NAME_RE.match(name):
        return {"error": "name must match ^[A-Za-z0-9_-]{1,64}$"}
    if not transcript.strip():
        return {"error": "transcript is required"}

    wav_path = voices_dir_path / f"{name}.wav"
    txt_path = voices_dir_path / f"{name}.txt"

    audio_bytes = await audio_file.read()
    audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    sf.write(str(wav_path), audio, sr, subtype="PCM_16")
    txt_path.write_text(transcript.strip(), encoding="utf-8")

    ok = _register_voice_file(wav_path)
    if not ok:
        return {"error": f"failed to register voice '{name}'"}

    return {
        "voices": [
            {"name": k, "prompt_text": v["prompt_text"][:50]}
            for k, v in voices.items()
        ]
    }


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None, "voices": list(voices.keys())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CosyVoice3 API Server")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument(
        "--model_dir",
        type=str,
        default="FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        help="Local path or ModelScope repo ID",
    )
    parser.add_argument(
        "--voices_dir",
        type=str,
        default="voices",
        help="Directory with reference voice .wav + .txt pairs",
    )
    args = parser.parse_args()

    logger.info(f"Loading CosyVoice3 model from {args.model_dir}...")
    os.environ.setdefault("HF_HOME", "D:/model_cache/huggingface")
    if not os.path.exists(args.model_dir):
        from huggingface_hub import snapshot_download
        args.model_dir = snapshot_download(args.model_dir, cache_dir="D:/model_cache/huggingface")
        logger.info(f"Resolved model path: {args.model_dir}")
    model = AutoModel(model_dir=args.model_dir, fp16=True)
    sample_rate = model.sample_rate
    logger.info(f"Model loaded. Sample rate: {sample_rate}")

    voices_dir_path = Path(args.voices_dir)
    load_voices(args.voices_dir)

    logger.info(f"Starting server on port {args.port}...")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
