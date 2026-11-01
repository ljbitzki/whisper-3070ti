"""Interface local de transcrição para um whisper.cpp CUDA existente.

O backend não usa modelos Python: controla um único whisper-server (127.0.0.1)
e converte arquivos com ffmpeg. Não expor portas em rede sem autenticação.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import subprocess
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

HOME = Path.home()
WHISPER_DIR = Path(os.environ.get("WHISPER_DIR", HOME / "IA/whisper.cpp")).expanduser().resolve()
MODELS_DIR = WHISPER_DIR / "models"
SERVER_BIN = WHISPER_DIR / "build-cuda/bin/whisper-server"
APP_DIR = Path(__file__).resolve().parent
SERVER_HOST = "127.0.0.1"
SERVER_PORT = int(os.environ.get("WHISPER_SERVER_PORT", "18080"))
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
LOG_PATH = APP_DIR / "whisper-server.log"
MAX_UPLOAD = 300 * 1024 * 1024
ALLOWED_EXT = {".mp3", ".mp4", ".wav", ".m4a", ".ogg", ".opus", ".webm", ".flac", ".aac", ".wma", ".mkv"}
LANGS = {"pt", "auto", "en", "es", "fr", "de", "it"}

proc: subprocess.Popen | None = None
log_file = None
active_model: str | None = None
model_lock = asyncio.Lock()


def available_models() -> dict[str, Path]:
    # Os caminhos são sempre derivados do diretório local; jamais aceitamos caminhos arbitrários.
    if not MODELS_DIR.is_dir():
        return {}
    result = {}
    for path in sorted(MODELS_DIR.glob("ggml-*.bin")):
        if path.is_file() and not path.name.endswith(".en.bin"):
            result[path.name] = path.resolve()
    return result


def preferred_model(models: dict[str, Path]) -> str:
    for candidate in ("ggml-small.bin", "ggml-large-v3-turbo.bin", "ggml-medium.bin"):
        if candidate in models:
            return candidate
    if models:
        return next(iter(models))
    raise RuntimeError(f"Nenhum modelo multilíngue encontrado em {MODELS_DIR}")


async def wait_server(timeout_seconds: float = 120) -> None:
    end = asyncio.get_running_loop().time() + timeout_seconds
    async with httpx.AsyncClient(timeout=2.0, trust_env=False) as client:
        while asyncio.get_running_loop().time() < end:
            if proc is not None and proc.poll() is not None:
                raise RuntimeError(f"whisper-server encerrou (código {proc.returncode}). Veja {LOG_PATH}")
            try:
                res = await client.get(f"{SERVER_URL}/health")
                if res.status_code == 200 and res.json().get("status") == "ok":
                    return
            except (httpx.HTTPError, ValueError):
                pass
            await asyncio.sleep(0.4)
    raise RuntimeError(f"Timeout inicializando whisper-server. Veja {LOG_PATH}")


@asynccontextmanager
async def lifespan(_: FastAPI):
    global proc, log_file, active_model
    if not SERVER_BIN.is_file():
        raise RuntimeError(f"Não encontrei {SERVER_BIN}. Compile: cmake --build build-cuda --target whisper-server -j 4")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg não encontrado: sudo apt install ffmpeg")
    models = available_models()
    selected = preferred_model(models)
    log_file = LOG_PATH.open("a", encoding="utf-8")
    cmd = [str(SERVER_BIN), "--host", SERVER_HOST, "--port", str(SERVER_PORT),
           "-m", str(models[selected]), "-l", "pt", "-t", "4"]
    proc = subprocess.Popen(cmd, cwd=str(WHISPER_DIR), stdout=log_file,
                            stderr=subprocess.STDOUT, start_new_session=True)
    try:
        await wait_server()
        active_model = selected
        print(f"[Whisper Web] CUDA pronto, modelo {selected}, API {SERVER_URL}", flush=True)
        yield
    finally:
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                await asyncio.to_thread(proc.wait, 10)
            except subprocess.TimeoutExpired:
                proc.kill()
                await asyncio.to_thread(proc.wait)
        proc = None
        active_model = None
        if log_file:
            log_file.close()
            log_file = None


app = FastAPI(title="Whisper Web Local", lifespan=lifespan, docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")


@app.get("/")
async def home():
    return FileResponse(APP_DIR / "static/index.html")


@app.get("/api/status")
async def status():
    return {
        "models": list(available_models()),
        "active_model": active_model,
        "gpu_backend": "whisper.cpp CUDA (build-cuda)",
        "online": proc is not None and proc.poll() is None,
    }


def validate_parameters(model: str, language: str, temperature: float,
                        beam_size: int, best_of: int, prompt: str,
                        no_speech_thold: float) -> Path:
    models = available_models()
    if model not in models:
        raise HTTPException(422, "Modelo indisponível ou não multilíngue")
    if language not in LANGS:
        raise HTTPException(422, "Idioma não permitido")
    if not 0 <= temperature <= 1:
        raise HTTPException(422, "Temperatura deve estar entre 0 e 1")
    if not -1 <= beam_size <= 8:
        raise HTTPException(422, "Beam size deve estar entre -1 e 8")
    if not 1 <= best_of <= 8:
        raise HTTPException(422, "Best of deve estar entre 1 e 8")
    if not 0 <= no_speech_thold <= 1:
        raise HTTPException(422, "Limiar de silêncio deve estar entre 0 e 1")
    if len(prompt) > 1000:
        raise HTTPException(422, "Prompt deve ter no máximo 1.000 caracteres")
    return models[model]


def wav_conversion(source: Path, destination: Path) -> None:
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
           "-y", "-i", str(source), "-map", "0:a:0", "-vn", "-ac", "1",
           "-ar", "16000", "-c:a", "pcm_s16le", str(destination)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(408, "Conversão FFmpeg excedeu 10 minutos") from exc
    if result.returncode != 0 or not destination.is_file() or destination.stat().st_size < 44:
        raise HTTPException(422, f"Não foi possível converter o áudio: {result.stderr[-450:]}")


def srt_timestamp(seconds: float, separator: str = ",") -> str:
    milliseconds = max(0, round(float(seconds) * 1000))
    h, r = divmod(milliseconds, 3600000)
    m, r = divmod(r, 60000)
    s, ms = divmod(r, 1000)
    return f"{h:02}:{m:02}:{s:02}{separator}{ms:03}"


def make_subtitles(segments: list[dict], kind: str) -> str:
    blocks = []
    if kind == "vtt":
        blocks.append("WEBVTT")
    for i, seg in enumerate(segments, start=1):
        start = srt_timestamp(seg.get("start", 0), "." if kind == "vtt" else ",")
        end = srt_timestamp(seg.get("end", 0), "." if kind == "vtt" else ",")
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        prefix = f"{i}\n" if kind == "srt" else ""
        blocks.append(f"{prefix}{start} --> {end}\n{text}")
    return "\n\n".join(blocks) + "\n"


async def write_upload_limited(upload: UploadFile, target: Path):
    size = 0
    with target.open("wb") as out:
        while chunk := await upload.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD:
                raise HTTPException(413, "Arquivo maior que o limite de 300 MB")
            out.write(chunk)
    if not size:
        raise HTTPException(400, "Arquivo vazio")


@app.post("/api/transcribe")
async def transcribe(
    file: Annotated[UploadFile, File()],
    model: Annotated[str, Form()],
    language: Annotated[str, Form()] = "pt",
    temperature: Annotated[float, Form()] = 0,
    beam_size: Annotated[int, Form()] = -1,
    best_of: Annotated[int, Form()] = 2,
    no_speech_thold: Annotated[float, Form()] = 0.6,
    prompt: Annotated[str, Form()] = "",
):
    global active_model
    model_path = validate_parameters(model, language, temperature, beam_size, best_of,
                                     prompt, no_speech_thold)
    extension = Path(file.filename or "").suffix.lower()
    if extension not in ALLOWED_EXT:
        raise HTTPException(415, "Formato não aceito. Use MP3, MP4, WAV, M4A, OGG, Opus, WebM ou FLAC")
    with tempfile.TemporaryDirectory(prefix="whisper-web-") as scratch:
        tempdir = Path(scratch)
        source = tempdir / f"entrada{extension}"
        wav = tempdir / "audio.wav"
        try:
            await write_upload_limited(file, source)
        finally:
            await file.close()
        await asyncio.to_thread(wav_conversion, source, wav)
        # Inclui troca de modelo e transcrição no mesmo bloqueio. Não se misturam modelos.
        async with model_lock:
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(900.0, connect=10.0), trust_env=False) as client:
                    if model != active_model:
                        load = await client.post(f"{SERVER_URL}/load", files={"model": (None, str(model_path))})
                        load.raise_for_status()
                        active_model = model
                    with wav.open("rb") as input_stream:
                        result = await client.post(
                            f"{SERVER_URL}/inference",
                            data={
                                "language": language,
                                "response_format": "verbose_json",
                                "temperature": str(temperature),
                                "beam_size": str(beam_size),
                                "best_of": str(best_of),
                                "no_speech_thold": str(no_speech_thold),
                                "prompt": prompt,
                            },
                            files={"file": ("audio.wav", input_stream, "audio/wav")},
                        )
                    result.raise_for_status()
                    payload = result.json()
            except (httpx.HTTPError, ValueError) as exc:
                active_model = None  # próxima requisição tentará recarregar o modelo
                raise HTTPException(502, f"Falha na inferência do whisper-server: {str(exc)[:400]}") from exc
    segments = [{"start": float(s.get("start", 0)), "end": float(s.get("end", 0)),
                 "text": str(s.get("text", "")).strip()} for s in payload.get("segments", [])]
    return {"text": str(payload.get("text", "")).strip(),
            "segments": segments, "srt": make_subtitles(segments, "srt"),
            "vtt": make_subtitles(segments, "vtt"), "model": model}
