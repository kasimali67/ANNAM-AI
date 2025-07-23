import os, sys, subprocess, venv
from pathlib import Path

# Constants
PROJECT = Path("fastapi-whisper")
APP_DIR = PROJECT / "app"
VENV_DIR = PROJECT / ".venv"

def run(cmd, cwd=None):
    print(f"Running: {' '.join(map(str, cmd))} (cwd={cwd or os.getcwd()})")
    subprocess.check_call(cmd, cwd=cwd)

def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"Written: {path}")

def main():
    # Step 1: Create project and app directories
    APP_DIR.mkdir(parents=True, exist_ok=True)
    (APP_DIR / "__init__.py").touch(exist_ok=True)

    # Step 2: Create virtual environment
    if not VENV_DIR.exists():
        print("Creating virtual environment...")
        venv.create(str(VENV_DIR), with_pip=True, clear=True)

    pip = VENV_DIR / ("Scripts/pip.exe" if os.name == "nt" else "bin/pip")
    python = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    # Step 3: Write requirements.txt
    REQS = """\
fastapi==0.110.0
uvicorn[standard]==0.30.1
torch==2.3.0
torchaudio==2.3.0
soundfile==0.12.1
transformers==4.43.1
huggingface_hub[hf_transfer]==0.23.3
onnxruntime==1.22.1
numpy==1.26.4
python-multipart==0.0.9
""".strip()
    write(PROJECT / "requirements.txt", REQS)

    # Step 4: Install requirements
    print("Upgrading pip...")
    run([python, "-m", "pip", "install", "--upgrade", "pip"])
    print("Installing requirements...")
    run([str(pip), "install", "-r", "requirements.txt"], cwd=str(PROJECT))

    # Step 5: Write settings.py
    SETTINGS = """\
WHISPER_MODEL = "openai/whisper-base"
INDIC_MODEL = "ai4bharat/indic-conformer-600m-multilingual"
CACHE_DIR = None
"""
    write(APP_DIR / "settings.py", SETTINGS)

    # Step 6: Write main.py
    MAIN = '''\
from fastapi import FastAPI, UploadFile, File, HTTPException
import torch, torchaudio, tempfile, os, re, soundfile
from transformers import WhisperForConditionalGeneration, WhisperProcessor, AutoModel
from . import settings

app = FastAPI(title="Whisper-Base Transcription API")
device = "cuda" if torch.cuda.is_available() else "cpu"

_w_model = _w_proc = _indic = None
LANG_RE = re.compile(r"<\\|([a-z]{2})\\|>")

def load_models():
    global _w_model, _w_proc, _indic
    if _w_model is None:
        _w_model = WhisperForConditionalGeneration.from_pretrained(
            settings.WHISPER_MODEL, cache_dir=settings.CACHE_DIR
        ).to(device)
        _w_proc = WhisperProcessor.from_pretrained(
            settings.WHISPER_MODEL, cache_dir=settings.CACHE_DIR
        )
        _indic = AutoModel.from_pretrained(
            settings.INDIC_MODEL, trust_remote_code=True, cache_dir=settings.CACHE_DIR
        )

def detect_language(wav_path: str):
    wav, sr = torchaudio.load(wav_path, format="wav")
    wav = torchaudio.functional.resample(wav, sr, 16_000)
    inputs = _w_proc(wav.squeeze(), sampling_rate=16_000, return_tensors="pt").to(device)
    start_id = _w_proc.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    dec_in = torch.tensor([[start_id]], device=device)
    ids = _w_model.generate(inputs["input_features"], decoder_input_ids=dec_in, max_new_tokens=2)[0]
    token_str = _w_proc.tokenizer.decode(ids, skip_special_tokens=False)
    lang = LANG_RE.findall(token_str)[-1]
    return lang, wav.to(device)

@app.get("/")
async def root():
    return {"msg": "Whisper-Base API online", "docs": "/docs"}

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    load_models()
    if not (audio.content_type and audio.content_type.startswith("audio/")):
        raise HTTPException(400, "Upload an audio file")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name
    try:
        lang, wav = detect_language(tmp_path)
        ctc = _indic(wav, lang, "ctc")
        rnnt = _indic(wav, lang, "rnnt")
        return {"language": lang, "ctc": ctc, "rnnt": rnnt}
    finally:
        os.unlink(tmp_path)
'''.lstrip()
    write(APP_DIR / "main.py", MAIN)

    # Step 7: Write development run script (optional, for Windows PowerShell)
    write(PROJECT / "run_dev.ps1", 'python -m uvicorn app.main:app --reload')

    print("\nSetup complete!\n")
    print(f"To activate the environment and run the API (on Windows):")
    print(f"cd {PROJECT}")
    print(r".\.venv\Scripts\activate")
    print("uvicorn app.main:app --reload\n")
    print(f"On Linux/Mac:")
    print(f"cd {PROJECT}")
    print(r"source .venv/bin/activate")
    print("uvicorn app.main:app --reload\n")
    print("Or, to run directly (no activation):")
    print(f"{python} -m uvicorn app.main:app --reload")
    print("\nYou can now develop your FastAPI app in 'fastapi-whisper/app/main.py'.")

if __name__ == "__main__":
    main()

