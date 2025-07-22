from fastapi import FastAPI, UploadFile, File, HTTPException
import torchaudio
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoModel
import tempfile
import os
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(
    title="Language-Aware Audio Transcription API",
    description="Detect language and transcribe audio using Whisper + Indic Conformer",
    version="1.0.0"
)

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load models (will happen when first endpoint is called)
lang_id_model = None
lang_id_processor = None
model = None

def load_models():
    global lang_id_model, lang_id_processor, model
    if lang_id_model is None:
        print("Loading models... This may take a while.")
        lang_id_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large").to(device)
        lang_id_processor = WhisperProcessor.from_pretrained("openai/whisper-large")
        model = AutoModel.from_pretrained("ai4bharat/indic-conformer-600m-multilingual", trust_remote_code=True)
        print("Models loaded successfully!")

class TranscriptionResponse(BaseModel):
    detected_language: str
    ctc_transcription: str
    rnnt_transcription: str
    success: bool
    message: str

def detect_language(audio_path):
    waveform, sr = torchaudio.load(audio_path)
    waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
    inputs = lang_id_processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt").to(device)

    start_token_id = lang_id_processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    decoder_input_ids = torch.tensor([[start_token_id]], device=device)

    with torch.no_grad():
        outputs = lang_id_model.generate(
            inputs["input_features"],
            decoder_input_ids=decoder_input_ids,
            max_new_tokens=1,
        )

    lang_token = lang_id_processor.tokenizer.decode(outputs[0], skip_special_tokens=False)
    lang_code = lang_token.replace("<|", "").replace("|>", "").strip()
    return lang_code, waveform.to(device)

@app.get("/")
async def root():
    return {
        "message": "Language-Aware Audio Transcription API", 
        "docs": "/docs",
        "device": device,
        "status": "ready"
    }

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(audio_file: UploadFile = File(...)):
    """Upload an audio file and get language detection + transcription"""
    
    load_models()  # Load models on first use
    
    if not audio_file.content_type or not audio_file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="Please upload an audio file")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        content = await audio_file.read()
        temp_file.write(content)
        temp_audio_path = temp_file.name
    
    try:
        lang_code, wav = detect_language(temp_audio_path)
        transcription_ctc = model(wav, lang_code, "ctc")
        transcription_rnnt = model(wav, lang_code, "rnnt")
        
        os.unlink(temp_audio_path)
        
        return TranscriptionResponse(
            detected_language=lang_code,
            ctc_transcription=transcription_ctc,
            rnnt_transcription=transcription_rnnt,
            success=True,
            message="Transcription completed successfully"
        )
        
    except Exception as e:
        if os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "device": device, "models_loaded": lang_id_model is not None}
