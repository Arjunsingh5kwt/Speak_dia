import os
import io
import uuid
import base64
import numpy as np
import torch
import soundfile as sf
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import uvicorn
import logging

# Configure global logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import project-specific modules
from advanced_speaker_recognition import AdvancedSpeakerRecognition
from transcription import RealTimeTranscription
from voice_database import VoiceDatabase
from capture_voice import VoiceCapture
from text_to_speech import GoogleTextToSpeech

# Create FastAPI app
app = FastAPI(
    title="Advanced Voice Processing API",
    description="Comprehensive Voice Recognition and Processing API",
    version="1.0.0"
)

# Robust service initialization with error handling
def initialize_services():
    services = {}
    
    try:
        services['speaker_recognition'] = AdvancedSpeakerRecognition()
    except Exception as e:
        logger.error(f"Speaker Recognition init error: {e}")
        services['speaker_recognition'] = None

    try:
        services['transcription'] = RealTimeTranscription()
    except Exception as e:
        logger.error(f"Transcription init error: {e}")
        services['transcription'] = None

    try:
        services['voice_capture'] = VoiceCapture()
    except Exception as e:
        logger.error(f"Voice Capture init error: {e}")
        services['voice_capture'] = None

    try:
        services['voice_database'] = VoiceDatabase()
    except Exception as e:
        logger.error(f"Voice Database init error: {e}")
        services['voice_database'] = None

    try:
        services['tts_generator'] = GoogleTextToSpeech()
    except Exception as e:
        logger.error(f"TTS Generator init error: {e}")
        services['tts_generator'] = None

    return services

# Initialize services
SERVICES = initialize_services()

# Pydantic Models for Request Validation
class SpeakerEnrollmentRequest(BaseModel):
    name: str
    audio_base64: str

class TranscriptionRequest(BaseModel):
    audio_base64: str
    language: str = 'auto'

class TextToSpeechRequest(BaseModel):
    text: str

# Speaker Recognition Endpoints
@app.post("/speaker/enroll")
async def enroll_speaker(request: SpeakerEnrollmentRequest):
    """
    Enroll a new speaker in the voice database
    
    Endpoint: http://localhost:8000/speaker/enroll
    Method: POST
    Request Body:
    {
        "name": "John Doe",
        "audio_base64": "base64_encoded_audio_data"
    }
    """
    try:
        # Decode base64 audio
        audio_data = np.frombuffer(
            base64.b64decode(request.audio_base64), 
            dtype=np.float32
        )
        
        # Extract embedding
        embedding = SERVICES['speaker_recognition'].extract_embedding(audio_data)
        
        # Add to database
        result = SERVICES['speaker_recognition'].add_new_speaker(
            request.name, 
            embedding
        )
        
        return {
            "success": result, 
            "message": "Speaker enrolled successfully" if result else "Enrollment failed"
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/speaker/identify")
async def identify_speaker(file: UploadFile = File(...)):
    """
    Identify a speaker from an audio file
    
    Endpoint: http://localhost:8000/speaker/identify
    Method: POST
    Request Body: Multipart/form-data with audio file
    """
    try:
        # Read audio file
        audio_data, sample_rate = sf.read(file.file)
        
        # Normalize audio
        audio_data = audio_data.astype(np.float32)
        audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Extract embedding
        embedding = SERVICES['speaker_recognition'].extract_embedding(audio_data)
        
        # Identify speaker
        result = SERVICES['speaker_recognition'].identify_speaker(embedding)
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Transcription Endpoints
@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribe an audio file
    
    Endpoint: http://localhost:8000/transcribe
    Method: POST
    Request Body: Multipart/form-data with audio file
    """
    try:
        # Save uploaded file temporarily
        temp_path = f"/tmp/{uuid.uuid4()}.wav"
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Transcribe
        result = SERVICES['transcription'].process_transcription(temp_path)
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Text-to-Speech Endpoints
@app.post("/tts/generate")
async def generate_speech(request: TextToSpeechRequest, background_tasks: BackgroundTasks):
    """
    Generate speech from text
    
    Endpoint: http://localhost:8000/tts/generate
    Method: POST
    Request Body:
    {
        "text": "Hello, how are you?",
        "voice": "default",
        "language": "en"
    }
    """
    try:
        # Generate audio file
        output_path = SERVICES['tts_generator'].synthesize_speech(
            request.text
        )
        
        # Optional: Add cleanup task
        background_tasks.add_task(os.unlink, output_path)
        
        return FileResponse(
            output_path, 
            media_type="audio/wav", 
            filename="generated_speech.wav"
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Voice Database Management Endpoints
@app.get("/voice-database/list")
async def list_speakers():
    """
    List all enrolled speakers
    
    Endpoint: http://localhost:8000/voice-database/list
    Method: GET
    """
    try:
        speakers = SERVICES['voice_database'].get_all_speakers()
        return {"speakers": speakers}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/voice-database/remove/{speaker_name}")
async def remove_speaker(speaker_name: str):
    """
    Remove a speaker from the database
    
    Endpoint: http://localhost:8000/voice-database/remove/{speaker_name}
    Method: DELETE
    """
    try:
        result = SERVICES['voice_database'].remove_speaker(speaker_name)
        return {
            "success": result, 
            "message": f"Speaker {speaker_name} removed" if result else "Removal failed"
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Voice Capture Endpoints
@app.post("/voice/capture")
async def capture_voice(duration: float = 5.0):
    """
    Capture voice for a specified duration
    
    Endpoint: http://localhost:8000/voice/capture
    Method: POST
    Query Param: duration (optional, default=5.0)
    """
    try:
        audio_path = SERVICES['voice_capture'].record_audio(duration)
        return FileResponse(
            audio_path, 
            media_type="audio/wav", 
            filename="captured_voice.wav"
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Health Check Endpoint
@app.get("/health")
async def health_check():
    """
    API Health Check Endpoint
    
    Endpoint: http://localhost:8000/health
    Method: GET
    """
    return {
        "status": "healthy",
        "services": {
            "speaker_recognition": SERVICES['speaker_recognition'] is not None,
            "transcription": SERVICES['transcription'] is not None,
            "text_to_speech": SERVICES['tts_generator'] is not None,
            "voice_database": SERVICES['voice_database'] is not None
        }
    }

# Run the FastAPI application
if __name__ == "__main__":
    uvicorn.run(
        "main_fastapi:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True
    ) 