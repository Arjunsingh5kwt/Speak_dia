import whisper
import langdetect
import logging
import os
import soundfile as sf
import numpy as np
import re
import torch
import subprocess
import tempfile
import wave
import contextlib
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from advanced_speaker_recognition import AdvancedSpeakerRecognition
from voice_database import VoiceDatabase
import uuid
import pymongo
from datetime import datetime

class RealTimeTranscription:
    def __init__(self, model_size='base'):
        """
        Initialize Real-Time Transcription with Advanced Speaker Recognition
        
        Args:
            model_size (str): Size of the Whisper model to load
        """
        # Configure logging
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Load Whisper model
        try:
            self.whisper_model = whisper.load_model(model_size)
            self.logger.info(f"Loaded Whisper {model_size} model successfully")
        except Exception as e:
            self.logger.error(f"Whisper model loading error: {e}")
            raise

        # Load Wav2Vec2 for embedding extraction
        try:
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
            self.wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
            self.logger.info("Loaded Wav2Vec2 model successfully")
        except Exception as e:
            self.logger.error(f"Wav2Vec2 model loading error: {e}")
            raise

        # Initialize Advanced Speaker Recognition
        self.speaker_recognition = AdvancedSpeakerRecognition()
        
        # Initialize Voice Database
        self.voice_db = VoiceDatabase()

    def _convert_audio(self, input_path):
        """
        Convert audio to a compatible WAV format using multiple methods
        """
        try:
            self.logger.info(f"Attempting to convert audio: {input_path}")
            
            # Create a temporary file for converted audio
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_output:
                output_path = temp_output.name
            
            # Method 1: FFmpeg conversion
            ffmpeg_command = [
                'ffmpeg', 
                '-i', input_path,  # Input file
                '-acodec', 'pcm_s16le',  # PCM 16-bit little-endian
                '-ar', '16000',  # Sample rate 16kHz
                '-ac', '1',  # Mono channel
                output_path
            ]
            
            self.logger.info(f"FFmpeg Command: {' '.join(ffmpeg_command)}")
            
            # Run FFmpeg conversion
            result = subprocess.run(
                ffmpeg_command, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Check FFmpeg conversion
            if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                self.logger.info(f"Audio converted successfully to: {output_path}")
                return output_path
            
            # Method 2: Manual WAV conversion using soundfile
            try:
                # Read audio file
                audio_data, sample_rate = sf.read(input_path)
                
                # Ensure mono audio
                if audio_data.ndim > 1:
                    audio_data = audio_data.mean(axis=1)
                
                # Resample if necessary
                if sample_rate != 16000:
                    from scipy.signal import resample
                    audio_data = resample(audio_data, int(len(audio_data) * 16000 / sample_rate))
                
                # Write as WAV
                sf.write(output_path, audio_data, 16000)
                
                self.logger.info(f"Audio converted using soundfile: {output_path}")
                return output_path
            
            except Exception as sf_error:
                self.logger.error(f"Soundfile conversion error: {sf_error}")
            
            # Fallback: Return original path
            self.logger.warning("Falling back to original audio path")
            return input_path
        
        except Exception as e:
            self.logger.error(f"Comprehensive audio conversion error: {e}", exc_info=True)
            return input_path

    def _validate_wav_file(self, file_path):
        """
        Validate WAV file integrity
        """
        try:
            with contextlib.closing(wave.open(file_path, 'rb')) as wav_file:
                # Check basic WAV parameters
                n_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                framerate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                
                self.logger.info(f"WAV File Details:")
                self.logger.info(f"Channels: {n_channels}")
                self.logger.info(f"Sample Width: {sample_width}")
                self.logger.info(f"Frame Rate: {framerate}")
                self.logger.info(f"Total Frames: {n_frames}")
                
                return n_frames > 0 and framerate > 0
        except Exception as e:
            self.logger.error(f"WAV file validation error: {e}")
            return False

    def extract_embedding(self, audio_path):
        """
        Extract voice embedding using Wav2Vec2
        
        Args:
            audio_path (str): Path to the audio file
        
        Returns:
            numpy.ndarray or None: Voice embedding
        """
        try:
            # Load audio file
            audio_input, sample_rate = sf.read(audio_path)
            
            # Ensure mono audio
            if audio_input.ndim > 1:
                audio_input = audio_input.mean(axis=1)
            
            # Resample if necessary
            if sample_rate != 16000:
                from scipy.signal import resample
                audio_input = resample(audio_input, int(len(audio_input) * 16000 / sample_rate))
            
            # Normalize audio
            audio_input = audio_input.astype(np.float32)
            audio_input = audio_input / np.max(np.abs(audio_input))
            
            # Process audio
            inputs = self.feature_extractor(audio_input, sampling_rate=16000, return_tensors="pt")
            
            # Extract features
            with torch.no_grad():
                outputs = self.wav2vec_model(inputs.input_values)
            
            # Average pooling of the last hidden states
            embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze().numpy()
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Embedding extraction error: {e}")
            return None

    def identify_speaker(self, embedding):
        """
        Identify speaker using Advanced Speaker Recognition
        
        Args:
            embedding (numpy.ndarray): Voice embedding to match
        
        Returns:
            dict: Speaker identification results
        """
        try:
            # Use Advanced Speaker Recognition to identify speaker
            speaker_match = self.speaker_recognition.identify_speaker(embedding)
            
            return speaker_match
        
        except Exception as e:
            self.logger.error(f"Speaker identification error: {e}")
            return {
                'match': False,
                'name': 'Unknown',
                'confidence': 0.0
            }

    def process_transcription(self, audio_path):
        """
        Process audio transcription with speaker identification
        
        Args:
            audio_path (str): Path to the audio file
        
        Returns:
            dict: Transcription and speaker identification results
        """
        try:
            # Transcribe audio
            transcription_result = self.whisper_model.transcribe(
                audio_path, 
                fp16=False  # Ensure CPU compatibility
            )
            
            # Extract embedding
            embedding = self.extract_embedding(audio_path)
            
            if embedding is None:
                return {
                    'original_text': transcription_result['text'],
                    'detected_language': 'Unknown',
                    'speaker': 'Unknown',
                    'speaker_confidence': 0.0
                }
            
            # Identify speaker
            speaker_match = self.identify_speaker(embedding)
            
            # Prepare result
            result = {
                'original_text': transcription_result['text'],
                'detected_language': transcription_result.get('language', 'Unknown'),
                'speaker': speaker_match['name'],
                'speaker_confidence': speaker_match['confidence']
            }
            
            return result
        
        except Exception as e:
            self.logger.error(f"Transcription processing error: {e}")
            return {
                'original_text': '',
                'detected_language': 'Unknown',
                'speaker': 'Unknown',
                'speaker_confidence': 0.0
            }

    def detect_language(self, text):
        """
        Detect language of the transcribed text
        """
        try:
            # Simplified language mapping focusing on Hindi and English
            language_names = {
                'en': 'English',
                'hi': 'Hindi',
                'ur': 'Hindi'  # Treat Urdu as Hindi for our purposes
            }
            
            # Use langdetect for language detection
            if not text:
                return 'English'
            
            detected_lang = langdetect.detect(text)
            
            # Return only English or Hindi, default to English if not recognized
            return language_names.get(detected_lang, 'English')
        
        except Exception:
            return 'English'  # Default to English on detection failure

    def _language_specific_transcription(self, text, language):
        """
        Perform language-specific transcription and normalization
        """
        try:
            # Normalize text
            text = self._normalize_text(text, language)
            
            # Language-specific processing
            if language == 'Hindi':
                # Ensure text is not empty
                if not text:
                    return ''
                
                try:
                    # Use Hindi-specific model for more accurate transcription
                    hindi_result = self.whisper_model.transcribe(
                        text, 
                        task='transcribe',
                        language='hi',
                        fp16=torch.cuda.is_available()
                    )
                    
                    # Normalize and clean Hindi text
                    normalized_text = self._normalize_hindi_text(hindi_result['text'])
                    
                    return normalized_text
                except Exception as hindi_error:
                    self.logger.error(f"Hindi transcription error: {hindi_error}")
                    # Fallback to original text if Hindi transcription fails
                    return text
            
            # Default to English
            return text
        
        except Exception as e:
            self.logger.error(f"Language-specific transcription error: {e}")
            return text

    def _normalize_text(self, text, language):
        """
        Normalize text based on language
        """
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Language-specific normalization
        if language == 'Hindi':
            # Remove non-Devanagari characters if needed
            text = re.sub(r'[^\u0900-\u097F\s]', '', text)
        elif language == 'English':
            # Remove non-English characters
            text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        return text

    def _normalize_hindi_text(self, text):
        """
        Advanced Hindi text normalization
        """
        try:
            # Remove extra spaces
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Normalize Devanagari characters
            text = text.replace('ॅ', 'े')  # Normalize common character variations
            text = text.replace('ॆ', 'े')
            
            # Remove non-Devanagari characters
            text = re.sub(r'[^\u0900-\u097F\s]', '', text)
            
            # Additional normalization for Urdu-like text
            text = text.replace('ی', 'ी')  # Convert Urdu 'ی' to Hindi 'ी'
            text = text.replace('ے', 'े')  # Convert Urdu 'ے' to Hindi 'े'
            
            return text
        except Exception as e:
            self.logger.error(f"Hindi text normalization error: {e}")
            return text 

def main():
    """
    Demonstration of Real-Time Transcription
    """
    try:
        # Initialize transcription
        transcriber = RealTimeTranscription()
        
        # Prompt for audio file
        audio_path = input("Enter path to audio file for transcription: ").strip()
        
        if os.path.exists(audio_path):
            # Process transcription
            result = transcriber.process_transcription(audio_path)
            
            # Display results
            print("\n--- Transcription Results ---")
            print(f"Text: {result['original_text']}")
            print(f"Language: {result['detected_language']}")
            print(f"Speaker: {result['speaker']} (Confidence: {result['speaker_confidence']:.2%})")
        else:
            print("File does not exist. Please check the path.")
    
    except Exception as e:
        print(f"Transcription error: {e}")

if __name__ == "__main__":
    main() 