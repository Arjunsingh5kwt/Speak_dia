import os
import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper
import torch
import logging
from pyannote.audio import Pipeline
from pyannote.core import Annotation
import pymongo
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RealTimeDiarization:
    def __init__(self):
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # MongoDB Connection
        try:
            mongo_uri = os.getenv('MONGO_URI', 'mongodb+srv://arjuns5kwt:arjun5kwt@menuversa.szl9h.mongodb.net/')
            self.mongo_client = pymongo.MongoClient(mongo_uri)
            self.db = self.mongo_client['voice_recognition_db']
            self.speakers_collection = self.db['voice_embedding']
        except Exception as e: 
            self.logger.error(f"MongoDB connection error: {e}")
            raise

        # Load Whisper model
        try:
            self.whisper_model = whisper.load_model('small')
        except Exception as e:
            self.logger.error(f"Whisper model loading error: {e}")
            raise

        # Load Pyannote Diarization Pipeline
        try:
            # Note: You might need to use a HuggingFace token for some models
            self.diarization_pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization')
        except Exception as e:
            self.logger.error(f"Diarization pipeline loading error: {e}")
            raise

    def process_audio_stream(self, audio_data, sample_rate=16000):
        """
        Process audio stream for real-time transcription and diarization
        
        Args:
            audio_data (numpy.ndarray): Audio time series data
            sample_rate (int): Sampling rate of audio
        
        Returns:
            dict: Transcription and diarization results
        """
        try:
            # Temporary file for processing
            temp_wav_path = "temp_stream.wav"
            sf.write(temp_wav_path, audio_data, sample_rate)

            # Transcription
            transcription = self.whisper_model.transcribe(
                temp_wav_path, 
                fp16=torch.cuda.is_available()
            )

            # Diarization
            diarization = self.diarization_pipeline(temp_wav_path)

            # Match speakers from database
            speaker_mapping = self._match_speakers(diarization)

            # Clean up temporary file
            os.unlink(temp_wav_path)

            return {
                'transcription': transcription['text'],
                'diarization': speaker_mapping,
                'segments': transcription.get('segments', [])
            }

        except Exception as e:
            self.logger.error(f"Audio processing error: {e}")
            return None

    def _match_speakers(self, diarization):
        """
        Match diarization labels to speaker names from database
        
        Args:
            diarization (Annotation): Pyannote diarization result
        
        Returns:
            dict: Mapped speaker names and their segments
        """
        try:
            # Fetch all registered speakers
            registered_speakers = list(self.speakers_collection.find())

            # Create mapping dictionary
            speaker_mapping = {}

            for turn, _, speaker in diarization.itertracks(yield_label=True):
                # Basic mapping strategy
                best_match = self._find_best_speaker_match(speaker, registered_speakers)
                
                speaker_mapping[speaker] = {
                    'name': best_match,
                    'start': turn.start,
                    'end': turn.end
                }

            return speaker_mapping

        except Exception as e:
            self.logger.error(f"Speaker matching error: {e}")
            return {}

    def _find_best_speaker_match(self, diarization_label, registered_speakers):
        """
        Find best speaker match using basic string comparison
        
        Args:
            diarization_label (str): Diarization speaker label
            registered_speakers (list): List of registered speakers
        
        Returns:
            str: Best matched speaker name
        """
        # Simple matching strategy
        # In a real-world scenario, you'd use more advanced matching techniques
        default_name = f"Unknown Speaker {diarization_label}"
        
        if not registered_speakers:
            return default_name

        # Basic matching - you can enhance this with fuzzy matching or ML techniques
        for speaker in registered_speakers:
            if speaker.get('name'):
                return speaker['name']

        return default_name

    def real_time_processing(self, duration=10, sample_rate=16000):
        """
        Real-time audio processing with continuous streaming
        
        Args:
            duration (int): Duration of audio chunk (in seconds)
            sample_rate (int): Sampling rate
        """
        def audio_callback(indata, frames, time, status):
            if status:
                self.logger.warning(f"Audio stream status: {status}")
            
            # Process audio chunk
            result = self.process_audio_stream(indata.flatten(), sample_rate)
            
            if result:
                # Print real-time results
                print("\n--- Real-Time Diarization ---")
                print(f"Transcription: {result['transcription']}")
                
                for speaker, details in result.get('diarization', {}).items():
                    print(f"Speaker {speaker}: {details['name']}")
                    print(f"Segment: {details['start']:.2f}s - {details['end']:.2f}s")

        # Start audio stream
        with sd.InputStream(
            samplerate=sample_rate, 
            channels=1, 
            callback=audio_callback,
            blocksize=int(sample_rate * duration)
        ):
            sd.sleep(float('inf'))  # Continuous processing

def main():
    """
    Demonstration of Real-Time Diarization
    """
    try:
        diarization_system = RealTimeDiarization()
        print("Starting Real-Time Speaker Diarization...")
        print("Speak into the microphone. Press Ctrl+C to stop.")
        
        diarization_system.real_time_processing()
    
    except KeyboardInterrupt:
        print("\nStopping real-time diarization.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 