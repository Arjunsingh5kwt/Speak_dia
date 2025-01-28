import os
import threading
import queue
import time
import sounddevice as sd
import numpy as np
import whisper
import torch
import re
import tempfile
import uuid
import logging
import soundfile as sf
import tkinter as tk
from tkinter import messagebox, filedialog
import pygame
import io

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from capture_voice import VoiceCapture
from voice_database import VoiceDatabase
from transcription import RealTimeTranscription
from text_to_speech import GoogleTextToSpeech
from audio_analysis import AudioAnalyzer
from advanced_speaker_recognition import AdvancedSpeakerRecognition

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s: %(message)s')

class MicrophonePermissionDialog:
    def request_permission(self):
        """
        Request microphone access permission via console
        """
        while True:
            response = input("Do you want to allow microphone access? (yes/no): ").lower().strip()
            if response in ['yes', 'y']:
                return True
            elif response in ['no', 'n']:
                print("Microphone access denied.")
                return False
            else:
                print("Please enter 'yes' or 'no'.")

class RealTimeSpeakerDiarization:
    def __init__(self, duration=0.5, sample_rate=16000):
        # Request microphone permission
        permission_dialog = MicrophonePermissionDialog()
        if not permission_dialog.request_permission():
            print("Microphone access denied. Exiting.")
            exit()
        
        # Automatically select input device
        self._select_best_input_device()
        
        self.voice_capture = VoiceCapture()
        self.voice_db = VoiceDatabase()
        self.transcriber = RealTimeTranscription()
        
        self.duration = duration  # Reduced duration for more frequent updates
        self.sample_rate = sample_rate
        self.channels = 1
        
        # Audio recording parameters
        self.block_size = int(self.sample_rate * self.duration)
        
        # Temporary file management
        self.temp_dir = tempfile.mkdtemp()
        
        # Prevent multiple simultaneous processing
        self.processing_lock = threading.Lock()
        
        # Audio processing queue
        self.audio_queue = queue.Queue()
        self.processing_thread = threading.Thread(target=self._process_audio_queue)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        # Add Advanced Speaker Recognition
        self.speaker_recognition = AdvancedSpeakerRecognition()

    def _select_best_input_device(self):
        """ 
        Automatically select the best input device
        Prioritizes devices with more input channels
        """
        devices = sd.query_devices()
        input_devices = [
            i for i, device in enumerate(devices) 
            if device['max_input_channels'] > 0
        ]
        
        if not input_devices:
            raise RuntimeError("No input devices found")
        
        # Select device with most input channels
        best_device = max(
            input_devices, 
            key=lambda i: devices[i]['max_input_channels']
        )
        
        # Set default input device
        sd.default.device[0] = best_device
        print(f"Auto-selected input device: {devices[best_device]['name']}")

    def _process_audio_queue(self):
        """
        Continuously process audio chunks from the queue
        """
        while True:
            try:
                audio_path = self.audio_queue.get()
                self._process_single_chunk(audio_path)
            except Exception as e:
                logging.error(f"Queue processing error: {e}")

    def _process_single_chunk(self, audio_path):
        """
        Process a single audio chunk
        """
        try:
            # Validate audio file
            if not os.path.exists(audio_path):
                return
            
            # Check audio file content
            try:
                audio_data, sample_rate = sf.read(audio_path)
            except Exception:
                return
            
            # Ensure audio data is not empty
            if audio_data is None or len(audio_data) == 0:
                return
            
            # Minimum audio length check
            min_duration = 0.5
            min_samples = int(min_duration * sample_rate)
            if len(audio_data) < min_samples:
                return
            
            # Ensure mono audio
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)
            
            # Normalize audio
            audio_data = audio_data.astype(np.float32)
            audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Extract voice embedding
            embedding = self.voice_capture.extract_embedding(audio_path)
            
            if embedding is None:
                print("Failed to extract embedding")
                return

            # Identify speaker using Advanced Speaker Recognition
            speaker_match = self.speaker_recognition.identify_speaker(embedding)
            
            # Transcribe audio
            transcription_result = self.transcriber.process_transcription(audio_path)
            
            # Extract transcribed text
            transcribed_text = transcription_result['original_text'].strip()
            detected_language = transcription_result['detected_language']
            
            # Remove numbers and extra whitespace
            transcribed_text = re.sub(r'\d+', '', transcribed_text)
            transcribed_text = ' '.join(transcribed_text.split())
            
            # Print results only if there's meaningful text
            if len(transcribed_text) > 1:
                # Display speaker and transcription
                if speaker_match['match']:
                    print(f"Speaker: {speaker_match['name']} (Confidence: {speaker_match['confidence']:.2%})")
                else:
                    print("Speaker: Unknown")
                
                print(f"Language: {detected_language}") 
                print(f"Transcription: {transcribed_text}\n")
            
            # Clean up temporary file
            os.unlink(audio_path)
        
        except Exception as e:
            logging.error(f"Error processing audio chunk: {e}")

    def start_real_time_diarization(self):
        """Start real-time speaker diarization"""
        print("Starting Real-Time Speaker Diarization...")
        print("Speak into the microphone. Press Ctrl+C to stop.")
        
        try:
            # Use a generator-based approach for continuous recording
            with sd.InputStream(
                samplerate=self.sample_rate, 
                channels=self.channels,
                dtype=np.int16,
                blocksize=self.block_size,
                callback=self.audio_callback
            ):
                # Keep the main thread running
                while True:
                    time.sleep(0.1)
        
        except KeyboardInterrupt:
            print("\nStopping Real-Time Speaker Diarization...")
        except Exception as e:
            print(f"Unexpected error: {e}")

    def audio_callback(self, indata, frames, time_info, status):
        """
        Robust audio callback with extensive error handling
        """
        # Validate input data
        if indata is None or len(indata) == 0:
            return
        
        # Ensure we're not processing multiple chunks simultaneously
        if not self.processing_lock.acquire(blocking=False):
            return
        
        try:
            # Generate unique filename
            temp_filename = os.path.join(
                self.temp_dir, 
                f"temp_chunk_{uuid.uuid4().hex}.wav"
            )
            
            # Save chunk to a temporary file
            self.voice_capture.save_audio_chunk(indata, temp_filename)
            
            # Add to processing queue
            self.audio_queue.put(temp_filename)
        
        except Exception as e:
            logging.error(f"Audio callback error: {e}")
        
        finally:
            # Release the lock
            self.processing_lock.release()

def audio_analysis_menu():
    """
    Audio Analysis Menu
    """
    audio_analyzer = AudioAnalyzer()
    
    while True:
        print("\n--- Audio Analysis Menu ---")
        print("1. Analyze Audio File")
        print("2. Return to Main Menu")
        
        choice = input("Enter your choice: ").strip()
        
        if choice == '1':
            # Direct file path input
            file_path = input("Enter the full path to the audio file: ").strip()
            
            if os.path.exists(file_path):
                try:
                    print("\nAnalyzing audio file...")
                    analysis_results = audio_analyzer.analyze_audio(file_path)
                    
                    # Display Results
                    print("\n--- Analysis Results ---")
                    
                    # Transcription
                    print("\nTranscription:")
                    print(analysis_results['transcription'].get('text', 'No transcription'))
                    
                    # Language
                    print("\nLanguage:")
                    print(analysis_results['transcription'].get('language', 'Unknown'))
                    
                    # Speaker Diarization
                    print("\nSpeaker Diarization:")
                    diarization = analysis_results['speaker_diarization']
                    print(f"Total Speakers: {diarization.get('total_speakers', 'N/A')}")
                    
                    # Speaker Identification
                    print("\nSpeaker Identification:")
                    speaker_ids = analysis_results['speaker_identification']
                    for speaker, name in speaker_ids.items():
                        print(f"Speaker {speaker}: {name}")
                    
                    # Generate Summary
                    print("\nSummary:")
                    summary = audio_analyzer.generate_summary(analysis_results)
                    print(summary)
                
                except Exception as e:
                    print(f"Analysis error: {e}")
            else:
                print("File does not exist. Please check the path.")
        
        elif choice == '2':
            break
        
        else:
            print("Invalid choice. Please try again.")

def play_audio(audio_stream):
    if audio_stream is None:
        print("No audio stream to play.")
        return

    # Initialize pygame mixer
    pygame.mixer.init()

    # Save the audio stream to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmpfile:
        tmpfile.write(audio_stream.read())
        tmpfile_path = tmpfile.name

    print(f"Playing audio from {tmpfile_path}")  # Debug statement

    # Load and play the audio file
    pygame.mixer.music.load(tmpfile_path)
    pygame.mixer.music.play()

    # Wait for playback to finish
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    # Clean up the temporary file after playback is done
    pygame.mixer.music.unload()
    os.remove(tmpfile_path)

def main():
    try:
        # Read credentials from environment variables
        subscription_key = os.getenv('AZURE_SPEECH_KEY')
        region = os.getenv('AZURE_SPEECH_REGION')
        endpoint = os.getenv('AZURE_SPEECH_ENDPOINT')  # Optional
        
        # MongoDB URI handling
        mongo_uri = os.getenv('MONGO_URI', 'mongodb+srv://arjuns5kwt:arjun5kwt@menuversa.szl9h.mongodb.net/')
        
        # If environment variables are not set, raise an error
        if not subscription_key:
            raise ValueError("Azure Speech Key is not set in the .env file")
        
        if not region:
            raise ValueError("Azure Speech Region is not set in the .env file")
        
        if not mongo_uri:
            raise ValueError("MongoDB URI is not set in the .env file")
        
        # Initialize Text-to-Speech
        tts_service = GoogleTextToSpeech()
        
        # Initialize components
        voice_capture = VoiceCapture()
        voice_db = VoiceDatabase()  # This will use the provided MONGO_URI
        
        while True:
            print("\n1. Capture and Save Voice")
            print("2. Real-Time Speaker Diarization")
            print("3. Text to Speech")
            print("4. Register User Voice")
            print("5. Audio Analysis")
            print("6. Exit")
            choice = input("Enter your choice: ")

            if choice == '1':
                # Capture voice
                name = input("Enter name for voice registration: ")
                voice_capture = VoiceCapture()
                embedding = voice_capture.capture_verified_voice(name)
                
                if embedding is not None:
                    # Save to database
                    voice_db.save_voice(name, embedding)
                    print(f"Voice for {name} saved successfully in speaker recognition database!")
                else:
                    print("Voice capture failed. Please try again.")

            elif choice == '2':
                # Real-Time Speaker Diarization
                diarization = RealTimeSpeakerDiarization()
                diarization.start_real_time_diarization()

            elif choice == '3':
                while True:
                    text = input("Please enter the text you want to convert to speech (or type 'q' to return to main menu): ")
                    if text.lower() in ['q', 'quit']:
                        break
                    audio_stream = tts_service.synthesize_speech(text)
                    if audio_stream:
                        play_audio(audio_stream)
                    else:
                        print("Failed to synthesize speech.")

            elif choice == '4':
                # Register User Voice
                registered_voice = tts_service.select_voice()
                if registered_voice:
                    print(f"Successfully selected voice: {registered_voice}")

            elif choice == '5':
                # New Audio Analysis Menu
                audio_analysis_menu()

            elif choice == '6':
                print("Exiting the program.")
                break
            else:
                print("Invalid choice. Try again.")
    
    except Exception as e:
        print(f"Initialization error: {e}")
        print("Please check your .env configuration.")
        return

if __name__ == "__main__":
    main()