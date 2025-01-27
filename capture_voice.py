import sounddevice as sd
import numpy as np
import wave
import scipy.io.wavfile as wav
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import torch
import os
import time
import whisper
from voice_database import VoiceDatabase  # Import VoiceDatabase
import soundfile as sf
import threading
import pymongo  # Add this import at the top of the file
import uuid
from datetime import datetime
import logging

class VoiceCapture:
    def __init__(self):
        # Configure logging first
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s: %(message)s',
            handlers=[
                logging.StreamHandler(),  # Output to console
                logging.FileHandler('voice_capture.log', mode='a')  # Append to log file
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Ensure voice_registrations directory exists
        self.registrations_dir = "voice_registrations"
        os.makedirs(self.registrations_dir, exist_ok=True)

        # Audio recording parameters
        self.sample_rate = 16000
        self.channels = 1
        self.dtype = 'float32'

        # Initialize audio buffer
        self.audio_buffer = []

        # Initialize voice database
        self.voice_db = VoiceDatabase()
        
        # Load pre-trained Wav2Vec model for embedding
        try:
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
            self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
             
            # Load Whisper for transcription verification
            self.whisper_model = whisper.load_model("base")
            
            self.logger.info("Wav2Vec2 and Whisper models loaded successfully!")
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            raise
        
        # Predefined verification sentences
        self.verification_sentences = [
            "hello how are you",
            "what are you doing",
            "having any plans for today"
        ]

        # Add MongoDB connection
        try:
            # Use environment variable or default local connection
            mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
            self.mongo_client = pymongo.MongoClient(mongo_uri)
            
            # Select or create database
            self.db = self.mongo_client['voice_recognition_db']
            
            # Create or get voice_embedding collection
            self.voice_collection = self.db['voice_embedding']
            
            # Create index for efficient querying
            self.voice_collection.create_index([('name', pymongo.ASCENDING)])
            
            self.logger.info("MongoDB connection established successfully")
        
        except Exception as e:
            self.logger.error(f"MongoDB connection error: {e}")
            raise

    def verify_transcription(self, audio_path, expected_text):
        """
        Verify if the spoken text matches the expected text
        """
        try:
            # Transcribe audio
            result = self.whisper_model.transcribe(audio_path, fp16=False)
            transcribed_text = result['text'].lower().strip()
            expected_text = expected_text.lower().strip()

            # Calculate similarity
            similarity = self._calculate_text_similarity(transcribed_text, expected_text)
            
            print(f"\n--- Transcription Verification ---")
            print(f"Expected:    {expected_text}")
            print(f"Transcribed: {transcribed_text}")
            print(f"Similarity:  {similarity:.2%}")

            # More lenient similarity threshold
            return similarity > 0.7  # 70% similarity threshold

        except Exception as e:
            print(f"Transcription verification error: {e}")
            return False

    def _calculate_text_similarity(self, text1, text2):
        """
        Calculate text similarity using Levenshtein distance
        """
        from difflib import SequenceMatcher
        
        # Normalize texts
        text1 = ''.join(c.lower() for c in text1 if c.isalnum() or c.isspace())
        text2 = ''.join(c.lower() for c in text2 if c.isalnum() or c.isspace())
        
        return SequenceMatcher(None, text1, text2).ratio()

    def capture_verified_voice(self, name):
        """
        Capture and verify voice recording with guided interaction
        
        Args:
            name (str): Name of the person being recorded
        
        Returns:
            numpy.ndarray or None: Voice embedding or None if capture fails
        """
        try:
            # Sanitize filename
            safe_name = "".join(x for x in name if x.isalnum() or x in [' ', '_']).rstrip()
            
            # Generate filename
            filename = f"{safe_name}_voice.wav"
            filepath = os.path.join(self.registrations_dir, filename)

            # Predefined verification sentences
            verification_sentences = [
                "hello how are you",
                "what are you doing",
                "having any plans for today"
            ]

            # Voice recording with verification
            final_recordings = []

            print("\n--- Voice Registration Process ---")
            print(f"Registering voice for: {name}")
            input("\nPress Enter to start the guided voice registration process...")

            for i, sentence in enumerate(verification_sentences, 1):
                attempts = 0
                verification_successful = False

                while attempts < 3 and not verification_successful:
                    print(f"\nSentence {i}: Please speak the following sentence clearly:")
                    print(f"'{sentence}'")
                    input("Press Enter when you are ready to record...")

                    print("Recording started. Speak now...")
                    
                    # Record audio
                    recording = sd.rec(
                        int(5 * self.sample_rate),  # 5 seconds recording
                        samplerate=self.sample_rate, 
                        channels=self.channels,
                        dtype=self.dtype
                    )
                    sd.wait()  # Wait until recording is complete

                    # Temporary file for this recording
                    temp_filepath = f"temp_verification_{i}.wav"
                    sf.write(temp_filepath, recording, self.sample_rate)

                    # Verify transcription
                    verification_result = self.verify_transcription(temp_filepath, sentence)

                    if verification_result:
                        print("✅ Verification successful!")
                        final_recordings.append(recording)
                        verification_successful = True
                    else:
                        attempts += 1
                        print(f"❌ Verification failed. Attempt {attempts}/3")
                        
                        if attempts < 3:
                            print("Please try speaking the sentence again more clearly.")
                        else:
                            print("Maximum attempts reached. Skipping this sentence.")

                    # Clean up temporary file
                    os.unlink(temp_filepath)

                if not verification_successful:
                    print(f"Could not verify sentence {i}. Continuing with available recordings.")

            # Consolidate recordings if at least one is successful
            if final_recordings:
                consolidated_recording = np.concatenate(final_recordings)
                
                # Save consolidated recording
                sf.write(filepath, consolidated_recording, self.sample_rate)
                print(f"\n✅ Voice recording saved to {filepath}")

                # Extract embedding and save automatically
                embedding = self.extract_embedding(filepath, name)
                
                return embedding
            else:
                print("❌ Voice registration failed. No valid recordings.")
                return None

        except Exception as e:
            print(f"Voice capture error: {e}")
            return None

    def record_audio(self, filename, duration=5):
        """
        Record audio and save to file
        """
        print("Recording... Speak now!")
        
        # Record audio
        recording = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.int16
        )
        sd.wait()

        try:
            # Save the recording
            self.save_audio_chunk(recording, filename)
            print(f"Recording saved to {filename}")
            return filename
            
        except Exception as e:
            print(f"Error saving recording: {e}")
            return None

    def save_audio_chunk(self, recording, filename):
        """
        Save audio chunk to WAV file
        """
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(recording.tobytes())

    def extract_embedding(self, audio_path, name=None):
        """
        Extract voice embedding using Wav2Vec2
        
        Args:
            audio_path (str): Path to the audio file
            name (str, optional): Name to associate with the embedding
        
        Returns:
            numpy.ndarray or None: Voice embedding
        """
        try:
            # Load audio file using scipy
            sample_rate, audio_input = wav.read(audio_path)
            
            # Convert to float and normalize
            if audio_input.dtype == np.int16:
                audio_input = audio_input.astype(np.float32) / 32768.0
            
            # Ensure mono audio
            if audio_input.ndim > 1:
                audio_input = audio_input.mean(axis=1)
            
            # Resample if necessary
            if sample_rate != self.sample_rate:
                from scipy.signal import resample
                audio_input = resample(audio_input, int(len(audio_input) * self.sample_rate / sample_rate))
            
            # Process audio
            inputs = self.feature_extractor(audio_input, sampling_rate=self.sample_rate, return_tensors="pt")
            
            # Extract features
            with torch.no_grad():
                outputs = self.model(inputs.input_values)
            
            # Average pooling of the last hidden states
            embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze().numpy()
            
            # Automatically save embedding with the provided name
            if name:
                self.save_embedding_to_mongodb(embedding, name)
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Error extracting embedding: {e}")
            print(f"Embedding extraction error: {e}")
            return None

    def save_embedding_to_mongodb(self, embedding, name):
        """
        Save voice embedding to MongoDB
        
        Args:
            embedding (numpy.ndarray): Voice embedding to save
            name (str): Name for the voice embedding
        
        Returns:
            bool: True if successfully saved, False otherwise
        """
        try:
            # Validate embedding and name
            if embedding is None:
                self.logger.warning("No embedding provided to save")
                return False

            if not name or name.strip() == '':
                self.logger.warning("Invalid name for embedding")
                return False

            # Prepare embedding document
            embedding_doc = {
                '_id': str(uuid.uuid4()),  # Unique identifier
                'name': name,
                'embedding': embedding.tolist(),  # Convert to list for MongoDB storage
                'timestamp': datetime.utcnow()
            }

            # Insert into MongoDB
            result = self.voice_collection.insert_one(embedding_doc)
            
            self.logger.info(f"Voice embedding for {name} saved successfully")
            print(f"Voice embedding for {name} saved to database")
            
            return True

        except pymongo.errors.PyMongoError as e:
            self.logger.error(f"MongoDB insertion error: {e}")
            print(f"Error saving voice embedding: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error saving embedding: {e}")
            print(f"Unexpected error: {e}")
            return False

    def save_audio_buffer(self, chunk):
        """
        Save audio chunk to buffer
        """
        try:
            # Ensure chunk is in the correct format
            if isinstance(chunk, (bytes, np.ndarray)):
                # Convert to int16 if needed
                if isinstance(chunk, bytes):
                    chunk = np.frombuffer(chunk, dtype=np.int16)
                elif chunk.dtype != np.int16:
                    chunk = chunk.astype(np.int16)
                
                # Append to buffer
                self.audio_buffer.append(chunk)
                return True
            return False
        except Exception as e:
            print(f"Error saving audio chunk: {e}")
            return False

    def get_audio_data(self):
        """
        Get concatenated audio data from buffer
        """
        if self.audio_buffer:
            return np.concatenate(self.audio_buffer)
        return np.array([])

    def clear_buffer(self):
        """
        Clear the audio buffer
        """
        self.audio_buffer = []

def main():
    voice_capture = VoiceCapture()
    name = input("Enter name for voice registration: ")
    voice_capture.capture_verified_voice(name)

if __name__ == "__main__":
    main() 