import os
import sys
import warnings
import numpy as np
import torch
import torchaudio
# from speechbrain.inference import SpeakerRecognition
from sklearn.metrics.pairwise import cosine_similarity
from voice_database import VoiceDatabase
import shutil
import stat
import requests
import pymongo
from scipy.spatial.distance import cosine
from dotenv import load_dotenv
import logging
from datetime import datetime
import time
import soundfile as sf
from typing import Dict, List, Union
from audio_utils import configure_audio_backend

# Pyannote imports
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.core import Annotation

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Call before class initialization
configure_audio_backend()

def set_file_permissions(file_path):
    """
    Set full permissions for a file
    """
    try:
        # Attempt to modify file permissions
        os.chmod(file_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    except Exception as e:
        print(f"Permission setting error: {e}")

def ensure_directory_permissions(directory):
    """
    Ensure full permissions for a directory and its contents
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Set directory permissions
        os.chmod(directory, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        
        # Recursively set permissions for existing contents
        for root, dirs, files in os.walk(directory):
            for dir in dirs:
                try:
                    os.chmod(os.path.join(root, dir), stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
                except Exception as e:
                    print(f"Directory permission error: {e}")
            
            for file in files:
                try:
                    set_file_permissions(os.path.join(root, file))
                except Exception as e:
                    print(f"File permission error: {e}")
    
    except Exception as e:
        print(f"Directory permission setup error: {e}")

def download_file(url, local_filename):
    """
    Download a file from a URL with robust error handling
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(local_filename), exist_ok=True)
        
        # Stream download to handle large files
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    f.write(chunk)
        
        # Set full permissions
        os.chmod(local_filename, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        
        return local_filename
    except Exception as e:
        print(f"Download error for {url}: {e}")
        return None

class AdvancedSpeakerRecognition:
    def __init__(self):
        """
        Initialize Advanced Speaker Recognition with Pyannote
        """
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

        # Initialize Pyannote Speaker Diarization
        try:
            auth_token = os.getenv('PYANNOTE_AUTH_TOKEN')
            self.diarization_pipeline = SpeakerDiarization(
                segmentation='pyannote/segmentation',
                embedding='pyannote/embedding',
                use_auth_token=auth_token
            )
        except Exception as e:
            self.logger.error(f"Diarization pipeline loading error: {e}")
            raise

    def extract_speaker_embedding(self, audio_path: str) -> Union[np.ndarray, None]:
        """
        Extract speaker embedding using Pyannote Diarization
        
        Args:
            audio_path (str): Path to audio file
        
        Returns:
            numpy.ndarray: Speaker embedding
        """
        try:
            diarization = self.diarization_pipeline(audio_path)
            embeddings = []

            # Parallelize embedding extraction
            def process_segment(segment):
                return self.diarization_pipeline.embedding(audio_path, segment)

            with ThreadPoolExecutor() as executor:
                embeddings = list(executor.map(
                    process_segment, 
                    [segment for segment, _, _ in diarization.itertracks(yield_label=True)]
                ))

            if embeddings:
                return np.mean(embeddings, axis=0)  # Average embeddings if multiple segments
            return None
        except Exception as e:
            self.logger.error(f"Embedding extraction error: {e}")
            return None

    def perform_diarization(self, audio_path: str) -> Dict[str, Dict]:
        """
        Perform speaker diarization using Pyannote
        
        Args:
            audio_path (str): Path to audio file
        
        Returns:
            dict: Diarization results with speaker segments
        """
        try:
            diarization = self.diarization_pipeline(audio_path)
            speaker_segments = {}

            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if speaker not in speaker_segments:
                    speaker_segments[speaker] = {
                        'total_speech_time': 0,
                        'segments': []
                    }

                speaker_segments[speaker]['total_speech_time'] += turn.end - turn.start
                speaker_segments[speaker]['segments'].append({
                    'start': turn.start,
                    'end': turn.end
                })

            return speaker_segments
        except Exception as e:
            self.logger.error(f"Diarization error: {e}")
            return {}

    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute similarity between two embeddings
        
        Args:
            emb1 (numpy.ndarray): First embedding
            emb2 (numpy.ndarray): Second embedding
        
        Returns:
            float: Similarity score
        """
        try:
            emb1 = emb1 / np.linalg.norm(emb1)
            emb2 = emb2 / np.linalg.norm(emb2)
            cosine_sim = 1 - cosine(emb1, emb2)
            return cosine_sim
        except Exception as e:
            self.logger.error(f"Similarity computation error: {e}")
            return 0.0

    def identify_speaker(self, input_embedding: np.ndarray, threshold: float = 0.5) -> Dict:
        """
        Identify speaker by matching input embedding with database
        
        Args:
            input_embedding (numpy.ndarray): Input speaker embedding
            threshold (float): Similarity threshold for matching
        
        Returns:
            dict: Speaker identification results
        """
        try:
            registered_speakers = list(self.speakers_collection.find())

            if not registered_speakers:
                self.logger.warning("No speakers found in database")
                return {
                    'match': False,
                    'name': 'Unknown',
                    'confidence': 0.0
                }

            similarities = []
            for speaker in registered_speakers:
                ref_embedding = np.array(speaker['embedding'])
                similarity = self.compute_similarity(input_embedding, ref_embedding)
                similarities.append((speaker['name'], similarity))

            best_match = max(similarities, key=lambda x: x[1], default=None)

            if best_match and best_match[1] >= threshold:
                return {
                    'match': True,
                    'name': best_match[0],
                    'confidence': best_match[1]
                }

            return {
                'match': False,
                'name': 'Unknown',
                'confidence': 0.0
            }
        except Exception as e:
            self.logger.error(f"Speaker identification error: {e}")
            return {
                'match': False,
                'name': 'Unknown',
                'confidence': 0.0
            }

    def add_new_speaker(self, name: str, embedding: np.ndarray) -> bool:
        """
        Add a new speaker to the database
        
        Args:
            name (str): Speaker name
            embedding (numpy.ndarray): Speaker voice embedding
        
        Returns:
            bool: Success of operation
        """
        try:
            if embedding is None or len(embedding) == 0:
                self.logger.error("Invalid embedding: Cannot add speaker")
                return False

            existing_speaker = self.speakers_collection.find_one({'name': name})

            if existing_speaker:
                self.speakers_collection.update_one(
                    {'name': name},
                    {'$set': {
                        'embedding': embedding.tolist(),
                        'last_updated': datetime.now()
                    }}
                )
                self.logger.info(f"Updated existing speaker: {name}")
            else:
                self.speakers_collection.insert_one({
                    'name': name,
                    'embedding': embedding.tolist(),
                    'created_at': datetime.now()
                })
                self.logger.info(f"Added new speaker: {name}")

            return True
        except Exception as e:
            self.logger.error(f"Error adding speaker: {e}")
            return False

# Example usage
def main():
    speaker_rec = AdvancedSpeakerRecognition()
    audio_path = input("Enter path to audio file: ").strip()

    if os.path.exists(audio_path):
        embedding = speaker_rec.extract_speaker_embedding(audio_path)

        if embedding is not None:
            diarization_results = speaker_rec.perform_diarization(audio_path)
            print("\nDiarization Results:")
            for speaker, details in diarization_results.items():
                print(f"Speaker {speaker}: Total Speech Time: {details['total_speech_time']:.2f}s")

            match = speaker_rec.identify_speaker(embedding)
            print(f"\nSpeaker Match: {match['name']} (Confidence: {match['confidence']:.2%})")
        else:
            print("Failed to extract embedding.")
    else:
        print("Audio file does not exist.")


if __name__ == "__main__":
    main() 