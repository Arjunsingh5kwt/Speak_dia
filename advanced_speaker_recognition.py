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
            mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
            self.mongo_client = pymongo.MongoClient(mongo_uri)
            self.db = self.mongo_client['voice_recognition_db']
            self.speakers_collection = self.db['voice_embeddings']
        except Exception as e:
            self.logger.error(f"MongoDB connection error: {e}")
            raise
 
        # Initialize Pyannote Speaker Diarization
        try:
            # Use pre-trained speaker diarization pipeline
            self.diarization_pipeline = SpeakerDiarization(
                segmentation='pyannote/segmentation',
                embedding='pyannote/embedding',
                use_auth_token="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"
            )
        except Exception as e:
            self.logger.error(f"Diarization pipeline loading error: {e}")
            raise

    def extract_speaker_embedding(self, audio_path: str) -> np.ndarray:
        """
        Extract speaker embedding using Pyannote Diarization
        
        Args:
            audio_path (str): Path to audio file
        
        Returns:
            numpy.ndarray: Speaker embedding
        """
        try:
            # Perform diarization to get embeddings
            diarization = self.diarization_pipeline(audio_path)
            
            # Extract embeddings from diarization
            embeddings = []
            for segment, _, speaker in diarization.itertracks(yield_label=True):
                # Get embedding for each speaker segment
                embedding = self.diarization_pipeline.embedding(audio_path, segment)
                embeddings.append(embedding)
            
            # Average embeddings if multiple segments
            if embeddings:
                return np.mean(embeddings, axis=0)
            
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
            # Perform diarization
            diarization = self.diarization_pipeline(audio_path)
            
            # Process diarization results
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

    def identify_speaker(self, input_embedding: np.ndarray, threshold: float = 0.5) -> Dict:
        """
        Advanced speaker identification with comprehensive matching strategies
        
        Args:
            input_embedding (numpy.ndarray): Input speaker embedding
            threshold (float): Similarity threshold for matching
        
        Returns:
            dict: Speaker identification results
        """
        try:
            # Fetch all registered speakers
            registered_speakers = list(self.speakers_collection.find())
            
            # No speakers in database
            if not registered_speakers:
                self.logger.warning("No speakers found in database")
                return {
                    'match': False,
                    'name': 'Unknown',
                    'confidence': 0.0
                }
            
            # Debugging: Print input embedding details
            self.logger.info(f"Input Embedding Shape: {input_embedding.shape}")
            self.logger.info(f"Input Embedding Type: {type(input_embedding)}")
            
            # Comprehensive similarity computation
            def compute_similarity(emb1, emb2):
                """
                Compute multiple similarity metrics with robust error handling
                """
                try:
                    # Ensure embeddings are numpy arrays
                    emb1 = np.asarray(emb1)
                    emb2 = np.asarray(emb2)
                    
                    # Normalize embeddings
                    emb1 = emb1 / np.linalg.norm(emb1)
                    emb2 = emb2 / np.linalg.norm(emb2)
                    
                    # Multiple similarity metrics
                    cosine_sim = 1 - cosine(emb1, emb2)
                    dot_product_sim = np.dot(emb1, emb2)
                    
                    # Weighted similarity
                    similarity = 0.7 * cosine_sim + 0.3 * dot_product_sim
                    
                    return similarity
                except Exception as e:
                    self.logger.error(f"Similarity computation error: {e}")
                    return 0.0
            
            # Compare embeddings
            similarities = []
            speaker_details = []
            
            for speaker in registered_speakers:
                if 'embedding' in speaker:
                    try:
                        # Convert stored embedding to numpy array
                        ref_embedding = np.array(speaker['embedding'])
                        
                        # Debugging: Print reference embedding details
                        self.logger.info(f"Ref Embedding Shape: {ref_embedding.shape}")
                        
                        # Ensure embeddings have compatible shapes
                        if ref_embedding.ndim == input_embedding.ndim:
                            # Compute similarity
                            similarity = compute_similarity(input_embedding, ref_embedding)
                            
                            # Debugging: Log individual similarities
                            self.logger.info(f"Similarity for {speaker.get('name', 'Unknown')}: {similarity}")
                            
                            similarities.append(similarity)
                            speaker_details.append({
                                'name': speaker.get('name', 'Unknown'),
                                'embedding': ref_embedding
                            })
                        else:
                            self.logger.warning(f"Shape mismatch for {speaker.get('name', 'Unknown')}")
                    
                    except Exception as e:
                        self.logger.error(f"Embedding comparison error: {e}")
            
            # Find best match
            if similarities:
                best_match_index = np.argmax(similarities)
                best_similarity = similarities[best_match_index]
                
                # Adaptive thresholding with logging
                dynamic_threshold = max(0.5, threshold)
                self.logger.info(f"Dynamic Threshold: {dynamic_threshold}")
                self.logger.info(f"Best Similarity: {best_similarity}")
                
                # Check against threshold
                if best_similarity >= dynamic_threshold:
                    best_match = speaker_details[best_match_index]
                    return {
                        'match': True,
                        'name': best_match['name'],
                        'confidence': best_similarity,
                        'embedding': best_match['embedding'].tolist()
                    }
                else:
                    self.logger.warning(f"No match found. Best similarity {best_similarity} below threshold {dynamic_threshold}")
            
            # No match found
            return {
                'match': False,
                'name': 'Unknown',
                'confidence': 0.0
            }
        
        except Exception as e:
            self.logger.error(f"Comprehensive speaker identification error: {e}")
            return {
                'match': False,
                'name': 'Unknown',
                'confidence': 0.0
            }

    def add_new_speaker(self, name: str, embedding: np.ndarray) -> bool:
        """
        Add a new speaker to the database with embedding validation
        
        Args:
            name (str): Speaker name
            embedding (numpy.ndarray): Speaker voice embedding
        
        Returns:
            bool: Success of speaker addition
        """
        try:
            # Validate embedding
            if embedding is None or len(embedding) == 0:
                self.logger.error("Invalid embedding: Cannot add speaker")
                return False
            
            # Check if speaker already exists
            existing_speaker = self.speakers_collection.find_one({'name': name})
            
            if existing_speaker:
                # Update existing speaker
                self.speakers_collection.update_one(
                    {'name': name},
                    {'$set': {
                        'embedding': embedding.tolist(),
                        'last_updated': datetime.now()
                    }}
                )
                self.logger.info(f"Updated existing speaker: {name}")
            else:
                # Insert new speaker
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

def main():
    """
    Demonstration of Advanced Speaker Recognition
    """
    try:
        # Initialize speaker recognition
        speaker_rec = AdvancedSpeakerRecognition()
        
        # Prompt for audio file
        audio_path = input("Enter path to audio file: ").strip()
        
        if os.path.exists(audio_path):
            # Extract embedding
            embedding = speaker_rec.extract_speaker_embedding(audio_path)
            
            if embedding is not None:
                # Perform diarization
                diarization_results = speaker_rec.perform_diarization(audio_path)
                print("\nDiarization Results:")
                for speaker, details in diarization_results.items():
                    print(f"Speaker {speaker}:")
                    print(f"Total Speech Time: {details['total_speech_time']:.2f}s")
                    print("Segments:", details['segments'])
                
                # Identify speaker
                speaker_match = speaker_rec.identify_speaker(embedding)
                print("\nSpeaker Identification:")
                print(f"Name: {speaker_match['name']}")
                print(f"Confidence: {speaker_match['confidence']:.2%}")
                print(f"Match: {'Yes' if speaker_match['match'] else 'No'}")
            else:
                print("Failed to extract speaker embedding.")
        else:
            print("File does not exist.")
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 