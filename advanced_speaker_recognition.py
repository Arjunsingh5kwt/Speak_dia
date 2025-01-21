import os
import sys
import warnings
import numpy as np
import torch
import torchaudio
from speechbrain.inference import SpeakerRecognition
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

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

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
        # Load environment variables
        load_dotenv()

        # Configure logging
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # MongoDB Connection
        try:
            # Use environment variable or default local connection
            mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
            self.mongo_client = pymongo.MongoClient(mongo_uri)
            
            # Select database and collection
            self.db = self.mongo_client['voice_recognition_db']
            self.voice_collection = self.db['voice_embedding']
            
            # Cache embeddings for faster processing
            self.cached_embeddings = self.load_embeddings()
            
            self.logger.info(f"Loaded {len(self.cached_embeddings)} voice embeddings")
        
        except Exception as e:
            self.logger.error(f"MongoDB connection error: {e}")
            self.cached_embeddings = {}
            raise

    def load_embeddings(self):
        """
        Load all voice embeddings from MongoDB
        
        Returns:
            dict: Cached embeddings with name as key and embedding as value
        """
        try:
            # Retrieve all embeddings from the collection
            embeddings = {}
            for doc in self.voice_collection.find():
                embeddings[doc['name']] = np.array(doc['embedding'])
            
            return embeddings
        
        except Exception as e:
            self.logger.error(f"Error loading embeddings: {e}")
            return {}

    def calculate_similarity(self, embedding1, embedding2):
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1 (numpy.ndarray): First embedding
            embedding2 (numpy.ndarray): Second embedding
        
        Returns:
            float: Similarity score (0-1)
        """
        try:
            # Ensure embeddings are numpy arrays
            embedding1 = np.array(embedding1)
            embedding2 = np.array(embedding2)
            
            # Calculate cosine similarity
            # Note: cosine returns distance, so 1 - distance gives similarity
            similarity = 1 - cosine(embedding1, embedding2)
            
            return similarity
        
        except Exception as e:
            self.logger.error(f"Similarity calculation error: {e}")
            return 0.0

    def identify_speaker(self, input_embedding):
        """
        Identify speaker based on input embedding
        
        Args:
            input_embedding (numpy.ndarray): Embedding to match
        
        Returns:
            dict: Speaker identification results
        """
        try:
            # If no cached embeddings, return unknown
            if not self.cached_embeddings:
                return {
                    'match': False,
                    'name': 'Unknown',
                    'confidence': 0.0
                }
            
            # Find best match
            best_match = {
                'match': False,
                'name': 'Unknown',
                'confidence': 0.0
            }
            
            # Compare with all cached embeddings
            for name, stored_embedding in self.cached_embeddings.items():
                similarity = self.calculate_similarity(input_embedding, stored_embedding)
                
                # Update best match if similarity is higher
                if similarity > 0.5 and similarity > best_match['confidence']:
                    best_match = {
                        'match': True,
                        'name': name,
                        'confidence': similarity
                    }
            
            return best_match
        
        except Exception as e:
            self.logger.error(f"Speaker identification error: {e}")
            return {
                'match': False,
                'name': 'Unknown',
                'confidence': 0.0
            }

    def add_new_speaker(self, name, embedding):
        """
        Add a new speaker to the database
        
        Args:
            name (str): Name of the speaker
            embedding (numpy.ndarray): Voice embedding
        
        Returns:
            bool: True if successfully added, False otherwise
        """
        try:
            # Prepare embedding document
            embedding_doc = {
                'name': name,
                'embedding': embedding.tolist(),
                'timestamp': datetime.utcnow()
            }
            
            # Insert into MongoDB
            result = self.voice_collection.insert_one(embedding_doc)
            
            # Update cached embeddings
            self.cached_embeddings[name] = embedding
            
            self.logger.info(f"New speaker {name} added successfully")
            return True
        
        except Exception as e:
            self.logger.error(f"Error adding new speaker: {e}")
            return False

    def refresh_embeddings(self):
        """
        Refresh cached embeddings from database
        """
        try:
            self.cached_embeddings = self.load_embeddings()
            self.logger.info(f"Refreshed embeddings. Total: {len(self.cached_embeddings)}")
        except Exception as e:
            self.logger.error(f"Embedding refresh error: {e}")

class SpeakerRecognition:
    def __init__(self, embedding_model, voice_database):
        # Use lighter model or pre-compute embeddings
        self.model = embedding_model
        self.voice_database = voice_database
        
        # Enable GPU acceleration if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Use faster inference methods
        torch.set_grad_enabled(False)  # Disable gradient computation
        self.model.eval()  # Set model to evaluation mode

    def recognize_speaker(self, audio_chunk):
        # Reduce processing time by:
        # 1. Using smaller audio chunks
        # 2. Implementing faster similarity calculation
        start_time = time.time()
        
        # Convert audio to tensor and move to device
        audio_tensor = torch.from_numpy(audio_chunk).to(self.device)
        
        # Use faster embedding extraction
        embedding = self.model.extract_embedding(audio_tensor)
        
        # Vectorized similarity computation
        similarities = np.array([
            np.dot(embedding, ref_embedding) / 
            (np.linalg.norm(embedding) * np.linalg.norm(ref_embedding))
            for ref_embedding in self.voice_database.embeddings
        ])
        
        best_match_index = np.argmax(similarities)
        confidence = similarities[best_match_index]
        
        processing_time = time.time() - start_time
        print(f"Speaker recognition took {processing_time:.4f} seconds")
        
        return self.voice_database.speakers[best_match_index], confidence

def main():
    """
    Demonstration of Advanced Speaker Recognition
    """
    try:
        # Initialize speaker recognition
        speaker_rec = AdvancedSpeakerRecognition()
        
        # Example usage
        print("Advanced Speaker Recognition Initialized")
        print(f"Cached Speakers: {list(speaker_rec.cached_embeddings.keys())}")
    
    except Exception as e:
        print(f"Initialization error: {e}")

if __name__ == "__main__":
    main() 