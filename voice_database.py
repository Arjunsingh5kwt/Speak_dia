import os
import numpy as np
import json
import pickle
from pymongo import MongoClient
from bson import Binary
import datetime
import base64
import urllib.parse
from dotenv import load_dotenv

class VoiceDatabase:
    def __init__(self):
        """
        Initialize Voice Database with MongoDB or fallback to local storage
        """
        # Load environment variables
        load_dotenv()

        # Use MongoDB URI from environment variable
        mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
        
        try:
            # Connect to MongoDB
            self.client = MongoClient(mongo_uri)
            
            # Select database
            self.db = self.client['voice_recognition_db']
            
            # Select collections
            self.voice_collection = self.db['voice']
            
            # Create index for efficient querying
            self.voice_collection.create_index('name')
        
        except Exception as e:
            print(f"MongoDB connection error: {e}")
            raise

        # Default to local storage
        self.storage_type = 'local'
        self.storage_dir = os.path.join(os.path.expanduser('~'), '.voice_database')
        os.makedirs(self.storage_dir, exist_ok=True)
        self.voices_file = os.path.join(self.storage_dir, 'voices.json')

    def save_voice(self, name, embedding, metadata=None):
        """
        Save voice with fallback to local storage
        """
        if self.storage_type == 'local':
            return self._save_voice_local(name, embedding, metadata)
        
        # MongoDB save method
        try:
            serialized_embedding = Binary(pickle.dumps(embedding))
            
            voice_doc = {
                'name': name,
                'embedding': serialized_embedding,
                'metadata': metadata or {},
                'created_at': datetime.datetime.utcnow()
            }
            
            result = self.voice_collection.replace_one(
                {'name': name},
                voice_doc,
                upsert=True
            )
            
            print(f"Voice for {name} saved successfully.")
            return result
        
        except Exception as e:
            print(f"Error saving voice: {e}")
            return None

    def _save_voice_local(self, name, embedding, metadata=None):
        """
        Save voice to local JSON file
        """
        try:
            # Read existing voices
            voices = {}
            if os.path.exists(self.voices_file):
                with open(self.voices_file, 'r') as f:
                    voices = json.load(f)
            
            # Serialize embedding
            serialized_embedding = base64.b64encode(pickle.dumps(embedding)).decode('utf-8')
            
            # Save voice
            voices[name] = {
                'embedding': serialized_embedding,
                'metadata': metadata or {},
                'created_at': datetime.datetime.utcnow().isoformat()
            }
            
            # Write back to file
            with open(self.voices_file, 'w') as f:
                json.dump(voices, f)
            
            print(f"Voice for {name} saved locally.")
            return True
        
        except Exception as e:
            print(f"Local voice save error: {e}")
            return False

    def get_voice_embedding(self, name):
        """
        Retrieve voice embedding for a specific name
        
        :param name: Name of the speaker
        :return: Voice embedding or None
        """
        try:
            # Find document by name
            voice_doc = self.voice_collection.find_one({'name': name})
            
            if voice_doc and 'embedding' in voice_doc:
                # Deserialize embedding
                return pickle.loads(voice_doc['embedding'])
            
            return None
        
        except Exception as e:
            print(f"Error retrieving voice embedding: {e}")
            return None

    def delete_voice(self, name):
        """
        Delete a voice from the database
        
        :param name: Name of the speaker to delete
        """
        try:
            # Delete document by name
            result = self.voice_collection.delete_one({'name': name})
            
            if result.deleted_count > 0:
                print(f"Voice for {name} deleted successfully.")
            else:
                print(f"No voice found for {name}.")
            
            return result.deleted_count > 0
        
        except Exception as e:
            print(f"Error deleting voice: {e}")
            return False

    def list_voices(self):
        """
        List all registered voices
        
        :return: Dictionary of registered voices
        """
        try:
            # Retrieve all documents
            voices = self.voice_collection.find()
            
            # Convert to dictionary
            voice_dict = {}
            for voice in voices:
                name = voice.get('name')
                metadata = voice.get('metadata', {})
                voice_dict[name] = metadata
            
            return voice_dict
        
        except Exception as e:
            print(f"Error listing voices: {e}")
            return {}

    def identify_speaker(self, embedding, threshold=0.7):
        """
        Identify a speaker from their embedding
        
        :param embedding: Speaker embedding to match
        :param threshold: Similarity threshold for identification
        :return: Name of the identified speaker or 'Unknown'
        """
        try:
            # Retrieve all voice embeddings
            voices = self.voice_collection.find()
            
            best_match = 'Unknown'
            best_similarity = 0
            
            for voice in voices:
                # Deserialize stored embedding
                stored_embedding = pickle.loads(voice['embedding'])
                
                # Cosine similarity
                similarity = np.dot(embedding, stored_embedding) / (
                    np.linalg.norm(embedding) * np.linalg.norm(stored_embedding)
                )
                
                if similarity > best_similarity and similarity > threshold:
                    best_match = voice['name']
                    best_similarity = similarity
            
            return best_match
        
        except Exception as e:
            print(f"Speaker identification error: {e}")
            return 'Unknown'

    def __del__(self):
        """
        Close MongoDB connection when object is deleted
        """
        if hasattr(self, 'client'):
            self.client.close() 