import os
import numpy as np
import soundfile as sf
import whisper
import torch
import librosa
from sklearn.cluster import KMeans
from pyannote.audio import Pipeline
from transformers import pipeline as hf_pipeline, Wav2Vec2FeatureExtractor, Wav2Vec2Model
import langdetect
import logging

from transcription import RealTimeTranscription
from voice_database import VoiceDatabase
from advanced_speaker_recognition import AdvancedSpeakerRecognition

class AudioAnalyzer:
    def __init__(self):
        """
        Initialize Audio Analysis Components
        """
        # Configure logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s: %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('audio_analysis_log.txt', mode='w')
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Load models during initialization
        self.load_models()
        
        # Transcription model
        self.transcription_model = RealTimeTranscription(model_size='small')
        
        # Voice Database and Speaker Recognition
        self.voice_db = VoiceDatabase()
        self.speaker_recognition = AdvancedSpeakerRecognition()
        
        # Language Detection
        self.language_detector = hf_pipeline("text-classification", 
                                             model="papluca/xlm-roberta-base-language-detection")
        
        # Initialize Wav2Vec2 for embedding extraction
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        self.wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

    def load_models(self):
        """
        Load necessary models with robust error handling
        """
        try:
            # Load Whisper model
            self.whisper_model = whisper.load_model("small")
            
            # Disable diarization pipeline for now
            self.diarization_pipeline = None

        except Exception as e:
            self.logger.error(f"Model loading error: {e}")
            # Fallback mechanism
            self.whisper_model = None
            raise

    def preprocess_audio(self, audio_path, target_sr=16000):
        """
        Preprocess audio to ensure consistent format and length
        
        Args:
            audio_path (str): Path to the audio file
            target_sr (int): Target sample rate
        
        Returns:
            numpy.ndarray: Preprocessed audio data
        """
        try:
            # Load audio with librosa for more robust handling
            audio, sr = librosa.load(audio_path, sr=target_sr)
            
            # Ensure minimum duration
            min_duration = target_sr * 5  # 5 seconds minimum
            if len(audio) < min_duration:
                # Pad audio if too short
                pad_length = min_duration - len(audio)
                audio = np.pad(audio, (0, pad_length), mode='constant')
            
            return audio
        
        except Exception as e:
            self.logger.error(f"Audio preprocessing error: {e}")
            raise

    def perform_basic_diarization(self, audio_data):
        """
        Perform basic speaker segmentation when advanced diarization is not available
        
        Args:
            audio_data (numpy.ndarray): Audio time series data
        
        Returns:
            dict: Basic diarization results
        """
        try:
            # Use RMS energy to detect potential speaker changes
            rms = librosa.feature.rms(y=audio_data)[0]
            
            # Threshold for detecting significant energy changes
            threshold = np.mean(rms) * 1.5
            
            # Identify potential speaker segments
            speaker_segments = []
            current_segment_start = 0
            current_speaker = 0
            
            for i, energy in enumerate(rms):
                if energy > threshold:
                    # New potential speaker segment
                    if i > current_segment_start:
                        speaker_segments.append({
                            'speaker': f'Speaker_{current_speaker}',
                            'start': current_segment_start / len(rms),
                            'end': i / len(rms)
                        })
                        current_speaker += 1
                        current_segment_start = i
            
            # Add final segment
            speaker_segments.append({
                'speaker': f'Speaker_{current_speaker}',
                'start': current_segment_start / len(rms),
                'end': len(rms) / len(rms)
            })
            
            return {
                'total_speakers': current_speaker + 1,
                'speakers': {
                    segment['speaker']: {
                        'total_speech_time': segment['end'] - segment['start'],
                        'segments': [segment]
                    } for segment in speaker_segments
                }
            }
        
        except Exception as e:
            self.logger.error(f"Basic diarization error: {e}")
            return {
                'total_speakers': 1,
                'speakers': {
                    'Speaker_0': {
                        'total_speech_time': 1.0,
                        'segments': [{'start': 0, 'end': 1.0}]
                    }
                }
            }

    def extract_speaker_embedding(self, audio_path):
        """
        Extract voice embedding using Wav2Vec2
        
        Args:
            audio_path (str): Path to the audio file
        
        Returns:
            numpy.ndarray: Voice embedding
        """
        try:
            # Load audio file
            audio_input, sample_rate = sf.read(audio_path)
            
            # Ensure mono audio
            if audio_input.ndim > 1:
                audio_input = audio_input.mean(axis=1)
            
            # Resample if necessary
            if sample_rate != 16000:
                audio_input = librosa.resample(audio_input, sample_rate, 16000)
            
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

    def identify_speaker(self, audio_path):
        """
        Identify speaker using Advanced Speaker Recognition
        
        Args:
            audio_path (str): Path to the audio file
        
        Returns:
            dict: Speaker identification results
        """
        try:
            # Extract embedding from the input audio
            embedding = self.extract_speaker_embedding(audio_path)
            
            if embedding is None:
                return {
                    'match': False,
                    'name': 'Unknown',
                    'confidence': 0.0
                }
            
            # Use Advanced Speaker Recognition to identify speaker
            speaker_match = self.speaker_recognition.identify_speaker(embedding)
            
            self.logger.info(f"Speaker Identification Result: {speaker_match}")
            
            return speaker_match
        
        except Exception as e:
            self.logger.error(f"Speaker identification error: {e}")
            return {
                'match': False,
                'name': 'Unknown',
                'confidence': 0.0
            }

    def analyze_audio(self, audio_path):
        """
        Comprehensive audio analysis with robust error handling
        
        :param audio_path: Path to the audio file
        :return: Dictionary with analysis results
        """
        try:
            # Validate whisper model is loaded
            if not hasattr(self, 'whisper_model') or self.whisper_model is None:
                self.load_models()

            # Preprocess audio
            audio_data = self.preprocess_audio(audio_path)
            
            # Temporary file for potential diarization
            temp_wav_path = "temp_diarization.wav"
            sf.write(temp_wav_path, audio_data, 16000)

            # Transcription
            transcription = self.whisper_model.transcribe(
                audio_path, 
                fp16=False  # Ensure CPU compatibility
            )

            # Speaker Diarization
            try:
                # Attempt advanced diarization if pipeline is available
                if self.diarization_pipeline:
                    diarization_results = self.perform_diarization(temp_wav_path)
                else:
                    # Fallback to basic diarization
                    diarization_results = self.perform_basic_diarization(audio_data)
            except Exception as diar_error:
                self.logger.warning(f"Diarization error, using basic segmentation: {diar_error}")
                diarization_results = self.perform_basic_diarization(audio_data)

            # Clean up temporary file
            os.unlink(temp_wav_path)

            # Add speaker identification
            speaker_identification = self.identify_speaker(audio_path)

            # Compile results
            analysis_results = {
                'transcription': {
                    'text': transcription['text'],
                    'language': transcription.get('language', 'Unknown')
                },
                'speaker_diarization': diarization_results,
                'audio_features': self.extract_audio_features(audio_data),
                'speaker_identification': {
                    'name': speaker_identification['name'],
                    'confidence': speaker_identification['confidence'],
                    'match': speaker_identification['match']
                }
            }

            return analysis_results

        except Exception as e:
            self.logger.error(f"Audio analysis error: {e}", exc_info=True)
            raise

    def transcribe_audio(self, audio_path):
        """
        Automatically transcribe audio file with multilingual support
        
        :param audio_path: Path to the audio file
        :return: Transcription details
        """
        try:
            self.logger.info(f"Attempting to transcribe audio: {audio_path}")
            
            # Validate audio file
            if not os.path.exists(audio_path):
                self.logger.error(f"Audio file not found: {audio_path}")
                return {'text': '', 'language': 'Unknown'}

            # Attempt to read audio file details
            try:
                audio_data, sample_rate = sf.read(audio_path)
                self.logger.info(f"Audio file details - Sample Rate: {sample_rate}, Shape: {audio_data.shape}")
            except Exception as read_error:
                self.logger.error(f"Error reading audio file: {read_error}")
                return {'text': '', 'language': 'Unknown'}

            # Process transcription
            transcription_result = self.transcription_model.process_transcription(audio_path)
            
            self.logger.info(f"Transcription result: {transcription_result}")
            
            return {
                'text': transcription_result.get('original_text', ''),
                'language': transcription_result.get('detected_language', 'Unknown')
            }
        
        except Exception as e:
            self.logger.error(f"Transcription error: {e}", exc_info=True)
            return {'text': '', 'language': 'Unknown'}

    def detect_language(self, text):
        """
        Detect language of the text with focus on English and Hindi
        
        :param text: Input text
        :return: Detected language
        """
        try:
            if not text:
                return 'English'
            
            # Simplified language mapping
            language_names = {
                'en': 'English',
                'hi': 'Hindi',
                'ur': 'Hindi'  # Treat Urdu as Hindi
            }
            
            # Use langdetect for language detection
            detected_lang = langdetect.detect(text)
            
            # Return only English or Hindi, default to English
            return language_names.get(detected_lang, 'English')
        
        except Exception:
            return 'English'  # Default to English on detection failure

    def perform_diarization(self, audio_path):
        """
        Perform speaker diarization
        
        Args:
            audio_path (str): Path to the audio file
        
        Returns:
            dict: Diarization results
        """
        try:
            # Perform diarization
            diarization = self.diarization_pipeline(audio_path)

            # Process diarization results
            speakers = {}
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if speaker not in speakers:
                    speakers[speaker] = {
                        'total_speech_time': 0,
                        'segments': []
                    }
                speakers[speaker]['total_speech_time'] += turn.end - turn.start
                speakers[speaker]['segments'].append({
                    'start': turn.start,
                    'end': turn.end
                })

            return {
                'total_speakers': len(speakers),
                'speakers': speakers
            }

        except Exception as e:
            self.logger.error(f"Diarization error: {e}")
            return {'error': str(e)}

    def extract_audio_features(self, audio_data):
        """
        Extract audio features
        
        Args:
            audio_data (numpy.ndarray): Audio time series
        
        Returns:
            dict: Audio features
        """
        try:
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data)[0]
            
            # Temporal features
            rms = librosa.feature.rms(y=audio_data)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)[0]

            return {
                'spectral_centroid_mean': np.mean(spectral_centroids),
                'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
                'rms_mean': np.mean(rms),
                'zero_crossing_rate_mean': np.mean(zero_crossing_rate)
            }

        except Exception as e:
            self.logger.error(f"Feature extraction error: {e}")
            return {}

    def generate_summary(self, analysis_results):
        """
        Generate a summary of audio analysis
        
        :param analysis_results: Results from audio analysis
        :return: Formatted summary
        """
        try:
            summary = "--- Audio Analysis Summary ---\n\n"
            
            # Transcription Summary
            transcription = analysis_results['transcription']
            summary += "Transcription:\n"
            summary += f"Text: {transcription['text']}\n"
            summary += f"Language: {transcription['language']}\n\n"
            
            # Diarization Summary
            diarization = analysis_results['speaker_diarization']
            summary += "Speaker Diarization:\n"
            summary += f"Total Speakers: {diarization.get('total_speakers', 'N/A')}\n"
            
            # Audio Features Summary
            features = analysis_results['audio_features']
            summary += "\nAudio Characteristics:\n"
            summary += f"Spectral Centroid: {features.get('spectral_centroid_mean', 'N/A')}\n"
            summary += f"RMS Energy: {features.get('rms_mean', 'N/A')}\n"
             
            # Speaker Identification Summary
            speaker_id = analysis_results.get('speaker_identification', {})
            summary += "\nSpeaker Identification:\n"
            summary += f"Name: {speaker_id.get('name', 'Unknown')}\n"
            summary += f"Confidence: {speaker_id.get('confidence', 0.0):.2%}\n"
            summary += f"Match: {'Yes' if speaker_id.get('match', False) else 'No'}\n"

            return summary
 
        except Exception as e:
            self.logger.error(f"Summary generation error: {e}")
            return f"Error generating summary: {e}"

def main():
    """
    Demonstration of audio analysis capabilities
    """
    try:
        # Initialize audio analyzer
        analyzer = AudioAnalyzer()
        
        # Prompt for audio file path
        audio_path = input("Enter the path to the audio file: ").strip()
        
        # Perform analysis
        results = analyzer.analyze_audio(audio_path)
        
        # Generate and print summary
        summary = analyzer.generate_summary(results)
        print("\nAnalysis Summary:")
        print(summary)
    
    except Exception as e:
        print(f"Audio analysis error: {e}")

if __name__ == "__main__":
    main() 