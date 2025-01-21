import os
import tkinter as tk
from tkinter import messagebox, scrolledtext, filedialog, ttk
import threading
import queue
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import whisper
import logging
import tempfile
import re
import uuid

from transcription import RealTimeTranscription
from advanced_speaker_recognition import AdvancedSpeakerRecognition
from capture_voice import VoiceCapture

# Disable specific warnings
logging.getLogger('whisper').setLevel(logging.ERROR)
logging.getLogger('torch').setLevel(logging.ERROR)

class RealTimeDiarizationApp:
    def __init__(self, root):
        """
        Initialize Real-Time Diarization Tkinter Application
        
        Args:
            root (tk.Tk): Main Tkinter window
        """
        self.root = root
        self.root.title("Real-Time Speaker Diarization")
        self.root.geometry("800x600")

        # Configure logging
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.voice_capture = VoiceCapture()
        self.transcriber = RealTimeTranscription()
        self.speaker_recognition = AdvancedSpeakerRecognition()

        # Audio processing parameters
        self.duration = 0.5
        self.sample_rate = 16000
        self.channels = 1
        self.block_size = int(self.sample_rate * self.duration)
        
        # Temporary file management
        self.temp_dir = tempfile.mkdtemp()
        
        # Audio processing queue
        self.audio_queue = queue.Queue(maxsize=10)
        
        # Processing state
        self.is_processing = False
        self.processing_thread = None
        self.capture_thread = None

        # Create UI
        self.create_ui()

    def create_ui(self):
        """
        Create Tkinter User Interface
        """
        # Main frame
        main_frame = tk.Frame(self.root, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = tk.Label(
            main_frame, 
            text="Real-Time Speaker Diarization", 
            font=("Helvetica", 16, "bold")
        )
        title_label.pack(pady=10)

        # Control Frame
        control_frame = tk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)

        # Start/Stop Button
        self.start_button = tk.Button(
            control_frame, 
            text="Start Diarization", 
            command=self.toggle_diarization,
            bg="green",
            fg="white"
        )
        self.start_button.pack(side=tk.LEFT, padx=5)

        # Status Label
        self.status_label = tk.Label(
            control_frame, 
            text="Ready", 
            fg="blue"
        )
        self.status_label.pack(side=tk.LEFT, padx=10)

        # Results Text Area
        results_frame = tk.Frame(main_frame)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        results_label = tk.Label(
            results_frame, 
            text="Transcription Results:", 
            font=("Helvetica", 12)
        )
        results_label.pack()

        self.results_text = scrolledtext.ScrolledText(
            results_frame, 
            wrap=tk.WORD, 
            height=20
        )
        self.results_text.pack(fill=tk.BOTH, expand=True)

        # Save and Clear Buttons
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)

        save_button = tk.Button(
            button_frame, 
            text="Save Results", 
            command=self.save_results
        )
        save_button.pack(side=tk.LEFT, padx=5)

        clear_button = tk.Button(
            button_frame, 
            text="Clear Results", 
            command=self.clear_results
        )
        clear_button.pack(side=tk.LEFT, padx=5)

    def toggle_diarization(self):
        """
        Toggle real-time diarization on and off
        """
        if not self.is_processing:
            self.start_diarization()
        else:
            self.stop_diarization()

    def start_diarization(self):
        """
        Start real-time speaker diarization
        """
        try:
            self.is_processing = True
            self.start_button.config(text="Stop Diarization", bg="red")
            self.status_label.config(text="Processing...", fg="green")

            # Start audio capture thread
            self.capture_thread = threading.Thread(
                target=self._capture_audio, 
                daemon=True
            )
            self.capture_thread.start()

            # Start processing thread
            self.processing_thread = threading.Thread(
                target=self._process_audio_queue, 
                daemon=True
            )
            self.processing_thread.daemon = True
            self.processing_thread.start()

        except Exception as e:
            self.logger.error(f"Diarization start error: {e}")
            messagebox.showerror("Error", str(e))
            self.stop_diarization()

    def stop_diarization(self):
        """
        Stop real-time diarization
        """
        self.is_processing = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1)
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1)
        
        self.start_button.config(text="Start Diarization", bg="green")
        self.status_label.config(text="Stopped", fg="red")

    def _capture_audio(self):
        """
        Continuously capture audio in real-time
        """
        try:
            def audio_callback(indata, frames, time_info, status):
                if status:
                    self.logger.warning(f"Audio input stream error: {status}")
                    return
                
                # Create temporary file for the audio chunk
                temp_audio_path = os.path.join(
                    self.temp_dir, 
                    f"audio_chunk_{uuid.uuid4()}.wav"
                )
                
                # Save audio chunk
                sf.write(temp_audio_path, indata, self.sample_rate)
                
                # Add to processing queue
                self.audio_queue.put(temp_audio_path)

            # Start audio stream
            with sd.InputStream(
                samplerate=self.sample_rate, 
                channels=self.channels,
                dtype='float32',
                callback=audio_callback,
                blocksize=self.block_size
            ):
                # Keep capturing until processing is stopped
                while self.is_processing:
                    time.sleep(self.duration)

        except Exception as e:
            self.logger.error(f"Audio capture error: {e}")
            self.root.after(0, self.stop_diarization)

    def _process_audio_queue(self):
        """
        Process audio chunks from the queue
        """
        try:
            while self.is_processing:
                try:
                    # Get audio chunk with timeout
                    audio_path = self.audio_queue.get(timeout=1)
                    
                    # Process the audio chunk
                    self._process_single_chunk(audio_path)
                
                except queue.Empty:
                    # No audio in queue, continue waiting
                    continue
                except Exception as chunk_error:
                    self.logger.error(f"Audio chunk processing error: {chunk_error}")
        
        except Exception as e:
            self.logger.error(f"Audio queue processing error: {e}")
            self.root.after(0, self.stop_diarization)

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
            min_duration = 0.3
            min_samples = int(min_duration * sample_rate)
            if len(audio_data) < min_samples:
                return
            
            # Ensure mono audio
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)
            
            # Normalize audio
            audio_data = audio_data.astype(np.float32)
            audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Process transcription
            transcription_result = self.transcriber.process_transcription(audio_path)
            
            # Extract transcribed text
            transcribed_text = transcription_result['original_text'].strip()
            detected_language = transcription_result['detected_language']
            speaker_name = transcription_result['speaker']
            speaker_confidence = transcription_result['speaker_confidence']
            
            # Remove numbers and extra whitespace
            transcribed_text = re.sub(r'\d+', '', transcribed_text)
            transcribed_text = ' '.join(transcribed_text.split())
            
            # Update UI with results
            if len(transcribed_text) > 3:
                result_text = (
                    f"Speaker: {speaker_name} (Confidence: {speaker_confidence:.2%})\n"
                    f"Language: {detected_language}\n"
                    f"Transcription: {transcribed_text}\n\n"
                )
                self.update_results(result_text)
            
            # Clean up temporary file
            os.unlink(audio_path)
        
        except Exception as e:
            self.logger.error(f"Error processing audio chunk: {e}")

    def update_results(self, result_text):
        """
        Update results text area from the main thread
        """
        def update():
            self.results_text.insert(tk.END, result_text)
            self.results_text.see(tk.END)
        
        self.root.after(0, update)

    def save_results(self):
        """
        Save transcription results to a text file
        """
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt")]
            )
            if file_path:
                with open(file_path, 'w') as f:
                    f.write(self.results_text.get('1.0', tk.END))
                messagebox.showinfo("Success", "Results saved successfully!")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def clear_results(self):
        """
        Clear the results display
        """
        self.results_text.delete('1.0', tk.END)

def main():
    """
    Launch Real-Time Diarization Tkinter App
    """
    root = tk.Tk()
    app = RealTimeDiarizationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()