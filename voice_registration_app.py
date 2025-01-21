import tkinter as tk
from tkinter import messagebox, filedialog
import os
import threading
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper
import difflib

class VoiceRegistrationApp:
    def __init__(self, parent_frame):
        self.parent = parent_frame
        
        # Create a frame within the parent frame
        self.frame = tk.Frame(self.parent)
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Ensure voice_registrations directory exists
        self.registrations_dir = "voice_registrations"
        os.makedirs(self.registrations_dir, exist_ok=True)

        # Voice registration parameters
        self.num_samples = 3
        self.sample_duration = 5  # seconds
        self.sample_rate = 16000
        self.current_sample = 0
        self.recordings = []
        self.expected_sentences = [
            "hello how are you.",
            "what is your name?",
            "what are you doing?"
        ]

        # Load Whisper model
        try:
            self.whisper_model = whisper.load_model("base")
        except Exception as e:                                                                                                                                                              
            messagebox.showerror("Model Loading Error", 
                                  f"Failed to load Whisper model: {e}")
            self.parent.quit()

        # Setup UI components
        self.setup_ui()

    def setup_ui(self):
        # Title Label
        title_label = tk.Label(self.frame, text="Voice Registration System", font=("Helvetica", 16))
        title_label.pack(pady=10)

        # Name Input
        name_label = tk.Label(self.frame, text="Enter Name:")
        name_label.pack()
        self.name_entry = tk.Entry(self.frame, width=30)
        self.name_entry.pack()

        # Instruction Label
        self.instruction_label = tk.Label(
            self.frame, 
            text="Click 'Start Registration' to begin",
            font=("Arial", 12)
        )
        self.instruction_label.pack(pady=20)

        # Progress Label
        self.progress_label = tk.Label(
            self.frame, 
            text="", 
            font=("Arial", 10)
        )
        self.progress_label.pack(pady=10)

        # Sentence Display
        self.sentence_label = tk.Label(
            self.frame, 
            text="", 
            font=("Arial", 12, "bold")
        )
        self.sentence_label.pack(pady=10)

        # Recording Button
        self.record_button = tk.Button(
            self.frame, 
            text="Start Registration", 
            command=self.start_registration
        )
        self.record_button.pack(pady=20)

        # Transcription Result
        self.transcription_label = tk.Label(
            self.frame, 
            text="", 
            font=("Arial", 10)
        )
        self.transcription_label.pack(pady=10)

    def start_registration(self):
        """
        Begin voice registration process
        """
        # Validate name
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning("Invalid Input", "Please enter your name")
            return

        # Reset registration process
        self.current_sample = 0
        self.recordings = []
        self.record_button.config(state=tk.DISABLED)

        # Start registration thread
        threading.Thread(
            target=self.registration_workflow, 
            daemon=True
        ).start()

    def registration_workflow(self):
        """
        Manage the entire registration workflow
        """
        try:
            for i in range(self.num_samples):
                # Update UI
                self.update_ui_from_thread(
                    progress=f"Sample {i+1}/{self.num_samples}",
                    sentence=self.expected_sentences[i]
                )

                # Record audio sample
                recording = self.record_audio_sample()
                
                # Verify transcription
                if not self.verify_transcription(recording, self.expected_sentences[i]):
                    # Retry mechanism
                    retry = messagebox.askyesno(
                        "Verification Failed", 
                        "Transcription did not match. Do you want to retry this sample?"
                    )
                    if not retry:
                        self.update_ui_from_thread(
                            progress="Registration Cancelled", 
                            sentence=""
                        )
                        return
                    i -= 1  # Retry the same sample
                    continue

                # Store successful recording
                self.recordings.append(recording)

            # Complete registration
            self.finalize_registration()

        except Exception as e:
            messagebox.showerror("Registration Error", str(e))
        finally:
            self.update_ui_from_thread(
                record_button_state=tk.NORMAL,
                progress="Registration Complete"
            )

    def record_audio_sample(self):
        """
        Record a single audio sample with countdown
        
        Returns:
            numpy.ndarray: Recorded audio data
        """
        # Countdown
        for countdown in range(3, 0, -1):
            self.update_ui_from_thread(
                progress=f"Recording in {countdown} seconds"
            )
            time.sleep(1)

        # Record audio
        self.update_ui_from_thread(
            progress="Recording...",
            sentence_color="red"
        )

        recording = sd.rec(
            int(self.sample_duration * self.sample_rate), 
            samplerate=self.sample_rate, 
            channels=1,
            dtype=np.float32
        )
        sd.wait()

        self.update_ui_from_thread(
            progress="Recording Complete",
            sentence_color="black"
        )

        return recording

    def verify_transcription(self, recording, expected_sentence):
        """
        Verify transcription accuracy
        
        Args:
            recording (numpy.ndarray): Audio recording
            expected_sentence (str): Expected transcription
        
        Returns:
            bool: Transcription verification result
        """
        # Save temporary audio file
        temp_audio_path = "temp_verification.wav"
        sf.write(temp_audio_path, recording, self.sample_rate)

        try:
            # Transcribe audio
            result = self.whisper_model.transcribe(temp_audio_path)
            transcribed_text = result['text'].strip().lower()
            expected_text = expected_sentence.strip().lower()

            # Calculate similarity
            similarity = difflib.SequenceMatcher(
                None, 
                transcribed_text, 
                expected_text
            ).ratio()

            # Update UI with transcription result
            self.update_ui_from_thread(
                transcription=f"Transcribed: {transcribed_text}\n"
                              f"Similarity: {similarity*100:.2f}%"
            )

            # Check similarity threshold
            return similarity > 0.7

        except Exception as e:
            messagebox.showerror("Transcription Error", str(e))
            return False
        finally:
            # Clean up temporary file
            os.unlink(temp_audio_path)

    def finalize_registration(self):
        """
        Consolidate and save voice registration data
        """
        name = self.name_entry.get().strip()
        
        # Sanitize filename
        safe_name = "".join(x for x in name if x.isalnum() or x in [' ', '_']).rstrip()
        
        # Consolidate recordings
        consolidated_recording = np.concatenate(self.recordings)
        
        # Save consolidated recording
        consolidated_path = os.path.join(
            self.registrations_dir, 
            f"{safe_name}_voice_consolidated.wav"
        )
        sf.write(consolidated_path, consolidated_recording, self.sample_rate)

        # Show success message
        messagebox.showinfo(
            "Registration Successful", 
            f"Voice registration for {name} completed successfully!"
        )

    def update_ui_from_thread(
        self, 
        progress=None, 
        sentence=None, 
        sentence_color="black",
        transcription=None,
        record_button_state=None
    ):
        """
        Thread-safe UI update method
        """
        def update():
            if progress is not None:
                self.progress_label.config(text=progress)
            
            if sentence is not None:
                self.sentence_label.config(
                    text=sentence, 
                    fg=sentence_color
                )
            
            if transcription is not None:
                self.transcription_label.config(text=transcription)
            
            if record_button_state is not None:
                self.record_button.config(state=record_button_state)

        self.frame.after(0, update)

def main():
    root = tk.Tk()
    app = VoiceRegistrationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 