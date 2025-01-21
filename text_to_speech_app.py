import tkinter as tk
from tkinter import scrolledtext, messagebox
import pygame
from gtts import gTTS
import tempfile
import os
import traceback
import uuid

class TextToSpeechApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Text to Speech App")
        self.root.geometry("500x600")  # Set a default window size

        # Text input area
        tk.Label(root, text="Enter Text to Convert to Speech:").pack(pady=(10,0))
        self.text_input = scrolledtext.ScrolledText(root, height=10, width=50, wrap=tk.WORD)
        self.text_input.pack(pady=10, padx=10)

        # Button frame
        button_frame = tk.Frame(root)
        button_frame.pack(pady=10)

        # Buttons
        buttons = [
            ("Convert to Speech", self.convert_to_speech),
            ("Replay Audio", self.replay_audio),
            ("Clear Text", self.clear_text),
            ("Exit", self.root.quit)
        ]

        for (text, command) in buttons:
            btn = tk.Button(button_frame, text=text, command=command)
            btn.pack(side=tk.LEFT, padx=5)

        # Log area
        tk.Label(root, text="Conversion Log:").pack(pady=(10,0))
        self.log = scrolledtext.ScrolledText(root, height=10, width=50, state='disabled', wrap=tk.WORD)
        self.log.pack(pady=10, padx=10)

        # Initialize pygame for audio playback
        pygame.mixer.init()

        # Temporary file for audio
        self.temp_audio_file = None

    def convert_to_speech(self):
        text = self.text_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showerror("Error", "Please enter some text to convert!")
            return

        try:
            # Clean up previous audio file if it exists
            self.cleanup_temp_audio()

            # Create a unique temporary file with a full path
            temp_dir = tempfile.gettempdir()
            audio_filename = f"tts_{uuid.uuid4().hex}.mp3"
            audio_filepath = os.path.join(temp_dir, audio_filename)

            # Generate speech and save to the specific filepath
            tts = gTTS(text=text, lang='en')
            
            # Attempt to save with multiple tries
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    tts.save(audio_filepath)
                    break
                except PermissionError:
                    if attempt < max_attempts - 1:
                        # Wait a bit and try again
                        import time
                        time.sleep(0.5)
                    else:
                        raise

            # Verify file was created
            if not os.path.exists(audio_filepath):
                raise IOError("Failed to create audio file")

            # Store the filepath for potential replay
            self.temp_audio_file = audio_filepath

            # Play the audio
            self.play_audio(audio_filepath)

            # Log the conversion
            self.log_conversion(text, audio_filepath)

        except Exception as e:
            error_msg = f"Failed to convert text to speech: {str(e)}\n{traceback.format_exc()}"
            messagebox.showerror("Error", error_msg)
            print(error_msg)  # Also print to console for debugging

    def cleanup_temp_audio(self):
        """
        Attempt to clean up the previous temporary audio file
        """
        if self.temp_audio_file and os.path.exists(self.temp_audio_file):
            try:
                # Stop any ongoing music playback
                pygame.mixer.music.stop()
                pygame.mixer.music.unload()

                # Try to remove the file
                os.remove(self.temp_audio_file)
                self.temp_audio_file = None
            except Exception as e:
                print(f"Error cleaning up temp audio file: {e}")

    def play_audio(self, file_path):
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Audio file not found: {file_path}")

            # Stop any currently playing music
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()

            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()

        except Exception as e:
            messagebox.showerror("Playback Error", f"Could not play audio: {str(e)}")
            print(f"Playback error: {traceback.format_exc()}")

    def replay_audio(self):
        if self.temp_audio_file and os.path.exists(self.temp_audio_file):
            self.play_audio(self.temp_audio_file)
        else:
            messagebox.showinfo("Info", "No audio to replay. Please convert some text first.")

    def log_conversion(self, text, file_path):
        self.log.config(state='normal')
        log_entry = f"Converted: {text[:50]}...\nAudio File: {file_path}\n\n"
        self.log.insert(tk.END, log_entry)
        self.log.config(state='disabled')
        self.log.see(tk.END)  # Scroll to the bottom

    def clear_text(self):
        self.text_input.delete("1.0", tk.END)

    def __del__(self):
        # Clean up temporary audio file when object is deleted
        self.cleanup_temp_audio()

def main():
    root = tk.Tk()
    app = TextToSpeechApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 