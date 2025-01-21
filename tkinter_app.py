import tkinter as tk
from tkinter import ttk, messagebox
import os
import sys

# Import individual application modules
from voice_registration_app import VoiceRegistrationApp
from real_time_diarization_app import RealTimeDiarizationApp
from text_to_speech_app import TextToSpeechApp
from audio_analysis_gui import AudioAnalysisApp

class UnifiedMultimediaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Multimedia Processing Toolkit")
        self.root.geometry("800x600")

        # Create a main frame
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Create a notebook (tabbed interface)
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Create tabs for each application
        self.create_tabs()

        # Add a status bar
        self.status_bar = tk.Label(root, text="Welcome to Multimedia Processing Toolkit", 
                                   bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def create_tabs(self):
        """
        Create tabs for each application module
        """
        # Voice Registration Tab
        voice_reg_frame = tk.Frame(self.notebook)
        self.notebook.add(voice_reg_frame, text="Voice Registration")
        voice_reg_app = VoiceRegistrationApp(voice_reg_frame)

        # Real-Time Diarization Tab
        diarization_frame = tk.Frame(self.notebook)
        self.notebook.add(diarization_frame, text="Real-Time Diarization")
        diarization_app = RealTimeDiarizationApp(diarization_frame)

        # Text-to-Speech Tab
        tts_frame = tk.Frame(self.notebook)
        self.notebook.add(tts_frame, text="Text-to-Speech")
        tts_app = TextToSpeechApp(tts_frame)

        # Audio Analysis Tab
        audio_analysis_frame = tk.Frame(self.notebook)
        self.notebook.add(audio_analysis_frame, text="Audio Analysis")
        audio_analysis_app = AudioAnalysisApp(audio_analysis_frame)

def main():
    root = tk.Tk()
    app = UnifiedMultimediaApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 