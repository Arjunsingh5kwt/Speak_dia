import os
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import threading
import logging

# Import project modules
from audio_analysis import AudioAnalyzer

class AudioAnalysisApp:
    def __init__(self, parent_frame):
        """
        Initialize Audio Analysis Tkinter Application
        """
        self.parent = parent_frame
        
        # Create a frame within the parent frame
        self.frame = tk.Frame(self.parent)
        self.frame.pack(fill=tk.BOTH, expand=True)
        
        # Configure logging
        self.setup_logging()
        
        # Initialize AudioAnalyzer
        self.audio_analyzer = AudioAnalyzer()
        
        # Selected audio file path
        self.selected_audio_path = None
        
        # Create UI components
        self.setup_ui()

    def setup_logging(self):
        """
        Configure logging for the application
        """
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('audio_analysis_gui.log', mode='w')
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_ui(self):
        """
        Create the user interface components
        """
        # Title Label
        title_label = tk.Label(self.frame, text="Audio Analysis", font=("Helvetica", 16))
        title_label.pack(pady=10)

        # Instructions
        instructions = tk.Label(
            self.frame, 
            text="Select an audio file to automatically analyze transcription, language, and speakers",
            font=("Arial", 10)
        )
        instructions.pack(pady=5)

        # File Selection Frame
        file_frame = tk.Frame(self.frame)
        file_frame.pack(pady=10)

        # File Path Display
        self.file_path_var = tk.StringVar()
        self.file_path_entry = tk.Entry(
            file_frame, 
            textvariable=self.file_path_var, 
            width=50
        )
        self.file_path_entry.pack(side=tk.LEFT, padx=5)

        # Browse Button
        browse_button = tk.Button(
            file_frame, 
            text="Browse", 
            command=self.browse_audio_file
        )
        browse_button.pack(side=tk.LEFT)

        # Bind Enter key to start analysis
        self.file_path_entry.bind('<Return>', self.auto_start_analysis)

        # Analysis Status Label
        self.status_label = tk.Label(
            self.frame, 
            text="", 
            font=("Arial", 10)
        )
        self.status_label.pack(pady=5)

        # Results Display
        results_label = tk.Label(
            self.frame, 
            text="Analysis Results:", 
            font=("Arial", 12, "bold")
        )
        results_label.pack(pady=5)

        self.results_text = scrolledtext.ScrolledText(
            self.frame, 
            wrap=tk.WORD, 
            width=80, 
            height=20
        )
        self.results_text.pack(pady=10)

        # Save Results Button
        self.save_button = tk.Button(
            self.frame, 
            text="Save Results", 
            command=self.save_results,
            state=tk.DISABLED
        )
        self.save_button.pack(pady=5)

    def browse_audio_file(self):
        """
        Open file dialog to select audio file
        """
        try:
            # Supported audio file types
            file_types = [
                ('Audio Files', '*.wav *.mp3 *.ogg *.flac'),
                ('WAV Files', '*.wav'),
                ('MP3 Files', '*.mp3'),
                ('All Files', '*.*')
            ]
            
            # Open file dialog
            audio_path = filedialog.askopenfilename(
                title="Select Audio File",
                filetypes=file_types
            )
            
            # Update file path
            if audio_path:
                self.selected_audio_path = audio_path
                self.file_path_var.set(audio_path)
                
                # Automatically start analysis
                self.start_analysis()
                
                # Log file selection
                self.logger.info(f"Audio file selected: {audio_path}")
        
        except Exception as e:
            self.logger.error(f"File selection error: {e}")
            messagebox.showerror("File Selection Error", str(e))

    def auto_start_analysis(self, event=None):
        """
        Start analysis from file path entry
        """
        # Get path from entry
        audio_path = self.file_path_var.get().strip()
        
        # Validate file path
        if os.path.exists(audio_path):
            self.selected_audio_path = audio_path
            self.start_analysis()
        else:
            messagebox.showwarning("Invalid Path", "Please select a valid audio file.")

    def start_analysis(self):
        """
        Start audio analysis in a separate thread
        """
        try:
            # Validate file selection
            if not self.selected_audio_path:
                messagebox.showwarning("Warning", "Please select an audio file first.")
                return

            # Update status
            self.status_label.config(text="Analyzing...", fg="blue")
            
            # Disable save button
            self.save_button.config(state=tk.DISABLED)
            
            # Clear previous results
            self.results_text.delete('1.0', tk.END)
            
            # Start analysis in a separate thread
            threading.Thread(
                target=self.perform_analysis, 
                daemon=True
            ).start()
        
        except Exception as e:
            self.logger.error(f"Analysis start error: {e}")
            messagebox.showerror("Analysis Error", str(e))

    def perform_analysis(self):
        """
        Perform comprehensive audio analysis
        """
        try:
            # Log analysis start
            self.logger.info(f"Starting analysis for {self.selected_audio_path}")
            
            # Perform audio analysis
            analysis_results = self.audio_analyzer.analyze_audio(
                self.selected_audio_path
            )
            
            # Generate summary
            summary = self.audio_analyzer.generate_summary(analysis_results)
            
            # Update UI with results
            self.update_results_ui(summary, analysis_results)
        
        except Exception as e:
            self.logger.error(f"Analysis error: {e}")
            self.show_error_message(str(e))

    def update_results_ui(self, summary, full_results):
        """
        Update UI with analysis results
        """
        def update():
            try:
                # Display summary
                self.results_text.insert(tk.END, summary)
                
                # Update status
                self.status_label.config(text="Analysis Complete", fg="green")
                
                # Enable save button
                self.save_button.config(state=tk.NORMAL)
                
                # Log successful analysis
                self.logger.info("Audio analysis completed successfully")
            
            except Exception as e:
                self.logger.error(f"Results UI update error: {e}")
        
        # Use after method to update UI from thread
        self.frame.after(0, update)

    def save_results(self):
        """
        Save analysis results to a text file
        """
        try:
            # Open save file dialog
            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text Files", "*.txt")]
            )
            
            # Save if file path selected
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.results_text.get('1.0', tk.END))
                
                # Show success message
                messagebox.showinfo(
                    "Save Successful", 
                    f"Results saved to {file_path}"
                )
                
                # Log file save
                self.logger.info(f"Results saved to {file_path}")
        
        except Exception as e:
            self.logger.error(f"File save error: {e}")
            messagebox.showerror("Save Error", str(e))

    def show_error_message(self, error_message):
        """
        Display error message in UI
        """
        def show_error():
            # Clear previous results
            self.results_text.delete('1.0', tk.END)
            
            # Update status
            self.status_label.config(text="Analysis Failed", fg="red")
            
            # Insert error message
            self.results_text.insert(
                tk.END, 
                f"Analysis Error:\n{error_message}"
            )
        
        # Use after method to show error from thread
        self.frame.after(0, show_error)

def main():
    """
    Main application entry point
    """
    root = tk.Tk()
    app = AudioAnalysisApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 