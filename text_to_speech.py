import io
from gtts import gTTS
import pygame

class GoogleTextToSpeech:
    def __init__(self):
        # No credentials needed for gTTS
        pass

    def synthesize_speech(self, text):
        """
        Synthesize speech from text using gTTS and return an audio stream.
        
        Args:
            text (str): Text to be converted to speech.
        
        Returns:
            io.BytesIO: An in-memory audio stream of the synthesized speech.
        """
        if not text:
            print("No text provided for synthesis.")
            return None

        try:
            # Create a gTTS object
            tts = gTTS(text=text, lang='en')
            mp3_fp = io.BytesIO()
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0)  # Rewind the file pointer to the start of the stream
            return mp3_fp
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

def main():
    tts = GoogleTextToSpeech()
    tts.synthesize_speech("Hello, how are you doing today?")

if __name__ == "__main__":
    main() 