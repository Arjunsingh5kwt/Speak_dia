import os
import sys
import warnings

def configure_audio_backend():
    """
    Configure audio backend for torchaudio with Windows compatibility
    """
    try:
        import torchaudio
        
        # Suppress specific warnings
        warnings.filterwarnings('ignore', category=UserWarning)
        
        # Check operating system
        if sys.platform == 'win32':
            # Explicitly set backend for Windows
            try:
                torchaudio.set_audio_backend('soundfile')
            except Exception as e:
                print(f"Error setting audio backend: {e}")
        
        return True
    
    except ImportError:
        print("torchaudio not installed. Audio processing may be limited.")
        return False

# Call this function early in your main script or application initialization
configure_audio_backend() 