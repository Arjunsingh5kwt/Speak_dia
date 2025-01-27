import os
import sys
import warnings
import platform

def setup_audio_environment():
    """
    Comprehensive audio environment setup
    """
    # System-specific audio backend configuration
    system = platform.system()
    
    if system == 'Windows':
        # Windows-specific configurations
        os.environ['TORCHAUDIO_BACKEND'] = 'soundfile'
    
    elif system == 'Linux':
        # Linux-specific configurations
        os.environ['TORCHAUDIO_BACKEND'] = 'sox_io'
    
    elif system == 'Darwin':  # macOS
        # macOS-specific configurations
        os.environ['TORCHAUDIO_BACKEND'] = 'sox_io'
    
    # Suppress specific warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    try:
        import torchaudio
        
        # Additional torchaudio configuration
        torchaudio.set_audio_backend(
            os.environ.get('TORCHAUDIO_BACKEND', 'soundfile')
        )
    
    except ImportError:
        print("torchaudio not installed. Audio processing may be limited.")

# Call this function at the start of your main application
setup_audio_environment() 