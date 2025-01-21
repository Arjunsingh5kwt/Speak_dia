import numpy as np
import scipy.signal as signal
import librosa

class NoiseReductionProcessor:
    def __init__(self, sample_rate=16000):
        """
        Advanced noise reduction and audio preprocessing
        """
        self.sample_rate = sample_rate

    def reduce_noise(self, audio_input):
        """
        Multi-stage noise reduction technique
        """
        # Ensure mono audio
        if audio_input.ndim > 1:
            audio_input = audio_input.mean(axis=1)
        
        # Convert to float
        audio_input = audio_input.astype(np.float32)
        
        # 1. Noise gate
        audio_input = self._noise_gate(audio_input)
        
        # 2. Spectral subtraction
        audio_input = self._spectral_subtraction(audio_input)
        
        # 3. Low-pass filtering
        audio_input = self._low_pass_filter(audio_input)
        
        # 4. Normalize
        audio_input = librosa.util.normalize(audio_input)
        
        return audio_input

    def _noise_gate(self, audio_input, threshold=0.01):
        """
        Remove low-energy noise
        """
        return np.where(np.abs(audio_input) > threshold, audio_input, 0)

    def _spectral_subtraction(self, audio_input, noise_profile_duration=0.5):
        """
        Spectral subtraction for noise reduction
        """
        # Estimate noise profile from the beginning of the audio
        noise_samples = int(noise_profile_duration * self.sample_rate)
        noise_profile = audio_input[:noise_samples]
        
        # Compute FFT
        audio_fft = np.fft.rfft(audio_input)
        noise_fft = np.fft.rfft(noise_profile)
        
        # Estimate noise magnitude
        noise_mag = np.abs(noise_fft)
        audio_mag = np.abs(audio_fft)
        
        # Spectral subtraction
        enhanced_mag = np.maximum(audio_mag - noise_mag, 0)
        
        # Reconstruct audio
        enhanced_fft = enhanced_mag * np.exp(1j * np.angle(audio_fft))
        return np.fft.irfft(enhanced_fft)

    def _low_pass_filter(self, audio_input, cutoff=3000):
        """
        Apply low-pass filter to remove high-frequency noise
        """
        nyquist = 0.5 * self.sample_rate
        normal_cutoff = cutoff / nyquist
        b, a = signal.butter(10, normal_cutoff, btype='low', analog=False)
        return signal.filtfilt(b, a, audio_input) 