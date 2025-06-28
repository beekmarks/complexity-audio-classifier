#!/usr/bin/env python3
"""
Generate Enhanced Realistic Machine Sound Samples

This script generates a larger dataset of realistic machine sound samples
with four degradation levels (healthy, early warning, moderate issue, severe issue).
It creates more nuanced and varied samples to improve classifier training.
"""

import os
import sys
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

# Ensure we can import from parent directory
sys_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys_path not in sys.path:
    sys.path.append(sys_path)


class EnhancedSampleGenerator:
    """
    Generates enhanced realistic machine sound samples with varying degradation levels.
    """
    
    def __init__(self, output_dir='data/realistic_samples', sr=22050, duration=3.0):
        """
        Initialize the sample generator
        
        Args:
            output_dir: Directory to save generated samples
            sr: Sample rate
            duration: Duration of each sample in seconds
        """
        self.output_dir = output_dir
        self.sr = sr
        self.duration = duration
        self.num_samples = int(sr * duration)
        
        # Create output directory structure
        self.categories = ['healthy', 'early_warning', 'moderate_issue', 'severe_issue']
        for category in self.categories:
            os.makedirs(os.path.join(output_dir, category), exist_ok=True)
        
        # Create plots directory
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    
    def generate_base_harmonic_sound(self, fundamental_freq=440.0):
        """
        Generate a base harmonic sound with multiple harmonics
        
        Args:
            fundamental_freq: Fundamental frequency in Hz
            
        Returns:
            numpy array: Generated sound
        """
        t = np.linspace(0, self.duration, self.num_samples, endpoint=False)
        
        # Generate fundamental and harmonics
        harmonics = [1, 2, 3, 4, 5]  # Fundamental and 4 harmonics
        harmonic_weights = [1.0, 0.5, 0.25, 0.125, 0.0625]  # Decreasing amplitude
        
        signal = np.zeros_like(t)
        for h, weight in zip(harmonics, harmonic_weights):
            signal += weight * np.sin(2 * np.pi * fundamental_freq * h * t)
        
        # Normalize
        signal = signal / np.max(np.abs(signal))
        
        return signal
    
    def add_frequency_instability(self, signal, severity=0.0):
        """
        Add frequency instability/wobble to the signal
        
        Args:
            signal: Input signal
            severity: Severity of instability (0.0-1.0)
            
        Returns:
            numpy array: Modified signal
        """
        if severity == 0.0:
            return signal
        
        t = np.linspace(0, self.duration, self.num_samples, endpoint=False)
        
        # Create frequency modulation
        mod_freq = 2.0 + 8.0 * severity  # 2-10 Hz modulation
        mod_depth = 0.005 * severity  # Depth increases with severity
        
        # Apply time-varying phase shift
        phase_mod = mod_depth * np.sin(2 * np.pi * mod_freq * t)
        indices = np.clip(
            np.arange(len(signal)) + (phase_mod * self.sr).astype(int),
            0, len(signal) - 1
        )
        
        return signal[indices]
    
    def add_resonances(self, signal, num_resonances=0, severity=0.0):
        """
        Add resonance peaks to the signal
        
        Args:
            signal: Input signal
            num_resonances: Number of resonance peaks to add
            severity: Severity of resonances (0.0-1.0)
            
        Returns:
            numpy array: Modified signal
        """
        if num_resonances == 0 or severity == 0.0:
            return signal
        
        # Create a copy to avoid modifying the original
        modified = signal.copy()
        
        # Apply resonances in frequency domain
        X = np.fft.rfft(modified)
        freqs = np.fft.rfftfreq(len(modified), 1/self.sr)
        
        for _ in range(num_resonances):
            # Random resonance frequency between 1kHz and 8kHz
            res_freq = np.random.uniform(1000, 8000)
            
            # Create resonance peak
            bandwidth = 50 + 200 * severity  # Wider bandwidth with higher severity
            amplitude = 2.0 + 8.0 * severity  # Higher amplitude with higher severity
            
            # Apply resonance
            peak = amplitude * np.exp(-((freqs - res_freq) ** 2) / (2 * bandwidth ** 2))
            X = X * (1 + peak)
        
        # Convert back to time domain
        modified = np.fft.irfft(X, len(modified))
        
        # Normalize
        modified = modified / np.max(np.abs(modified))
        
        return modified
    
    def add_artifacts(self, signal, artifact_rate=0.0, severity=0.0):
        """
        Add random artifacts/glitches to the signal
        
        Args:
            signal: Input signal
            artifact_rate: Rate of artifacts (0.0-1.0)
            severity: Severity of artifacts (0.0-1.0)
            
        Returns:
            numpy array: Modified signal
        """
        if artifact_rate == 0.0 or severity == 0.0:
            return signal
        
        # Create a copy to avoid modifying the original
        modified = signal.copy()
        
        # Number of artifacts to add
        num_artifacts = int(10 * artifact_rate)
        
        for _ in range(num_artifacts):
            # Random position
            pos = np.random.randint(0, len(modified) - 1000)
            
            # Random artifact duration (10-100ms)
            duration = np.random.randint(int(0.01 * self.sr), int(0.1 * self.sr))
            
            # Create artifact based on severity
            if np.random.random() < 0.5:
                # Amplitude spike
                modified[pos:pos+duration] *= (1 + 5 * severity)
            else:
                # Frequency glitch
                # Ensure glitch has exactly the same length as the slice we're modifying
                actual_duration = min(duration, len(modified) - pos)
                glitch = np.sin(2 * np.pi * np.random.uniform(500, 2000) * 
                               np.arange(actual_duration) / self.sr)
                modified[pos:pos+actual_duration] = (1 - severity) * modified[pos:pos+actual_duration] + severity * glitch
        
        # Normalize
        modified = modified / np.max(np.abs(modified))
        
        return modified
    
    def add_noise(self, signal, noise_level=0.0):
        """
        Add background noise to the signal
        
        Args:
            signal: Input signal
            noise_level: Level of noise (0.0-1.0)
            
        Returns:
            numpy array: Modified signal
        """
        if noise_level == 0.0:
            return signal
        
        # Generate white noise
        noise = np.random.normal(0, noise_level, len(signal))
        
        # Mix with original signal
        mixed = (1.0 - noise_level) * signal + noise
        
        # Normalize
        mixed = mixed / np.max(np.abs(mixed))
        
        return mixed
    
    def generate_sample(self, machine_id, category):
        """
        Generate a single machine sound sample with specified degradation level
        
        Args:
            machine_id: ID of the machine (for naming)
            category: Degradation category
            
        Returns:
            tuple: (audio_data, sample_rate, file_path)
        """
        # Base parameters
        fundamental_freq = np.random.uniform(200, 600)  # Base frequency varies by machine
        
        # Generate base sound
        signal = self.generate_base_harmonic_sound(fundamental_freq)
        
        # Apply degradation based on category
        if category == 'healthy':
            # Healthy: Minimal degradation
            signal = self.add_frequency_instability(signal, severity=0.05)
            signal = self.add_resonances(signal, num_resonances=1, severity=0.1)
            signal = self.add_artifacts(signal, artifact_rate=0.01, severity=0.1)
            signal = self.add_noise(signal, noise_level=0.01)
            
        elif category == 'early_warning':
            # Early Warning: Slight degradation
            signal = self.add_frequency_instability(signal, severity=0.2)
            signal = self.add_resonances(signal, num_resonances=2, severity=0.3)
            signal = self.add_artifacts(signal, artifact_rate=0.05, severity=0.2)
            signal = self.add_noise(signal, noise_level=0.05)
            
        elif category == 'moderate_issue':
            # Moderate Issue: Noticeable degradation
            signal = self.add_frequency_instability(signal, severity=0.5)
            signal = self.add_resonances(signal, num_resonances=3, severity=0.6)
            signal = self.add_artifacts(signal, artifact_rate=0.2, severity=0.5)
            signal = self.add_noise(signal, noise_level=0.1)
            
        elif category == 'severe_issue':
            # Severe Issue: Significant degradation
            signal = self.add_frequency_instability(signal, severity=0.9)
            signal = self.add_resonances(signal, num_resonances=5, severity=0.9)
            signal = self.add_artifacts(signal, artifact_rate=0.4, severity=0.8)
            signal = self.add_noise(signal, noise_level=0.2)
        
        # Add some randomness to make each sample unique
        variation = np.random.uniform(0.95, 1.05, len(signal))
        signal = signal * variation
        
        # Normalize final signal
        signal = signal / np.max(np.abs(signal))
        
        # Create file path
        file_name = f"machine_{machine_id}_{category}.wav"
        file_path = os.path.join(self.output_dir, category, file_name)
        
        return signal, self.sr, file_path
    
    def generate_dataset(self, num_machines=20, visualize=True):
        """
        Generate a complete dataset of machine sounds
        
        Args:
            num_machines: Number of different machines to simulate
            visualize: Whether to create visualization plots
        """
        print(f"Generating enhanced dataset with {num_machines} machines...")
        
        # Generate samples for each machine and category
        for machine_id in tqdm(range(1, num_machines + 1)):
            for category in self.categories:
                signal, sr, file_path = self.generate_sample(machine_id, category)
                
                # Save audio file
                sf.write(file_path, signal, sr)
        
        print(f"Generated {num_machines * len(self.categories)} samples")
        
        # Create visualizations
        if visualize:
            self.create_visualizations()
    
    def create_visualizations(self):
        """Create visualizations of sample waveforms and spectrograms"""
        print("Creating visualizations...")
        
        # Select one sample from each category for visualization
        plt.figure(figsize=(15, 10))
        
        for i, category in enumerate(self.categories):
            # Get first file in category
            files = os.listdir(os.path.join(self.output_dir, category))
            wav_files = [f for f in files if f.endswith('.wav')]
            if not wav_files:
                continue
                
            file_path = os.path.join(self.output_dir, category, wav_files[0])
            
            # Load audio
            y, sr = librosa.load(file_path, sr=None)
            
            # Plot waveform
            plt.subplot(4, 2, i*2 + 1)
            plt.title(f"{category} - Waveform")
            plt.plot(y)
            plt.ylim(-1, 1)
            
            # Plot spectrogram
            plt.subplot(4, 2, i*2 + 2)
            plt.title(f"{category} - Spectrogram")
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
            plt.colorbar(format='%+2.0f dB')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'plots', 'sample_visualizations.png'))
        plt.close()


def main():
    """Main function to run the sample generator"""
    parser = argparse.ArgumentParser(description='Generate enhanced machine sound samples')
    parser.add_argument('--output-dir', type=str, default='data/realistic_samples',
                        help='Directory to save generated samples')
    parser.add_argument('--num-machines', type=int, default=20,
                        help='Number of different machines to simulate')
    parser.add_argument('--no-visualize', action='store_false', dest='visualize',
                        help='Disable visualization generation')
    
    args = parser.parse_args()
    
    # Create and run the generator
    generator = EnhancedSampleGenerator(output_dir=args.output_dir)
    generator.generate_dataset(num_machines=args.num_machines, visualize=args.visualize)


if __name__ == "__main__":
    main()
