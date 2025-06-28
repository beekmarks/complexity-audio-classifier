#!/usr/bin/env python3
"""
Adaptive Hybrid Complexity Filter

This script implements an improved version of the hybrid complexity filter that uses
adaptive thresholds based on statistical analysis of the dataset. It leverages both
BSG and RBS algorithms as fast first-pass filters to determine when to trigger more
advanced audio feature extraction and classification.

Building on previous findings that BSG assigns higher structural complexity to ordered
signals with clear spectral patterns than to chaotic signals.
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import librosa
import librosa.display
import seaborn as sns
import soundfile as sf
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_curve, auc
from scipy.ndimage import uniform_filter

# Import BSG and RBS algorithms
sys_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys_path not in sys.path:
    sys.path.append(sys_path)

from core.binary_split_game import extract_bsg_features
from core.recursive_bilateral_symmetry import detect_rbs, LCE_Processor

# Constants
SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 512
BINARIZE_THRESHOLD = 0.5  # Threshold for binarization

# Default complexity thresholds (will be overridden by adaptive thresholds)
DEFAULT_BSG_THRESHOLD = 1800
DEFAULT_RBS_THRESHOLD = 3
DEFAULT_NORM_TC_THRESHOLD = 0.4  # Normalized transition count threshold

def save_binary_representation(binary_spectrogram, file_path):
    """
    Save binary representation of a spectrogram to a text file.
    """
    # Create binary_data directory if it doesn't exist
    os.makedirs('data/binary_data', exist_ok=True)
    
    # Extract filename without extension
    base_name = os.path.basename(file_path).split('.')[0]
    output_path = f'data/binary_data/{base_name}_binary.txt'
    
    # Save binary representation to file
    with open(output_path, 'w') as f:
        # Add header with information
        f.write(f"Binary representation of spectrogram for {base_name}\n")
        f.write(f"Shape: {binary_spectrogram.shape}\n")
        f.write("=" * 80 + "\n\n")
        
        # Save central rows of the binary spectrogram (more informative)
        start_row = binary_spectrogram.shape[0] // 4
        end_row = 3 * binary_spectrogram.shape[0] // 4
        
        f.write(f"Central rows ({start_row} to {end_row}) of binary spectrogram:\n\n")
        
        for i in range(start_row, end_row, 5):  # Save every 5th row to keep file size reasonable
            row_str = ''.join(map(str, binary_spectrogram[i].astype(int)))
            f.write(f"Row {i}: {row_str}\n")
        
        # Also save a flattened version (used for complexity analysis)
        f.write("\n" + "=" * 80 + "\n")
        f.write("Flattened binary string (used for complexity analysis):\n")
        flattened = binary_spectrogram.flatten()
        flattened_str = ''.join(map(str, flattened.astype(int)))
        
        # Write in chunks to avoid very long lines
        chunk_size = 100
        for i in range(0, len(flattened_str), chunk_size):
            f.write(flattened_str[i:i+chunk_size] + "\n")
    
    print(f"Saved binary representation to {output_path}")
    return output_path


class AdaptiveComplexityFilter:
    """
    Implements an adaptive hybrid complexity filter that uses BSG and RBS as
    fast first-pass filters before triggering advanced audio analysis.
    """
    
    def __init__(self, data_dir='data', calibration_size=5):
        """Initialize the adaptive complexity filter with data directory."""
        self.data_dir = data_dir
        self.calibration_size = calibration_size
        self.thresholds = {
            'bsg_complexity': DEFAULT_BSG_THRESHOLD,
            'max_rbs_order': DEFAULT_RBS_THRESHOLD,
            'normalized_tc': DEFAULT_NORM_TC_THRESHOLD
        }
        self.primary_metric = 'normalized_tc'  # Default primary metric
        self.secondary_metric = 'bsg_complexity'  # Default secondary metric
        self.calibrated = False
        self.results = []
        self.processing_times = []
        self.healthy_metrics = []
        self.failing_metrics = []
        self.classifier = None
        
    def load_audio(self, file_path):
        """Load audio file and return the signal."""
        audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        return audio
    
    def compute_spectrogram(self, audio):
        """
        Compute the spectrogram of an audio signal.
        """
        # Compute spectrogram with more frequency resolution
        n_fft = 2048  # Increased from default for better frequency resolution
        hop_length = 512  # Standard hop length
        spectrogram = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        
        # Convert to magnitude spectrogram and apply log scaling
        spectrogram = np.abs(spectrogram)
        spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)
        
        return spectrogram
    
    def binarize_spectrogram(self, spectrogram):
        """
        Binarize a spectrogram using adaptive thresholding.
        """
        # Normalize the spectrogram to 0-1 range
        spec_norm = (spectrogram - np.min(spectrogram)) / (np.max(spectrogram) - np.min(spectrogram))
        
        # Apply different thresholding methods and combine them for better discrimination
        # 1. Global threshold at median
        global_threshold = np.median(spec_norm)
        binary_global = (spec_norm > global_threshold).astype(int)
        
        # 2. Local adaptive threshold (compare each value to local neighborhood mean)
        from scipy.ndimage import uniform_filter
        local_mean = uniform_filter(spec_norm, size=5)
        binary_local = (spec_norm > local_mean - 0.05).astype(int)
        
        # 3. Frequency-dependent threshold (different for each frequency band)
        freq_threshold = np.median(spec_norm, axis=1, keepdims=True) * 0.8
        binary_freq = (spec_norm > freq_threshold).astype(int)
        
        # Combine the binary spectrograms (logical OR)
        binary_combined = np.logical_or(np.logical_or(binary_global, binary_local), binary_freq).astype(int)
        
        # Extract a subset for faster processing (central region with most information)
        height, width = binary_combined.shape
        start_h, end_h = height // 4, 3 * height // 4
        start_w, end_w = 0, min(width, 100)  # Take first 100 time frames or all if less
        
        binary_subset = binary_combined[start_h:end_h, start_w:end_w]
        
        return binary_subset
    
    def extract_quick_complexity_metrics(self, binary_spectrogram):
        """
        Extract quick complexity metrics from a binary spectrogram.
        """
        # Extract multiple binary strings from different rows of the spectrogram
        # This captures frequency-specific patterns better than flattening
        rows, cols = binary_spectrogram.shape
        
        # Sample rows from different frequency regions
        row_indices = [rows//4, rows//2, 3*rows//4]  # Low, mid, and high frequencies
        binary_strings = []
        for idx in row_indices:
            binary_strings.append(''.join(map(str, binary_spectrogram[idx, :])))
        
        # Also include a flattened version for overall pattern analysis
        flattened = ''.join(map(str, binary_spectrogram.flatten()))
        binary_strings.append(flattened)
        
        # Calculate metrics for each binary string
        bsg_complexities = []
        max_rbs_orders = []
        avg_rbs_orders = []
        hamming_weights = []
        transition_counts = []
        normalized_tcs = []
        
        for binary_string in binary_strings:
            # Calculate BSG complexity
            bsg_complexity = self.calculate_bsg_complexity(binary_string)
            bsg_complexities.append(bsg_complexity)
            
            # Calculate RBS metrics with timeout
            max_rbs, avg_rbs = self.calculate_rbs_metrics(binary_string)
            max_rbs_orders.append(max_rbs)
            avg_rbs_orders.append(avg_rbs)
            
            # Calculate Hamming weight (number of 1s)
            hamming = binary_string.count('1')
            hamming_weights.append(hamming)
            
            # Calculate transition count
            transitions = sum(1 for i in range(len(binary_string)-1) if binary_string[i] != binary_string[i+1])
            transition_counts.append(transitions)
            
            # Normalize transition count
            norm_tc = transitions / (len(binary_string) - 1) if len(binary_string) > 1 else 0
            normalized_tcs.append(norm_tc)
        
        # Calculate spectral pattern metrics (based on your previous BSG findings)
        # For ordered signals (like healthy machine sounds), BSG complexity tends to be higher
        # Calculate the ratio of BSG complexity to string length (complexity density)
        complexity_densities = [bsg / len(binary_strings[i]) for i, bsg in enumerate(bsg_complexities)]
        
        # Calculate spectral entropy (lower for ordered signals with clear patterns)
        spectral_entropy = self.calculate_spectral_entropy(binary_spectrogram)
        
        # Calculate pattern consistency across frequency bands
        # (higher for ordered signals with consistent patterns)
        pattern_consistency = self.calculate_pattern_consistency(binary_spectrogram)
        
        # Combine metrics
        complexity_metrics = {
            'bsg_complexity': np.mean(bsg_complexities),  # Average BSG complexity across samples
            'max_bsg_complexity': np.max(bsg_complexities),  # Maximum BSG complexity
            'complexity_density': np.mean(complexity_densities),  # Average complexity density
            'max_rbs_order': np.max(max_rbs_orders),  # Maximum RBS order
            'avg_rbs_order': np.mean(avg_rbs_orders),  # Average RBS order
            'hamming_weight': np.mean(hamming_weights),  # Average Hamming weight
            'transition_count': np.mean(transition_counts),  # Average transition count
            'normalized_tc': np.mean(normalized_tcs),  # Average normalized transition count
            'spectral_entropy': spectral_entropy,  # Spectral entropy
            'pattern_consistency': pattern_consistency  # Pattern consistency
        }
        
        return complexity_metrics
        
    def calculate_bsg_complexity(self, binary_string):
        """
        Calculate BSG complexity of a binary string.
        """
        # Limit string length for faster processing
        max_length = 1000
        if len(binary_string) > max_length:
            binary_string = binary_string[:max_length]
            
        # Calculate BSG features
        try:
            bsg_features = extract_bsg_features(binary_string)
            bsg_complexity = bsg_features.get('sc_total', 0)
            
            # Normalize by string length to make it comparable across different lengths
            normalized_complexity = bsg_complexity / len(binary_string) if len(binary_string) > 0 else 0
            return normalized_complexity * 1000  # Scale for better readability
        except Exception as e:
            print(f"BSG calculation error: {e}")
            return 0
    
    def calculate_rbs_metrics(self, binary_string):
        """
        Calculate RBS metrics of a binary string with timeout.
        """
        # Limit string length for RBS computation
        max_length = 100  # RBS is computationally expensive, use shorter strings
        if len(binary_string) > max_length:
            binary_string = binary_string[:max_length]
            
        # Use timeout to prevent long-running RBS computation
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("RBS computation timed out")
            
        # Set timeout for RBS computation
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(1)  # 1 second timeout
        
        try:
            rbs_orders = detect_rbs(binary_string, debug=False)
            # Reset alarm
            signal.alarm(0)
            
            # Process RBS orders
            non_negative_orders = [[o for o in row if o >= 0] for row in rbs_orders]
            
            # Calculate max RBS order
            max_orders = [max(row) if row else 0 for row in non_negative_orders]
            max_rbs_order = max(max_orders) if max_orders else 0
            
            # Calculate average RBS order
            all_orders = [o for row in non_negative_orders for o in row if o >= 0]
            avg_rbs_order = np.mean(all_orders) if all_orders else 0
            
            return max_rbs_order, avg_rbs_order
        except (TimeoutError, Exception) as e:
            # Reset alarm in case of exception
            signal.alarm(0)
            return 0, 0
    
    def calculate_spectral_entropy(self, binary_spectrogram):
        """
        Calculate spectral entropy of the binary spectrogram.
        Lower values indicate more ordered patterns.
        """
        # Calculate probability distribution
        flattened = binary_spectrogram.flatten()
        hist, _ = np.histogram(flattened, bins=2, density=True)
        
        # Calculate entropy
        entropy = 0
        for p in hist:
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy
    
    def calculate_pattern_consistency(self, binary_spectrogram):
        """
        Calculate pattern consistency across frequency bands.
        Higher values indicate more consistent patterns across frequencies.
        """
        rows, cols = binary_spectrogram.shape
        
        # Split into frequency bands
        n_bands = 4
        band_size = rows // n_bands
        bands = [binary_spectrogram[i*band_size:(i+1)*band_size, :] for i in range(n_bands)]
        
        # Calculate average pattern for each band
        band_patterns = [np.mean(band, axis=0) for band in bands]
        
        # Calculate correlation between band patterns
        correlations = []
        for i in range(n_bands):
            for j in range(i+1, n_bands):
                corr = np.corrcoef(band_patterns[i], band_patterns[j])[0, 1]
                if not np.isnan(corr):  # Avoid NaN values
                    correlations.append(corr)
        
        # Return average correlation (higher means more consistent patterns)
        return np.mean(correlations) if correlations else 0
    
    def extract_advanced_features(self, audio):
        """
        Extract advanced audio features including MFCCs, spectral and temporal features.
        """
        # Extract Mel-frequency cepstral coefficients (MFCCs)
        mfccs = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=13)
        mfcc_means = np.mean(mfccs, axis=1)
        mfcc_vars = np.var(mfccs, axis=1)
        
        # Extract spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=SAMPLE_RATE)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=SAMPLE_RATE)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=SAMPLE_RATE)[0]
        
        # Calculate statistics for spectral features
        centroid_mean = np.mean(spectral_centroid)
        centroid_var = np.var(spectral_centroid)
        bandwidth_mean = np.mean(spectral_bandwidth)
        bandwidth_var = np.var(spectral_bandwidth)
        rolloff_mean = np.mean(spectral_rolloff)
        rolloff_var = np.var(spectral_rolloff)
        
        # Extract temporal features
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
        zcr_mean = np.mean(zero_crossing_rate)
        zcr_var = np.var(zero_crossing_rate)
        
        # Calculate root mean square energy
        rms = librosa.feature.rms(y=audio)[0]
        rms_mean = np.mean(rms)
        rms_var = np.var(rms)
        
        # Combine all features into a single vector
        features = np.concatenate([
            mfcc_means, mfcc_vars,
            [centroid_mean, centroid_var, bandwidth_mean, bandwidth_var, rolloff_mean, rolloff_var],
            [zcr_mean, zcr_var, rms_mean, rms_var]
        ])
        
        # Create a dictionary with named features
        feature_names = [
            *[f'mfcc_mean_{i}' for i in range(13)],
            *[f'mfcc_var_{i}' for i in range(13)],
            'centroid_mean', 'centroid_var',
            'bandwidth_mean', 'bandwidth_var',
            'rolloff_mean', 'rolloff_var',
            'zcr_mean', 'zcr_var',
            'rms_mean', 'rms_var'
        ]
        
        advanced_features = {name: value for name, value in zip(feature_names, features)}
        
        return advanced_features
    
    def calibrate_thresholds(self):
        """
        Calibrate the complexity thresholds using a small set of training data.
        This determines the optimal thresholds to distinguish between healthy and failing machine sounds.
        """
        print("\n===== Calibrating Complexity Thresholds =====\n")
        
        healthy_dir = os.path.join(self.data_dir, 'healthy')
        failing_dir = os.path.join(self.data_dir, 'failing')
        
        # Get all .wav files
        healthy_files = [os.path.join(healthy_dir, f) for f in os.listdir(healthy_dir) if f.endswith('.wav')]
        failing_files = [os.path.join(failing_dir, f) for f in os.listdir(failing_dir) if f.endswith('.wav')]
        
        # Limit the number of files for calibration
        healthy_files = healthy_files[:self.calibration_size]
        failing_files = failing_files[:self.calibration_size]
        
        print(f"Using {len(healthy_files)} healthy files and {len(failing_files)} failing files for calibration")
        
        # Process healthy files
        healthy_metrics = []
        for i, file_path in enumerate(healthy_files):
            print(f"Processing healthy file {i+1}/{len(healthy_files)}")
            
            # Load and process audio
            audio = self.load_audio(file_path)
            spectrogram = self.compute_spectrogram(audio)
            binary_spectrogram = self.binarize_spectrogram(spectrogram)
            
            # Extract complexity metrics
            metrics = self.extract_quick_complexity_metrics(binary_spectrogram)
            metrics['file'] = os.path.basename(file_path)
            metrics['label'] = 'healthy'
            
            healthy_metrics.append(metrics)
        
        # Process failing files
        failing_metrics = []
        for i, file_path in enumerate(failing_files):
            print(f"Processing failing file {i+1}/{len(failing_files)}")
            
            # Load and process audio
            audio = self.load_audio(file_path)
            spectrogram = self.compute_spectrogram(audio)
            binary_spectrogram = self.binarize_spectrogram(spectrogram)
            
            # Extract complexity metrics
            metrics = self.extract_quick_complexity_metrics(binary_spectrogram)
            metrics['file'] = os.path.basename(file_path)
            metrics['label'] = 'failing'
            
            failing_metrics.append(metrics)
        
        # Store metrics
        self.healthy_metrics = healthy_metrics
        self.failing_metrics = failing_metrics
        all_metrics = healthy_metrics + failing_metrics
        
        # Convert to DataFrame for analysis
        metrics_df = pd.DataFrame(all_metrics)
        
        # Find optimal thresholds for each metric
        metrics_to_analyze = ['bsg_complexity', 'max_rbs_order', 'avg_rbs_order', 
                             'hamming_weight', 'transition_count', 'normalized_tc']
        
        optimal_thresholds = {}
        metric_performance = {}
        
        for metric in metrics_to_analyze:
            # Get values for healthy and failing samples
            healthy_values = metrics_df[metrics_df['label'] == 'healthy'][metric].values
            failing_values = metrics_df[metrics_df['label'] == 'failing'][metric].values
            
            # Calculate mean and std for both classes
            healthy_mean = np.mean(healthy_values)
            healthy_std = np.std(healthy_values)
            failing_mean = np.mean(failing_values)
            failing_std = np.std(failing_values)
            
            # Calculate class separation
            separation = abs(healthy_mean - failing_mean) / ((healthy_std + failing_std) / 2) if (healthy_std + failing_std) > 0 else 0
            
            # Create binary labels (1 for healthy, 0 for failing)
            y_true = (metrics_df['label'] == 'healthy').astype(int).values
            
            # Use the metric values as scores
            scores = metrics_df[metric].values
            
            # If healthy mean is lower than failing mean, invert scores
            if healthy_mean < failing_mean:
                scores = -scores
            
            # Calculate ROC curve and AUC
            fpr, tpr, thresholds = roc_curve(y_true, scores)
            roc_auc = auc(fpr, tpr)
            
            # Find optimal threshold (Youden's J statistic: max(tpr - fpr))
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            
            # If we inverted scores, convert threshold back
            if healthy_mean < failing_mean:
                optimal_threshold = -optimal_threshold
            
            # Calculate accuracy at optimal threshold
            predictions = (scores > optimal_threshold).astype(int) if healthy_mean > failing_mean else (scores < optimal_threshold).astype(int)
            accuracy = np.mean(predictions == y_true)
            
            # Store results
            optimal_thresholds[metric] = optimal_threshold
            metric_performance[metric] = {
                'accuracy': accuracy,
                'auc': roc_auc,
                'separation': separation,
                'healthy_mean': healthy_mean,
                'healthy_std': healthy_std,
                'failing_mean': failing_mean,
                'failing_std': failing_std
            }
            
            print(f"{metric}:")
            print(f"  Threshold: {optimal_threshold:.4f}")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  AUC: {roc_auc:.4f}")
            print(f"  Class Separation: {separation:.4f}")
            print(f"  Healthy: {healthy_mean:.4f} ± {healthy_std:.4f}")
            print(f"  Failing: {failing_mean:.4f} ± {failing_std:.4f}\n")
        
        # Sort metrics by accuracy
        sorted_metrics = sorted(metric_performance.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        # Select the top 2 metrics for the hybrid filter
        self.primary_metric = sorted_metrics[0][0]
        self.secondary_metric = sorted_metrics[1][0]
        
        # Update thresholds
        self.thresholds = {
            self.primary_metric: optimal_thresholds[self.primary_metric],
            self.secondary_metric: optimal_thresholds[self.secondary_metric]
        }
        
        print(f"Selected primary metric: {self.primary_metric} (threshold: {self.thresholds[self.primary_metric]:.4f})")
        print(f"Selected secondary metric: {self.secondary_metric} (threshold: {self.thresholds[self.secondary_metric]:.4f})")
        
        # Train a classifier on the complexity metrics
        X = metrics_df[metrics_to_analyze].values
        y = (metrics_df['label'] == 'healthy').astype(int).values
        
        # Train a Random Forest classifier
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.classifier.fit(X, y)
        
        # Evaluate classifier
        y_pred = self.classifier.predict(X)
        accuracy = accuracy_score(y, y_pred)
        print(f"Classifier accuracy on calibration data: {accuracy:.4f}")
        
        # Mark as calibrated
        self.calibrated = True
        
        return self.thresholds
    
    def process_file(self, file_path, label=None):
        """
        Process a single audio file using the adaptive hybrid complexity filter.
        
        Args:
            file_path: Path to the audio file
            label: Optional label ('healthy' or 'failing')
            
        Returns:
            Dictionary with processing results
        """
        print(f"\nProcessing file: {os.path.basename(file_path)}")
        
        # Initialize timing
        start_time = time.time()
        quick_assessment_time = 0
        advanced_analysis_time = 0
        
        # Load audio
        audio = self.load_audio(file_path)
        
        # Quick complexity assessment
        quick_start_time = time.time()
        
        # Compute spectrogram
        spectrogram = self.compute_spectrogram(audio)
        
        # Binarize spectrogram
        binary_spectrogram = self.binarize_spectrogram(spectrogram)
        
        # Extract complexity metrics
        complexity_metrics = self.extract_quick_complexity_metrics(binary_spectrogram)
        
        quick_assessment_time = time.time() - quick_start_time
        
        # Determine if advanced analysis is needed
        needs_advanced_analysis = False
        trigger_reason = None
        
        # Check if we've calibrated the thresholds
        if not self.calibrated:
            print("Thresholds not calibrated, using default thresholds")
            # Use default thresholds for primary and secondary metrics
            primary_value = complexity_metrics[self.primary_metric]
            secondary_value = complexity_metrics[self.secondary_metric]
            
            # For normalized_tc, lower values are more ordered (healthy)
            if self.primary_metric == 'normalized_tc':
                if primary_value < self.thresholds[self.primary_metric]:
                    needs_advanced_analysis = True
                    trigger_reason = f"{self.primary_metric} below threshold"
            # For BSG complexity, higher values are more ordered (healthy)
            elif primary_value > self.thresholds[self.primary_metric]:
                needs_advanced_analysis = True
                trigger_reason = f"{self.primary_metric} above threshold"
            
            # Check secondary metric if primary didn't trigger
            if not needs_advanced_analysis:
                if self.secondary_metric == 'normalized_tc':
                    if secondary_value < self.thresholds[self.secondary_metric]:
                        needs_advanced_analysis = True
                        trigger_reason = f"{self.secondary_metric} below threshold"
                elif secondary_value > self.thresholds[self.secondary_metric]:
                    needs_advanced_analysis = True
                    trigger_reason = f"{self.secondary_metric} above threshold"
        else:
            # Use classifier for decision if calibrated
            features = [complexity_metrics[m] for m in ['bsg_complexity', 'max_rbs_order', 'avg_rbs_order', 
                                                      'hamming_weight', 'transition_count', 'normalized_tc']]
            prediction = self.classifier.predict([features])[0]
            prediction_proba = self.classifier.predict_proba([features])[0]
            
            # If predicted as healthy or prediction confidence is low, trigger advanced analysis
            if prediction == 1 or max(prediction_proba) < 0.7:
                needs_advanced_analysis = True
                trigger_reason = f"Classifier prediction: {'healthy' if prediction == 1 else 'failing'} ({max(prediction_proba):.2f} confidence)"
        
        # Advanced analysis if needed
        advanced_features = None
        if needs_advanced_analysis:
            print(f"Triggering advanced analysis: {trigger_reason}")
            advanced_start_time = time.time()
            
            # Extract advanced features
            advanced_features = self.extract_advanced_features(audio)
            
            advanced_analysis_time = time.time() - advanced_start_time
        else:
            print("Quick assessment sufficient, skipping advanced analysis")
        
        # Calculate total processing time
        total_time = time.time() - start_time
        
        # Prepare result
        result = {
            'file': os.path.basename(file_path),
            'label': label,
            'quick_assessment_time': quick_assessment_time,
            'advanced_analysis_time': advanced_analysis_time,
            'total_time': total_time,
            'needs_advanced_analysis': needs_advanced_analysis,
            'trigger_reason': trigger_reason,
            **complexity_metrics
        }
        
        # Add advanced features if available
        if advanced_features:
            result.update(advanced_features)
        
        # Print complexity metrics
        print(f"BSG Complexity: {complexity_metrics['bsg_complexity']:.2f}")
        print(f"Max RBS Order: {complexity_metrics['max_rbs_order']}")
        print(f"Normalized TC: {complexity_metrics['normalized_tc']:.4f}")
        print(f"Quick assessment time: {quick_assessment_time:.4f} seconds")
        
        if needs_advanced_analysis:
            print(f"Advanced analysis time: {advanced_analysis_time:.4f} seconds")
        
        print(f"Total processing time: {total_time:.4f} seconds")
        
        return result
    
    def process_dataset(self, max_files=None):
        """
        Process all audio files in the dataset.
        
        Args:
            max_files: Maximum number of files to process from each class
            
        Returns:
            List of processing results
        """
        # Calibrate thresholds if not already done
        if not self.calibrated:
            self.calibrate_thresholds()
        
        healthy_dir = os.path.join(self.data_dir, 'healthy')
        failing_dir = os.path.join(self.data_dir, 'failing')
        
        # Get all .wav files
        healthy_files = [os.path.join(healthy_dir, f) for f in os.listdir(healthy_dir) if f.endswith('.wav')]
        failing_files = [os.path.join(failing_dir, f) for f in os.listdir(failing_dir) if f.endswith('.wav')]
        
        # Limit the number of files if specified
        if max_files:
            healthy_files = healthy_files[:max_files]
            failing_files = failing_files[:max_files]
        
        print(f"\nProcessing {len(healthy_files)} healthy files and {len(failing_files)} failing files")
        
        # Process healthy files
        healthy_results = []
        for i, file_path in enumerate(healthy_files):
            print(f"\nProcessing healthy file {i+1}/{len(healthy_files)}")
            result = self.process_file(file_path, 'healthy')
            healthy_results.append(result)
        
        # Process failing files
        failing_results = []
        for i, file_path in enumerate(failing_files):
            print(f"\nProcessing failing file {i+1}/{len(failing_files)}")
            result = self.process_file(file_path, 'failing')
            failing_results.append(result)
        
        # Combine results
        all_results = healthy_results + failing_results
        self.results = all_results
        
        return all_results
    
    def visualize_results(self):
        """
        Visualize the results of the adaptive hybrid complexity filter.
        """
        if not self.results:
            print("No results to visualize. Run process_dataset() first.")
            return
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(self.results)
        
        # Create figure for complexity metrics
        plt.figure(figsize=(12, 10))
        
        # Plot BSG complexity
        plt.subplot(2, 2, 1)
        sns.boxplot(x='label', y='bsg_complexity', data=results_df)
        plt.title('BSG Complexity by Class')
        plt.xlabel('Class')
        plt.ylabel('BSG Complexity')
        
        # Plot max RBS order
        plt.subplot(2, 2, 2)
        sns.boxplot(x='label', y='max_rbs_order', data=results_df)
        plt.title('Max RBS Order by Class')
        plt.xlabel('Class')
        plt.ylabel('Max RBS Order')
        
        # Plot normalized transition count
        plt.subplot(2, 2, 3)
        sns.boxplot(x='label', y='normalized_tc', data=results_df)
        plt.title('Normalized Transition Count by Class')
        plt.xlabel('Class')
        plt.ylabel('Normalized TC')
        
        # Plot advanced analysis trigger rate
        plt.subplot(2, 2, 4)
        trigger_counts = results_df.groupby(['label', 'needs_advanced_analysis']).size().unstack(fill_value=0)
        trigger_counts.plot(kind='bar', stacked=True)
        plt.title('Advanced Analysis Trigger Rate by Class')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.legend(['No Advanced Analysis', 'Advanced Analysis'])
        
        plt.tight_layout()
        plt.savefig('adaptive_complexity_metrics.png')
        print("Saved complexity metrics visualization to 'adaptive_complexity_metrics.png'")
        
        # Create figure for processing times
        plt.figure(figsize=(10, 6))
        
        # Plot processing times
        plt.subplot(1, 2, 1)
        times_df = results_df[['label', 'quick_assessment_time', 'advanced_analysis_time']]
        times_df = times_df.melt(id_vars=['label'], var_name='Phase', value_name='Time (s)')
        sns.boxplot(x='label', y='Time (s)', hue='Phase', data=times_df)
        plt.title('Processing Times by Class and Phase')
        plt.xlabel('Class')
        plt.ylabel('Time (seconds)')
        
        # Plot efficiency gain
        plt.subplot(1, 2, 2)
        # Calculate efficiency metrics
        efficiency_df = results_df.copy()
        efficiency_df['efficiency_gain'] = efficiency_df.apply(
            lambda row: 1 - (row['quick_assessment_time'] / (row['quick_assessment_time'] + row['advanced_analysis_time'])) 
            if row['needs_advanced_analysis'] else 1, axis=1
        )
        sns.boxplot(x='label', y='efficiency_gain', data=efficiency_df)
        plt.title('Efficiency Gain by Class')
        plt.xlabel('Class')
        plt.ylabel('Efficiency Gain (0-1)')
        
        plt.tight_layout()
        plt.savefig('adaptive_processing_times.png')
        print("Saved processing times visualization to 'adaptive_processing_times.png'")
        
        # Print summary statistics
        print("\n===== Summary Statistics =====\n")
        
        # Advanced analysis trigger rate
        trigger_rate = results_df['needs_advanced_analysis'].mean() * 100
        healthy_trigger_rate = results_df[results_df['label'] == 'healthy']['needs_advanced_analysis'].mean() * 100
        failing_trigger_rate = results_df[results_df['label'] == 'failing']['needs_advanced_analysis'].mean() * 100
        
        print(f"Overall advanced analysis trigger rate: {trigger_rate:.1f}%")
        print(f"Healthy files trigger rate: {healthy_trigger_rate:.1f}%")
        print(f"Failing files trigger rate: {failing_trigger_rate:.1f}%")
        
        # Processing time savings
        avg_quick_time = results_df['quick_assessment_time'].mean()
        avg_advanced_time = results_df[results_df['needs_advanced_analysis']]['advanced_analysis_time'].mean()
        avg_total_time = results_df['total_time'].mean()
        
        # Calculate potential time without hybrid approach (all files get advanced analysis)
        potential_total_time = avg_quick_time + avg_advanced_time
        time_savings = (potential_total_time - avg_total_time) / potential_total_time * 100
        
        print(f"Average quick assessment time: {avg_quick_time:.4f} seconds")
        print(f"Average advanced analysis time: {avg_advanced_time:.4f} seconds")
        print(f"Average total processing time: {avg_total_time:.4f} seconds")


def generate_test_data():
    """
    Generate test data for demonstration purposes.
    Creates synthetic healthy and failing audio files with highly distinct patterns.
    """
    print("Generating synthetic test data with 3.0 seconds duration at 22050 Hz...")
    
    # Create data directories if they don't exist
    os.makedirs('data/healthy', exist_ok=True)
    os.makedirs('data/failing', exist_ok=True)
    os.makedirs('data/audio_examples', exist_ok=True)
    
    # Parameters
    sample_rate = SAMPLE_RATE
    duration = 3.0  # seconds
    samples = int(sample_rate * duration)
    t = np.linspace(0, duration, samples, endpoint=False)
    
    # Generate 5 healthy files (HIGHLY ordered - clean harmonic signals)
    for i in range(5):
        # Create an ordered signal (healthy machine sound)
        freq = 220 * (i + 1)  # Different base frequencies
        
        # Start with strong harmonic components
        signal = np.sin(2 * np.pi * freq * t)  # Base frequency
        
        # Add harmonics (ordered overtones)
        for j in range(2, 6):  # Add 4 harmonics
            signal += (1/j) * np.sin(2 * np.pi * freq * j * t)  # Harmonics with decreasing amplitude
        
        # Add amplitude modulation (very ordered)
        am_freq = 5  # 5 Hz amplitude modulation
        signal *= (1 + 0.3 * np.sin(2 * np.pi * am_freq * t))
        
        # Add very minimal noise (healthy machines are clean)
        noise = np.random.normal(0, 0.01, samples)  # Minimal noise
        signal += noise
        
        # Normalize
        signal = signal / np.max(np.abs(signal))
        
        # Save to file
        file_path = f'data/healthy/healthy_{i+1}.wav'
        sf.write(file_path, signal, sample_rate)
        
        # Also save a copy to the audio_examples directory
        example_path = f'data/audio_examples/healthy_{i+1}.wav'
        sf.write(example_path, signal, sample_rate)
        
        print(f"Generated healthy_{i+1}.wav - highly ordered signal with frequency {freq} Hz")
        
        # Compute and save binary representation
        spectrogram = librosa.stft(signal, n_fft=N_FFT, hop_length=HOP_LENGTH)
        spectrogram = np.abs(spectrogram)
        spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)
        
        # Create an instance of AdaptiveComplexityFilter to use its binarize_spectrogram method
        temp_filter = AdaptiveComplexityFilter()
        binary_spectrogram = temp_filter.binarize_spectrogram(spectrogram)
        
        # Save binary representation
        save_binary_representation(binary_spectrogram, file_path)
    
    # Generate 5 failing files (HIGHLY chaotic - very noisy signals)
    for i in range(5):
        # Create a chaotic signal (failing machine sound)
        freq = 220 * (i + 1)
        
        # Start with minimal tonal component
        signal = 0.1 * np.sin(2 * np.pi * freq * t)  # Weak base tone
        
        # Add significant noise (dominant component)
        noise = np.random.normal(0, 0.9, samples)
        signal += noise
        
        # Add frequency instability (wobbling pitch)
        unstable_freq = freq * (1 + 0.3 * np.sin(2 * np.pi * 3.0 * t) + 0.2 * np.random.normal(0, 1, samples))
        signal += 0.3 * np.sin(2 * np.pi * np.cumsum(unstable_freq) / sample_rate)
        
        # Add many random clicks and pops (irregular artifacts)
        for j in range(30):  # More artifacts
            pos = np.random.randint(0, samples - 200)
            click_length = np.random.randint(50, 200)  # Variable length artifacts
            click_length = min(click_length, samples - pos)  # Ensure we don't go out of bounds
            signal[pos:pos+click_length] += np.random.normal(0, 1.5, click_length)  # Stronger artifacts
        
        # Add random bursts of energy (simulating mechanical failures)
        for j in range(5):
            burst_start = np.random.randint(0, samples - int(0.2 * sample_rate))
            burst_length = int(0.2 * sample_rate)  # 200ms burst
            burst_env = np.hanning(burst_length * 2)[:burst_length]  # Hanning window for smooth envelope
            burst_signal = np.random.normal(0, 2.0, burst_length) * burst_env
            signal[burst_start:burst_start+burst_length] += burst_signal
        
        # Normalize
        signal = signal / np.max(np.abs(signal))
        
        # Save to file
        file_path = f'data/failing/failing_{i+1}.wav'
        sf.write(file_path, signal, sample_rate)
        
        # Also save a copy to the audio_examples directory
        example_path = f'data/audio_examples/failing_{i+1}.wav'
        sf.write(example_path, signal, sample_rate)
        
        print(f"Generated failing_{i+1}.wav - highly chaotic signal with base frequency {freq} Hz")
        
        # Compute and save binary representation
        spectrogram = librosa.stft(signal, n_fft=N_FFT, hop_length=HOP_LENGTH)
        spectrogram = np.abs(spectrogram)
        spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)
        
        # Create an instance of AdaptiveComplexityFilter to use its binarize_spectrogram method
        temp_filter = AdaptiveComplexityFilter()
        binary_spectrogram = temp_filter.binarize_spectrogram(spectrogram)
        
        # Save binary representation
        save_binary_representation(binary_spectrogram, file_path)
    
    print(f"\nGenerated 5 synthetic healthy files and 5 synthetic failing files in the data directory")
    print(f"Audio examples are saved in data/audio_examples/ directory for easy access")
    print(f"Binary representations of spectrograms are saved in data/binary_data/ directory")
    print(f"\nThese files simulate HIGHLY ordered (healthy) vs. HIGHLY chaotic (failing) machine sounds.")
    print(f"Healthy files have clear harmonic structure and repeating patterns, while failing files")
    print(f"have significant noise, random artifacts, and unstable frequency components.")
    print(f"This aligns with the BSG complexity findings where more ordered signals tend to have higher complexity values.")
    print(f"\nData is ready for processing.\n")
def main():
    """Main function to run the adaptive hybrid complexity filter."""
    print("\n===== Adaptive Hybrid Complexity Filter =====\n")
    
    # Generate test data for demonstration
    generate_test_data()
    
    # Create and run the filter
    filter = AdaptiveComplexityFilter(data_dir='data')
    
    # Calibrate thresholds
    filter.calibrate_thresholds()
    
    # Process the dataset
    filter.process_dataset(max_files=5)
    
    # Visualize results
    filter.visualize_results()
    
    print("\nAdaptive hybrid complexity filter completed successfully!")
    print("====================================================================\n")


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
    main()
