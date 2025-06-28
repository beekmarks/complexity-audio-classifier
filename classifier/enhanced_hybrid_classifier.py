#!/usr/bin/env python3
"""Enhanced Hybrid Audio Classifier

This script implements an enhanced version of the hybrid audio classifier that integrates
BSG and RBS complexity metrics to detect multiple levels of machine degradation.
It builds on the findings from the realistic machine sound analysis and supports
four degradation levels: healthy, early warning, moderate issue, and severe issue.
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import librosa
import librosa.display
import seaborn as sns
import soundfile as sf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

# Import BSG and RBS algorithms
sys_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys_path not in sys.path:
    sys.path.append(sys_path)

from core.binary_split_game import binary_split_game
from core.recursive_bilateral_symmetry import detect_rbs

# Constants
SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
BINARIZE_THRESHOLD = 0.5  # Threshold for binarization

class EnhancedHybridClassifier:
    """Enhanced hybrid classifier for multi-level machine degradation detection"""
    
    def __init__(self, data_dir='data/realistic_samples', use_rbs=True):
        """Initialize the enhanced hybrid classifier
        
        Args:
            data_dir: Directory containing the audio samples
            use_rbs: Whether to use RBS metrics (computationally expensive)
        """
        self.data_dir = data_dir
        self.use_rbs = use_rbs
        self.categories = ['healthy', 'early_warning', 'moderate_issue', 'severe_issue']
        self.category_labels = {
            'healthy': 0,
            'early_warning': 1,
            'moderate_issue': 2,
            'severe_issue': 3
        }
        self.features_df = None
        self.classifier = None
        self.processing_times = {
            'bsg': [],
            'rbs': [],
            'feature_extraction': [],
            'classification': []
        }
        
        # Create output directories if they don't exist
        os.makedirs(os.path.join(data_dir, 'results'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'plots'), exist_ok=True)
    
    def load_audio(self, file_path):
        """Load audio file and return the signal
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Audio signal array
        """
        try:
            audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)
            return audio
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def compute_mel_spectrogram(self, audio):
        """Compute the Mel spectrogram of an audio signal
        
        Args:
            audio: Audio signal array
            
        Returns:
            Mel spectrogram
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    
    def binarize_spectrogram(self, spectrogram):
        """Binarize a spectrogram using median thresholding
        
        Args:
            spectrogram: Input spectrogram
            
        Returns:
            Binary spectrogram (0s and 1s)
        """
        # Use median as threshold for binarization
        threshold = np.median(spectrogram)
        binary_spectrogram = (spectrogram > threshold).astype(int)
        return binary_spectrogram
    
    def calculate_bsg_complexity(self, binary_string):
        """Calculate BSG complexity of a binary string
        
        Args:
            binary_string: Binary string to analyze
            
        Returns:
            BSG complexity value
        """
        start_time = time.time()
        
        # Convert string of '0's and '1's to a list of integers
        binary_list = [int(bit) for bit in binary_string]
        
        # Calculate BSG complexity
        result = binary_split_game(binary_list)
        complexity = result['sc']  # The key is 'sc' not 'structural_complexity'
        
        # Record processing time
        self.processing_times['bsg'].append(time.time() - start_time)
        
        return complexity
    
    def calculate_rbs_with_timeout(self, binary_string, timeout=3):
        """Calculate RBS metrics with timeout using ThreadPoolExecutor
        
        Args:
            binary_string: Binary string to analyze
            timeout: Maximum execution time in seconds
            
        Returns:
            Tuple of (max_order, avg_order)
        """
        # Limit the binary string length to improve performance
        if len(binary_string) > 100:
            binary_string = binary_string[:100]
        
        start_time = time.time()
        
        def run_detect_rbs():
            # Run RBS detection with debug=False to suppress verbose output
            rbs_orders = detect_rbs(binary_string, debug=False)
            # Extract the maximum RBS order for each position
            max_orders = [max(row) if row else 0 for row in rbs_orders]
            # Calculate the average of non-zero orders
            non_zero_orders = [order for order in max_orders if order > 0]
            avg_order = sum(non_zero_orders) / len(non_zero_orders) if non_zero_orders else 0
            return max(max_orders), avg_order
        
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_detect_rbs)
                max_order, avg_order = future.result(timeout=timeout)
        except (FutureTimeoutError, Exception) as e:
            print(f"RBS calculation timed out or failed: {str(e)}")
            max_order, avg_order = 0, 0
        
        # Record processing time
        self.processing_times['rbs'].append(time.time() - start_time)
        
        return max_order, avg_order
        
    def extract_complexity_features(self, file_path, category=None):
        """Extract complexity features from an audio file
        
        Args:
            file_path: Path to the audio file
            category: Optional category label
            
        Returns:
            Dictionary of extracted features
        """
        start_time = time.time()
        
        # Load audio and compute spectrogram
        audio = self.load_audio(file_path)
        if audio is None:
            return None
        
        # Compute mel spectrogram
        mel_spec = self.compute_mel_spectrogram(audio)
        
        # Binarize spectrogram
        binary_spec = self.binarize_spectrogram(mel_spec)
        
        # Extract central part of the spectrogram for analysis (more informative)
        # Focus on mid-frequency range and a subset of time frames
        start_row = binary_spec.shape[0] // 4
        end_row = 3 * binary_spec.shape[0] // 4
        central_spec = binary_spec[start_row:end_row, :]
        
        # Convert to binary string for complexity analysis
        binary_string = ''.join(map(str, central_spec.flatten()))
        
        # Calculate BSG complexity
        bsg_complexity = self.calculate_bsg_complexity(binary_string)
        
        # Calculate hamming weight (number of 1s)
        hamming_weight = binary_string.count('1')
        
        # Calculate transition count
        transitions = sum(1 for i in range(len(binary_string)-1) if binary_string[i] != binary_string[i+1])
        
        # Normalize transition count
        normalized_tc = transitions / (len(binary_string) - 1) if len(binary_string) > 1 else 0
        
        # Calculate RBS metrics if enabled
        if self.use_rbs:
            max_rbs_order, avg_rbs_order = self.calculate_rbs_with_timeout(binary_string)
        else:
            max_rbs_order, avg_rbs_order = 0, 0
        
        # Record processing time
        self.processing_times['feature_extraction'].append(time.time() - start_time)
        
        # Create feature dictionary
        features = {
            'file_name': os.path.basename(file_path),
            'category': category,
            'bsg_complexity': bsg_complexity,
            'max_rbs_order': max_rbs_order,
            'avg_rbs_order': avg_rbs_order,
            'hamming_weight': hamming_weight,
            'transition_count': transitions,
            'normalized_tc': normalized_tc
        }
        
        return features
    
    def process_dataset(self, max_files=None):
        """Process all audio files in the dataset
        
        Args:
            max_files: Maximum number of files to process from each category
            
        Returns:
            DataFrame with extracted features
        """
        print("\n===== Enhanced Hybrid Audio Classifier =====\n")
        print(f"RBS metrics: {'ENABLED' if self.use_rbs else 'DISABLED'}")
        if self.use_rbs:
            print("Note: RBS metrics are computationally expensive and may take a long time.")
        
        print("Loading audio samples and extracting features...\n")
        
        all_features = []
        
        # Process each category
        for category in self.categories:
            category_dir = os.path.join(self.data_dir, category)
            if not os.path.exists(category_dir):
                print(f"Warning: Directory {category_dir} does not exist. Skipping.")
                continue
            
            # Get all WAV files in the category directory
            wav_files = [f for f in os.listdir(category_dir) if f.endswith('.wav')]
            if max_files is not None:
                wav_files = wav_files[:max_files]
            
            print(f"\nProcessing {len(wav_files)} {category} files:")
            for wav_file in wav_files:
                print(f"  - {wav_file}")
                file_path = os.path.join(category_dir, wav_file)
                features = self.extract_complexity_features(file_path, category)
                if features is not None:
                    all_features.append(features)
        
        # Convert to DataFrame
        self.features_df = pd.DataFrame(all_features)
        
        print(f"\nExtracted features from {len(self.features_df)} files\n")
        
        # Save features to CSV
        features_path = os.path.join(self.data_dir, 'results', 'complexity_features.csv')
        self.features_df.to_csv(features_path, index=False)
        
        return self.features_df
    
    def train_classifier(self, test_size=0.3):
        """Train a Random Forest classifier on the extracted features
        
        Args:
            test_size: Proportion of data to use for testing
            
        Returns:
            Tuple of (accuracy, classification_report)
        """
        if self.features_df is None or len(self.features_df) == 0:
            print("No features available. Run process_dataset first.")
            return None, None
        
        print("Training Random Forest classifier...\n")
        
        # Prepare data
        X = self.features_df[['bsg_complexity', 'max_rbs_order', 'avg_rbs_order', 
                             'hamming_weight', 'transition_count', 'normalized_tc']]
        y = self.features_df['category'].map(self.category_labels)
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Train classifier
        start_time = time.time()
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.classifier.fit(X_train, y_train)
        
        # Record processing time
        self.processing_times['classification'].append(time.time() - start_time)
        
        # Evaluate classifier
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, 
                                      target_names=self.categories,
                                      zero_division=0)
        
        print(f"Classifier accuracy: {accuracy:.4f}\n")
        print("Classification Report:")
        print(report)
        
        # Save model
        model_path = os.path.join(self.data_dir, 'results', 'hybrid_classifier.pkl')
        pd.to_pickle(self.classifier, model_path)
        
        # Get feature importances
        feature_importances = pd.DataFrame({
            'feature': X.columns,
            'importance': self.classifier.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return accuracy, report, feature_importances
    
    def analyze_feature_distributions(self):
        """Analyze and visualize feature distributions across categories"""
        if self.features_df is None or len(self.features_df) == 0:
            print("No features available. Run process_dataset first.")
            return
        
        print("Analyzing feature distributions...\n")
        
        # Calculate statistics for each feature by category
        stats = self.features_df.groupby('category').agg({
            'bsg_complexity': ['mean', 'std'],
            'max_rbs_order': ['mean', 'std'],
            'avg_rbs_order': ['mean', 'std'],
            'hamming_weight': ['mean', 'std'],
            'transition_count': ['mean', 'std'],
            'normalized_tc': ['mean', 'std']
        })
        
        print("Feature statistics by category:\n")
        for category in self.categories:
            if category in stats.index:
                print(f"{category.upper()}:")
                for feature in ['bsg_complexity', 'max_rbs_order', 'avg_rbs_order', 
                               'hamming_weight', 'transition_count', 'normalized_tc']:
                    mean = stats.loc[category, (feature, 'mean')]
                    std = stats.loc[category, (feature, 'std')]
                    print(f"  {feature}: {mean:.4f} ± {std:.4f}")
                print()
        
        # Create feature distribution plots
        self.plot_feature_distributions()
        
    def plot_feature_distributions(self):
        """Create plots of feature distributions across categories"""
        if self.features_df is None or len(self.features_df) == 0:
            return
        
        # Set up the figure
        plt.figure(figsize=(20, 15))
        
        # List of features to plot
        features = ['bsg_complexity', 'max_rbs_order', 'avg_rbs_order', 
                   'hamming_weight', 'transition_count', 'normalized_tc']
        
        # Create a subplot for each feature
        for i, feature in enumerate(features):
            plt.subplot(3, 2, i+1)
            
            # Create boxplot
            sns.boxplot(x='category', y=feature, data=self.features_df, palette='viridis')
            
            # Add individual points
            sns.stripplot(x='category', y=feature, data=self.features_df, 
                         color='black', alpha=0.5, jitter=True)
            
            plt.title(f'{feature} by Category')
            plt.xticks(rotation=45)
            plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(self.data_dir, 'plots', 'feature_distributions.png'))
        
        # Create correlation heatmap
        plt.figure(figsize=(10, 8))
        corr_matrix = self.features_df[features].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, 'plots', 'feature_correlations.png'))
        
        # Create feature importance plot if classifier is trained
        if self.classifier is not None:
            plt.figure(figsize=(10, 6))
            feature_importances = pd.DataFrame({
                'feature': features,
                'importance': self.classifier.feature_importances_
            }).sort_values('importance', ascending=False)
            
            sns.barplot(x='importance', y='feature', data=feature_importances, palette='viridis')
            plt.title('Feature Importance')
            plt.tight_layout()
            plt.savefig(os.path.join(self.data_dir, 'plots', 'feature_importance.png'))
    
    def analyze_processing_times(self):
        """Analyze and report processing times"""
        print("\nAnalyzing processing times...")
        
        if self.processing_times['bsg']:
            avg_bsg_time = np.mean(self.processing_times['bsg'])
            print(f"Average BSG calculation time: {avg_bsg_time:.4f} seconds")
        
        if self.processing_times['rbs']:
            avg_rbs_time = np.mean(self.processing_times['rbs'])
            print(f"Average RBS calculation time: {avg_rbs_time:.4f} seconds")
        
        if self.processing_times['feature_extraction']:
            avg_feature_time = np.mean(self.processing_times['feature_extraction'])
            print(f"Average feature extraction time: {avg_feature_time:.4f} seconds")
        
        if self.processing_times['classification']:
            avg_class_time = np.mean(self.processing_times['classification'])
            print(f"Average classification time: {avg_class_time:.4f} seconds")
        
        print("\nAnalysis complete! Results saved to {}/results/".format(self.data_dir))


def main():
    """Main function to run the enhanced hybrid classifier"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Hybrid Audio Classifier')
    parser.add_argument('--data-dir', type=str, default='data/realistic_samples',
                        help='Directory containing audio samples')
    parser.add_argument('--use-rbs', action='store_true',
                        help='Enable RBS metrics (computationally expensive)')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Maximum number of files to process from each category')
    
    args = parser.parse_args()
    
    # Create and run the classifier
    classifier = EnhancedHybridClassifier(data_dir=args.data_dir, use_rbs=args.use_rbs)
    classifier.process_dataset(max_files=args.max_files)
    classifier.analyze_feature_distributions()
    classifier.train_classifier()
    classifier.analyze_processing_times()


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
    main()
