#!/usr/bin/env python3
"""
Hybrid Audio Classification Pipeline

This script integrates the enhanced hybrid classifier with the adaptive complexity filter
to create a complete pipeline for machine sound analysis with multi-level degradation detection.
It combines the strengths of both approaches:
1. Fast first-pass filtering using adaptive complexity thresholds
2. Detailed multi-level degradation classification using BSG and RBS metrics
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import argparse

# Import our custom modules
sys_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys_path not in sys.path:
    sys.path.append(sys_path)

from classifier.adaptive_complexity_filter import AdaptiveComplexityFilter
from classifier.enhanced_hybrid_classifier import EnhancedHybridClassifier


class HybridPipeline:
    """
    Integrates adaptive complexity filtering with enhanced hybrid classification
    for multi-level machine degradation detection.
    """
    
    def __init__(self, data_dir='data', realistic_samples_dir='data/realistic_samples', use_rbs=True):
        """
        Initialize the hybrid pipeline
        
        Args:
            data_dir: Directory containing binary healthy/failing samples
            realistic_samples_dir: Directory containing multi-level degradation samples
            use_rbs: Whether to use RBS metrics (computationally expensive)
        """
        self.data_dir = data_dir
        self.realistic_samples_dir = realistic_samples_dir
        self.use_rbs = use_rbs
        
        # Initialize components
        self.adaptive_filter = AdaptiveComplexityFilter(data_dir=data_dir)
        self.enhanced_classifier = EnhancedHybridClassifier(
            data_dir=realistic_samples_dir, 
            use_rbs=use_rbs
        )
        
        # Create output directory
        os.makedirs(os.path.join(data_dir, 'pipeline_results'), exist_ok=True)
    
    def train_pipeline(self):
        """
        Train both components of the pipeline
        """
        print("\n===== Hybrid Audio Classification Pipeline =====\n")
        print("Step 1: Calibrating adaptive complexity filter thresholds...")
        
        # Calibrate the adaptive filter thresholds
        self.adaptive_filter.calibrate_thresholds()
        
        print("\nStep 2: Training enhanced hybrid classifier...")
        
        # Process dataset and train the enhanced classifier
        self.enhanced_classifier.process_dataset()
        self.enhanced_classifier.analyze_feature_distributions()
        self.enhanced_classifier.train_classifier()
        
        print("\nPipeline training complete!")
    
    def process_file(self, file_path):
        """
        Process a single audio file through the pipeline
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Dictionary with processing results
        """
        print(f"\nProcessing file: {os.path.basename(file_path)}")
        
        # Step 1: Fast first-pass filtering
        print("Step 1: Fast complexity filtering...")
        filter_result = self.adaptive_filter.process_file(file_path)
        
        # Extract quick complexity metrics
        quick_metrics = {
            'bsg_complexity': filter_result['bsg_complexity'],
            'max_rbs_order': filter_result['max_rbs_order'],
            'normalized_tc': filter_result['normalized_tc']
        }
        
        # Determine if the file passes the first filter
        passes_filter = filter_result['passes_filter']
        binary_classification = filter_result['classification']
        
        print(f"Quick metrics: BSG={quick_metrics['bsg_complexity']:.2f}, "
              f"RBS={quick_metrics['max_rbs_order']}, "
              f"NormTC={quick_metrics['normalized_tc']:.2f}")
        print(f"Binary classification: {binary_classification}")
        print(f"Passes filter: {passes_filter}")
        
        # Step 2: If it's potentially failing, perform detailed analysis
        detailed_result = None
        degradation_level = None
        
        if binary_classification == 'failing' or not passes_filter:
            print("\nStep 2: Detailed degradation analysis...")
            
            # Extract detailed features
            features = self.enhanced_classifier.extract_complexity_features(file_path)
            
            # If classifier is trained, predict degradation level
            if self.enhanced_classifier.classifier is not None:
                # Prepare feature vector
                X = np.array([[
                    features['bsg_complexity'],
                    features['max_rbs_order'],
                    features['avg_rbs_order'],
                    features['hamming_weight'],
                    features['transition_count'],
                    features['normalized_tc']
                ]])
                
                # Predict degradation level
                prediction = self.enhanced_classifier.classifier.predict(X)[0]
                degradation_level = list(self.enhanced_classifier.category_labels.keys())[
                    list(self.enhanced_classifier.category_labels.values()).index(prediction)
                ]
                
                print(f"Degradation level: {degradation_level}")
                
                detailed_result = {
                    'features': features,
                    'degradation_level': degradation_level
                }
        else:
            print("\nFile classified as healthy, skipping detailed analysis.")
            degradation_level = 'healthy'
        
        # Combine results
        result = {
            'file_name': os.path.basename(file_path),
            'quick_metrics': quick_metrics,
            'binary_classification': binary_classification,
            'passes_filter': passes_filter,
            'degradation_level': degradation_level,
            'detailed_result': detailed_result
        }
        
        return result
    
    def process_dataset(self, test_dir=None, max_files=None):
        """
        Process a dataset of audio files through the pipeline
        
        Args:
            test_dir: Directory containing test files (if None, uses data_dir)
            max_files: Maximum number of files to process
            
        Returns:
            List of processing results
        """
        if test_dir is None:
            test_dir = os.path.join(self.data_dir, 'test')
        
        if not os.path.exists(test_dir):
            print(f"Test directory {test_dir} does not exist.")
            return []
        
        print(f"\nProcessing files from {test_dir}...")
        
        # Get all WAV files
        wav_files = [f for f in os.listdir(test_dir) if f.endswith('.wav')]
        if max_files is not None:
            wav_files = wav_files[:max_files]
        
        results = []
        for wav_file in wav_files:
            file_path = os.path.join(test_dir, wav_file)
            result = self.process_file(file_path)
            results.append(result)
        
        # Save results to CSV
        results_df = pd.DataFrame(results)
        results_path = os.path.join(self.data_dir, 'pipeline_results', 'pipeline_results.csv')
        results_df.to_csv(results_path, index=False)
        
        print(f"\nProcessed {len(results)} files. Results saved to {results_path}")
        
        return results
    
    def visualize_results(self, results):
        """
        Visualize the pipeline results
        
        Args:
            results: List of processing results
        """
        if not results:
            print("No results to visualize.")
            return
        
        # Convert to DataFrame if not already
        if not isinstance(results, pd.DataFrame):
            results_df = pd.DataFrame(results)
        else:
            results_df = results
        
        # Create confusion matrix if we have ground truth
        if 'true_category' in results_df.columns:
            plt.figure(figsize=(10, 8))
            cm = confusion_matrix(
                results_df['true_category'], 
                results_df['degradation_level']
            )
            categories = self.enhanced_classifier.categories
            
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            plt.colorbar()
            tick_marks = np.arange(len(categories))
            plt.xticks(tick_marks, categories, rotation=45)
            plt.yticks(tick_marks, categories)
            
            # Add text annotations
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, format(cm[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
            
            plt.tight_layout()
            plt.ylabel('True Category')
            plt.xlabel('Predicted Category')
            
            # Save the figure
            plt.savefig(os.path.join(self.data_dir, 'pipeline_results', 'confusion_matrix.png'))
        
        # Create distribution of degradation levels
        plt.figure(figsize=(10, 6))
        results_df['degradation_level'].value_counts().plot(kind='bar')
        plt.title('Distribution of Degradation Levels')
        plt.xlabel('Degradation Level')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, 'pipeline_results', 'degradation_distribution.png'))


def main():
    """Main function to run the hybrid pipeline"""
    parser = argparse.ArgumentParser(description='Hybrid Audio Classification Pipeline')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory containing binary healthy/failing samples')
    parser.add_argument('--realistic-dir', type=str, default='data/realistic_samples',
                        help='Directory containing multi-level degradation samples')
    parser.add_argument('--test-dir', type=str, default=None,
                        help='Directory containing test files')
    parser.add_argument('--use-rbs', action='store_true',
                        help='Enable RBS metrics (computationally expensive)')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Maximum number of files to process')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training and use existing models')
    
    args = parser.parse_args()
    
    # Create and run the pipeline
    pipeline = HybridPipeline(
        data_dir=args.data_dir,
        realistic_samples_dir=args.realistic_dir,
        use_rbs=args.use_rbs
    )
    
    if not args.skip_training:
        pipeline.train_pipeline()
    
    if args.test_dir:
        results = pipeline.process_dataset(test_dir=args.test_dir, max_files=args.max_files)
        pipeline.visualize_results(results)


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
    main()
