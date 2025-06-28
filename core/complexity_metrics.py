#!/usr/bin/env python3
"""
Common complexity metrics utilities for audio classification.

This module provides functions for calculating various complexity metrics
from binary representations of audio signals, including:
- Hamming weight (bit count)
- Transition counts
- Normalized transition counts
- Helper functions for BSG and RBS metric extraction
"""

import numpy as np
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import time
import sys
import os

# Ensure core modules are in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.binary_split_game import binary_split_game
from core.recursive_bilateral_symmetry import detect_rbs


def calculate_hamming_weight(binary_string):
    """
    Calculate the Hamming weight (number of 1s) in a binary string.
    
    Parameters:
    -----------
    binary_string : str
        Binary string representation
        
    Returns:
    --------
    int
        Hamming weight (number of 1s)
    """
    return binary_string.count('1')


def calculate_transition_count(binary_string):
    """
    Calculate the number of transitions (0->1 or 1->0) in a binary string.
    
    Parameters:
    -----------
    binary_string : str
        Binary string representation
        
    Returns:
    --------
    int
        Number of transitions
    """
    if len(binary_string) <= 1:
        return 0
    
    transitions = 0
    for i in range(1, len(binary_string)):
        if binary_string[i] != binary_string[i-1]:
            transitions += 1
    
    return transitions


def calculate_normalized_transition_count(binary_string):
    """
    Calculate the normalized transition count (transitions / length).
    
    Parameters:
    -----------
    binary_string : str
        Binary string representation
        
    Returns:
    --------
    float
        Normalized transition count
    """
    if len(binary_string) <= 1:
        return 0.0
    
    transition_count = calculate_transition_count(binary_string)
    return transition_count / (len(binary_string) - 1)


def calculate_bsg_metrics(binary_string):
    """
    Calculate Binary Split Game metrics for a binary string.
    
    Parameters:
    -----------
    binary_string : str
        Binary string representation
        
    Returns:
    --------
    dict
        Dictionary containing BSG metrics
    """
    # Handle edge cases
    if not binary_string or len(binary_string) <= 1:
        return {'sc': 0, 'depth': 0, 'max_width': len(binary_string)}
    
    # Calculate BSG metrics
    return binary_split_game(binary_string)


def calculate_rbs_metrics_with_timeout(binary_string, max_length=100, timeout=3):
    """
    Calculate Recursive Bilateral Symmetry metrics with a timeout.
    
    Parameters:
    -----------
    binary_string : str
        Binary string representation
    max_length : int, optional
        Maximum length of string to analyze (default: 100)
    timeout : int, optional
        Timeout in seconds (default: 3)
        
    Returns:
    --------
    dict
        Dictionary containing RBS metrics or None if timeout
    """
    # Truncate binary string if needed
    if len(binary_string) > max_length:
        binary_string = binary_string[:max_length]
    
    # Use ThreadPoolExecutor for timeout
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(detect_rbs, binary_string, False)
        
        try:
            rbs_matrix = future.result(timeout=timeout)
            
            # Extract metrics from RBS matrix
            max_order = 0
            total_order = 0
            count = 0
            
            for i in range(len(rbs_matrix)):
                for j in range(i, len(rbs_matrix)):
                    if rbs_matrix[i][j] >= 0:  # Valid RBS order
                        max_order = max(max_order, rbs_matrix[i][j])
                        total_order += rbs_matrix[i][j]
                        count += 1
            
            avg_order = total_order / count if count > 0 else 0
            
            return {
                'max_rbs_order': max_order,
                'avg_rbs_order': avg_order,
                'rbs_density': count / (len(binary_string) * (len(binary_string) + 1) / 2)
            }
            
        except FutureTimeoutError:
            print(f"RBS calculation timed out after {timeout} seconds")
            return {
                'max_rbs_order': 0,
                'avg_rbs_order': 0,
                'rbs_density': 0
            }


def extract_all_complexity_features(binary_string, use_rbs=True, max_rbs_length=100, rbs_timeout=3):
    """
    Extract all complexity features from a binary string.
    
    Parameters:
    -----------
    binary_string : str
        Binary string representation
    use_rbs : bool, optional
        Whether to calculate RBS metrics (default: True)
    max_rbs_length : int, optional
        Maximum length for RBS calculation (default: 100)
    rbs_timeout : int, optional
        Timeout for RBS calculation in seconds (default: 3)
        
    Returns:
    --------
    dict
        Dictionary containing all complexity features
    """
    features = {}
    
    # Basic metrics
    features['hamming_weight'] = calculate_hamming_weight(binary_string)
    features['transition_count'] = calculate_transition_count(binary_string)
    features['normalized_transition_count'] = calculate_normalized_transition_count(binary_string)
    
    # BSG metrics
    start_time = time.time()
    bsg_metrics = calculate_bsg_metrics(binary_string)
    features['bsg_time'] = time.time() - start_time
    features.update(bsg_metrics)
    
    # RBS metrics (optional)
    if use_rbs:
        start_time = time.time()
        rbs_metrics = calculate_rbs_metrics_with_timeout(
            binary_string, 
            max_length=max_rbs_length, 
            timeout=rbs_timeout
        )
        features['rbs_time'] = time.time() - start_time
        
        if rbs_metrics:
            features.update(rbs_metrics)
    
    return features
