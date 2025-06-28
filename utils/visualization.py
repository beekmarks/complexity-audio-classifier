#!/usr/bin/env python3
"""
Visualization utilities for the complexity-based audio classification system.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report


def plot_waveform_and_spectrogram(audio_data, sr, title=None, figsize=(12, 8)):
    """
    Plot waveform and spectrogram of an audio signal.
    
    Parameters:
    -----------
    audio_data : numpy.ndarray
        Audio signal data
    sr : int
        Sample rate
    title : str, optional
        Title for the plot
    figsize : tuple, optional
        Figure size (width, height)
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    # Plot waveform
    time = np.linspace(0, len(audio_data) / sr, len(audio_data))
    axes[0].plot(time, audio_data)
    axes[0].set_title('Waveform')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    
    # Plot spectrogram
    spec = plt.specgram(audio_data, Fs=sr, cmap='viridis')
    plt.colorbar(spec[3], ax=axes[1])
    axes[1].set_title('Spectrogram')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Frequency (Hz)')
    
    if title:
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    return fig


def plot_feature_distributions(features_df, target_column='degradation_level', figsize=(15, 10)):
    """
    Plot distributions of features across different classes.
    
    Parameters:
    -----------
    features_df : pandas.DataFrame
        DataFrame containing features and target variable
    target_column : str, optional
        Name of the target variable column
    figsize : tuple, optional
        Figure size (width, height)
    """
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    axes = axes.flatten()
    
    feature_cols = [col for col in features_df.columns if col != target_column]
    
    for i, feature in enumerate(feature_cols[:6]):  # Plot up to 6 features
        if i >= len(axes):
            break
            
        sns.boxplot(x=target_column, y=feature, data=features_df, ax=axes[i])
        axes[i].set_title(f'Distribution of {feature}')
        axes[i].set_xlabel('Degradation Level')
        axes[i].set_ylabel(feature)
    
    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_true, y_pred, class_names=None, figsize=(8, 6)):
    """
    Plot confusion matrix for classification results.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    class_names : list, optional
        List of class names
    figsize : tuple, optional
        Figure size (width, height)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names if class_names else 'auto',
                yticklabels=class_names if class_names else 'auto')
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    return fig


def plot_feature_importance(model, feature_names, figsize=(10, 6)):
    """
    Plot feature importance from a trained model.
    
    Parameters:
    -----------
    model : object
        Trained model with feature_importances_ attribute (e.g., RandomForest)
    feature_names : list
        List of feature names
    figsize : tuple, optional
        Figure size (width, height)
    """
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Model does not have feature_importances_ attribute")
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    plt.title('Feature Importance')
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    
    return fig


def plot_processing_times(times_dict, figsize=(10, 6)):
    """
    Plot processing times for different components.
    
    Parameters:
    -----------
    times_dict : dict
        Dictionary with component names as keys and processing times as values
    figsize : tuple, optional
        Figure size (width, height)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    components = list(times_dict.keys())
    times = list(times_dict.values())
    
    # Convert to milliseconds for better readability if times are small
    if max(times) < 0.1:
        times = [t * 1000 for t in times]
        unit = 'ms'
    else:
        unit = 's'
    
    ax.barh(components, times)
    ax.set_xlabel(f'Processing Time ({unit})')
    ax.set_ylabel('Component')
    ax.set_title('Processing Times by Component')
    
    # Add time values as text
    for i, v in enumerate(times):
        ax.text(v + max(times) * 0.01, i, f'{v:.2f} {unit}', va='center')
    
    plt.tight_layout()
    return fig
