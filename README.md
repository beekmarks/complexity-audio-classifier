# Complexity-Based Audio Classification System

A machine sound analysis system that leverages Binary Split Game (BSG) and Recursive Bilateral Symmetry (RBS) complexity metrics to detect multi-level machine degradation.

## Overview

This system integrates novel complexity metrics derived from information theory to analyze audio signals for machine condition monitoring. It can distinguish between four degradation levels:

- **Healthy**: Strong harmonics, minimal noise, stable frequency
- **Early Warning**: Slight instability, minor artifacts, subtle resonances
- **Moderate Issue**: Noticeable wobble, regular artifacts, multiple resonances
- **Severe Issue**: Highly unstable, frequent artifacts, bursts of noise

The hybrid pipeline combines fast first-pass filtering with detailed multi-level classification to achieve high accuracy (~87.5%) while maintaining reasonable processing times.

## Theoretical Foundation

### Binary Split Game (BSG)

The Binary Split Game is a recursive algorithm that operates on binary strings, splitting them into halves and matching corresponding positions to generate a sequence of transformed strings. 

Key BSG metrics include:
- **Structural Complexity (SC)**: Sum of lengths of all strings in the reduction path
- **Depth**: Number of reduction steps
- **Max Width**: Length of the initial binary string
- **Final State**: The terminating state (0, 1, or 'Null')

### Recursive Bilateral Symmetry (RBS)

RBS is a hierarchical structure analogous to fractal self-similarity, generalizing both standard and gapped palindromes. It represents a nested arrangement of palindromic structures within a string.

Key RBS metrics include:
- **RBS Order**: The level of recursive symmetry in a string
- **Maximum RBS Order**: The highest RBS order found in any substring
- **Average RBS Order**: The mean RBS order across all substrings

For more details, see the [RBS Algorithm paper](docs/theory/rbs_algorithm.pdf).

## Complementary Nature of BSG and RBS

While both BSG and RBS analyze binary patterns, they measure fundamentally different aspects of complexity and offer complementary benefits in our classification system:

### Different Complexity Aspects

- **BSG (Binary Split Game)** measures structural complexity through recursive splitting and matching, focusing on how patterns can be reduced through a series of transformations. It captures the overall reducibility and structure of patterns.

- **RBS (Recursive Bilateral Symmetry)** detects hierarchical palindromic structures, focusing specifically on symmetry patterns and their nested organization. It captures how symmetrical and self-similar the patterns are.

### Practical Differences in Application

1. **Computational Efficiency**
   - BSG is significantly faster (~0.0017s per sample vs ~0.0502s for RBS)
   - This makes BSG ideal for quick first-pass filtering in multi-stage classification

2. **Sensitivity to Different Degradation Types**
   - BSG may be more sensitive to certain types of machine degradation not well-represented in current datasets
   - RBS shows clearer differentiation between degradation levels in our current testing

3. **Feature Combination Power**
   - Even when individual metrics show limited variation, their combination in machine learning models can extract valuable patterns
   - Our Random Forest classifier leverages both BSG and RBS features to achieve higher accuracy than either alone

### When to Use Each Algorithm

- Use **BSG** when computational efficiency is critical or when analyzing overall pattern structure
- Use **RBS** when detailed symmetry analysis is needed and processing time is less critical
- Use **both together** for the highest classification accuracy, as they provide complementary information

## Application to Audio Classification

Our research has found that these complexity metrics effectively capture patterns in machine degradation:

1. **RBS Order Pattern**:
   - Healthy machines: Higher RBS order (avg ~15.5)
   - Early Warning: Moderate RBS order (avg ~10.1)
   - Moderate Issue: Lower RBS order (avg ~5.5)
   - Severe Issue: Lowest RBS order (avg ~3.6)

   This suggests that healthy machines maintain more symmetrical patterns in their acoustic signatures, while degradation leads to progressive loss of symmetry.

2. **Transition Count Pattern**:
   - Healthy machines: Lower transition counts
   - Degraded machines: Higher transition counts as degradation increases

   This indicates that degradation introduces more irregularities and transitions in the binary patterns.

## System Components

### Core Algorithms
- `binary_split_game.py`: Implementation of the BSG algorithm
- `recursive_bilateral_symmetry.py`: Implementation of the RBS detection algorithm

### Classifier Components
- `enhanced_hybrid_classifier.py`: Main classifier with BSG and RBS metrics
- `adaptive_complexity_filter.py`: Fast first-pass filter
- `hybrid_pipeline.py`: Complete classification pipeline

### Data Generation
- `generate_enhanced_samples.py`: Generator for realistic machine sounds

## Usage

### Installation

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Hybrid Pipeline

```bash
python classifier/hybrid_pipeline.py --use-rbs
```

Optional arguments:
- `--data-dir`: Directory containing binary healthy/failing samples
- `--realistic-dir`: Directory containing multi-level degradation samples
- `--test-dir`: Directory containing test files
- `--use-rbs`: Enable RBS metrics (computationally expensive)
- `--max-files`: Maximum number of files to process
- `--skip-training`: Skip training and use existing models

### Generating Test Samples

```bash
python data_generation/generate_enhanced_samples.py --num-machines 20
```

## Performance

- Classification accuracy: 87.5% across four degradation levels
- Processing times:
  - BSG calculation: ~0.0017 seconds per sample
  - RBS calculation: ~0.0502 seconds per sample
  - Total feature extraction: ~0.0711 seconds per sample

## References

1. "An Algorithmic and Theoretical Analysis of Integer Classification via Recursive Binary Splitting" - See `docs/theory/bsg_algorithm.pdf`
2. "An Optimal Algorithm for the Detection of Recursive Bilateral Symmetry in Strings" - See `docs/theory/rbs_algorithm.pdf`
