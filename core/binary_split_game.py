#!/usr/bin/env python3
"""
Binary Split Game (BSG) Algorithm Implementation

This module implements the Binary Split Game algorithm for analyzing binary strings.
It calculates the core BSG metrics:
- Structural Complexity (SC): Sum of lengths of all strings in the reduction path
- Depth: Number of reduction steps
- Max Width: Length of the initial binary string
- Final State: The terminating state (0, 1, or 'Null')

Based on the research findings that BSG assigns higher structural complexity to
ordered signals with clear spectral patterns than to chaotic signals.
"""

class BinarySplitGame:
    """
    Implementation of the Binary Split Game algorithm for binary string analysis.
    """
    
    def __init__(self, debug=False):
        """Initialize the BSG processor."""
        self.debug = debug
        self.reset()
    
    def reset(self):
        """Reset the processor state."""
        self.reduction_path = []
        self.structural_complexity = 0
        self.depth = 0
        self.max_width = 0
        self.final_state = None
    
    def process(self, binary_string):
        """
        Process a binary string through the BSG algorithm.
        
        Args:
            binary_string: A string of '0's and '1's
            
        Returns:
            dict: A dictionary containing the BSG metrics
        """
        self.reset()
        
        # Convert to string if input is a list or numpy array
        if not isinstance(binary_string, str):
            binary_string = ''.join(map(str, binary_string))
        
        # Record the initial string
        self.reduction_path.append(binary_string)
        self.max_width = len(binary_string)
        
        # Apply the BSG reduction process
        current_string = binary_string
        
        while len(current_string) > 1:
            # Split the string into left and right halves
            mid = len(current_string) // 2
            left = current_string[:mid]
            right = current_string[mid:]
            
            if self.debug:
                print(f"Current: {current_string}")
                print(f"Left: {left}, Right: {right}")
            
            # Compare the halves and build the next string
            next_string = self._compare_halves(left, right)
            
            if self.debug:
                print(f"Next: {next_string}")
                print("-" * 40)
            
            # If no matches were found, terminate
            if not next_string:
                self.final_state = 'Null'
                break
            
            # Add to reduction path and continue
            self.reduction_path.append(next_string)
            current_string = next_string
            self.depth += 1
            
            # If reduced to a single bit, record the final state
            if len(current_string) == 1:
                self.final_state = current_string
        
        # Calculate structural complexity (sum of lengths of all strings in the path)
        self.structural_complexity = sum(len(s) for s in self.reduction_path)
        
        # Return the metrics
        return {
            'sc': self.structural_complexity,
            'depth': self.depth,
            'max_width': self.max_width,
            'final_state': self.final_state,
            'reduction_path': self.reduction_path
        }
    
    def _compare_halves(self, left, right):
        """
        Compare the left and right halves of a string and return the next string.
        
        The comparison follows BSG rules:
        - For each position i, compare left[i] with right[i]
        - If they match, add the matching bit to the next string
        - If they don't match or one side is shorter, stop comparison
        
        Args:
            left: Left half of the string
            right: Right half of the string
            
        Returns:
            str: The next string in the reduction path, or empty string if no matches
        """
        next_string = ""
        min_len = min(len(left), len(right))
        
        for i in range(min_len):
            if left[i] == right[i]:
                next_string += left[i]
            else:
                break
        
        return next_string

def binary_split_game(binary_string, debug=False):
    """
    Convenience function to run the BSG algorithm on a binary string.
    
    Args:
        binary_string: A string of '0's and '1's
        debug: Whether to print debug information
        
    Returns:
        dict: A dictionary containing the BSG metrics
    """
    bsg = BinarySplitGame(debug=debug)
    return bsg.process(binary_string)

def extract_bsg_features(binary_spectrogram):
    """
    Extract BSG features from a binary spectrogram.
    
    This function processes each frequency bin (row) of the spectrogram
    and calculates aggregate BSG metrics across all bins.
    
    Args:
        binary_spectrogram: 2D numpy array of binary values (0 or 1)
        
    Returns:
        dict: A dictionary of BSG features
    """
    bsg = BinarySplitGame()
    
    # Process each frequency bin
    bin_results = []
    for freq_bin in binary_spectrogram:
        # Convert to string
        binary_string = ''.join(map(str, freq_bin))
        
        # Process with BSG
        result = bsg.process(binary_string)
        bin_results.append(result)
    
    # Calculate aggregate features
    features = {
        'sc_total': sum(r['sc'] for r in bin_results),
        'sc_mean': sum(r['sc'] for r in bin_results) / len(bin_results),
        'sc_max': max(r['sc'] for r in bin_results),
        'depth_mean': sum(r['depth'] for r in bin_results) / len(bin_results),
        'depth_max': max(r['depth'] for r in bin_results),
        'null_ratio': sum(1 for r in bin_results if r['final_state'] == 'Null') / len(bin_results),
        'zero_ratio': sum(1 for r in bin_results if r['final_state'] == '0') / len(bin_results),
        'one_ratio': sum(1 for r in bin_results if r['final_state'] == '1') / len(bin_results),
    }
    
    # Add histogram of depths
    max_depth = features['depth_max']
    depth_hist = [0] * (max_depth + 1)
    for r in bin_results:
        depth_hist[r['depth']] += 1
    
    for i, count in enumerate(depth_hist):
        features[f'depth_{i}_count'] = count / len(bin_results)
    
    return features

if __name__ == "__main__":
    # Test the BSG algorithm with examples from the research
    test_strings = [
        "10101010",  # Alternating bits (Thue-Morse-like)
        "11110000",  # Clear pattern
        "10010110",  # More complex pattern
        "10101100",  # Mixed pattern
        "11111111",  # All ones
        "00000000",  # All zeros
        "10000000",  # Single one
        "01010101"   # Alternating pattern
    ]
    
    print("Testing BSG algorithm with example strings:")
    for s in test_strings:
        result = binary_split_game(s, debug=True)
        print(f"\nString: {s}")
        print(f"Structural Complexity: {result['sc']}")
        print(f"Depth: {result['depth']}")
        print(f"Max Width: {result['max_width']}")
        print(f"Final State: {result['final_state']}")
        print(f"Reduction Path: {result['reduction_path']}")
        print("=" * 50)
