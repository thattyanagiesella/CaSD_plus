#!/usr/bin/env python3
"""
Simple test script for the calculate_sem_scores function.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import calculate_sem_scores

def test_calculate_sem_scores():
    """Test the calculate_sem_scores function with sample data."""
    print("Testing calculate_sem_scores function...")

    # Sample X scores (all variables present)
    x_scores_full = {
        "X1": 3.0, "X2": 3.0, "X3": 3.0, "X4": 3.0, "X5": 3.0,
        "X6": 3.0, "X7": 3.0, "X8": 3.0, "X9": 3.0, "X10": 3.0
    }

    # Test with full data
    try:
        result = calculate_sem_scores(x_scores_full, "1")
        print(f"Full data test passed: {result}")
        assert isinstance(result, dict), "Result should be a dict"
        assert "Y1" in result and "Y2" in result and "Y3" in result, "Result should have Y1, Y2, Y3"
        assert all(isinstance(v, (int, float)) for v in result.values()), "Y values should be numbers"
    except Exception as e:
        print(f"Full data test failed: {e}")
        return False

    # Test with missing X scores (should handle gracefully)
    x_scores_partial = {"X1": 2.5, "X2": 4.0}  # Only some variables
    try:
        result = calculate_sem_scores(x_scores_partial, "1")
        print(f"Partial data test passed: {result}")
        assert isinstance(result, dict), "Result should be a dict"
    except Exception as e:
        print(f"Partial data test failed: {e}")
        return False

    # Test with empty dict
    try:
        result = calculate_sem_scores({}, "1")
        print(f"Empty data test passed: {result}")
        assert isinstance(result, dict), "Result should be a dict"
    except Exception as e:
        print(f"Empty data test failed: {e}")
        return False

    # Test with different wilayah
    try:
        result = calculate_sem_scores(x_scores_full, "2")
        print(f"Different wilayah test passed: {result}")
    except Exception as e:
        print(f"Different wilayah test failed: {e}")
        return False

    print("All tests passed!")
    return True

if __name__ == "__main__":
    success = test_calculate_sem_scores()
    sys.exit(0 if success else 1)
