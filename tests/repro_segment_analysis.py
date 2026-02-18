import sys
import pandas as pd
import numpy as np
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from segment_analyzer import SegmentErrorAnalyzer

def run_verification():
    print("Generating synthetic data...")
    np.random.seed(42)
    n_samples = 1000
    
    # Generate features
    ages = np.random.randint(18, 70, size=n_samples)
    cities = np.random.choice(['New York', 'London', 'Paris', 'Tokyo'], size=n_samples)
    scores = np.random.normal(70, 10, size=n_samples)
    
    df = pd.DataFrame({
        'Age': ages,
        'City': cities,
        'Score': scores
    })
    
    # Create a BIAS: "Hard-to-learn" samples are mostly older people from New York
    # Let's say: Age > 50 AND City == 'New York' implies high probability of being in subset
    
    subset_probs = np.zeros(n_samples)
    mask_bias = (df['Age'] > 50) & (df['City'] == 'New York')
    subset_probs[mask_bias] = 0.8  # 80% chance for this group
    subset_probs[~mask_bias] = 0.1 # 10% chance for others
    
    is_hard = np.random.rand(n_samples) < subset_probs
    
    print(f"Total samples: {n_samples}")
    print(f"Hard samples (Subset): {sum(is_hard)}")
    print("-" * 30)
    
    # Run Analysis
    analyzer = SegmentErrorAnalyzer(df, is_hard)
    results = analyzer.analyze()
    
    # Print Top Results
    print("Top Riskiest Segments (High Lift):")
    print(results.head(10).to_string(index=False))
    
    # Validation checks
    print("\n" + "-" * 30)
    print("Validation Checks:")
    
    top_segment = results.iloc[0]
    print(f"Top detected segment: {top_segment['Segment']}")
    
    # We expect City == New York or Age intervals > 50 to be at the top
    assert 'New York' in top_segment['Segment'] or 'Age' in top_segment['Segment'], "Failed to detect the injected bias!"
    assert top_segment['Lift'] > 1.5, f"Lift {top_segment['Lift']} is too low for the injected bias."
    assert top_segment['P_Value'] < 0.05, "Top result is not statistically significant."
    
    print("SUCCESS: The analyzer correctly identified the biased segments.")

if __name__ == "__main__":
    run_verification()
