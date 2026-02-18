import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Dict, Optional, Union, Tuple

class SegmentErrorAnalyzer:
    """
    Analyzes which feature segments are statistically over-represented in a specific subset of data.
    Useful for Error Analysis and Data Map interpretation.
    """

    def __init__(self, df: pd.DataFrame, subset_mask: Union[pd.Series, np.ndarray, List[bool]]):
        """
        Initialize the analyzer.

        Args:
            df: The full dataset.
            subset_mask: A boolean mask or index indicating the subset of interest (e.g., hard-to-learn samples).
        """
        self.df = df.copy()
        
        # Ensure mask is boolean and aligned
        if isinstance(subset_mask, (list, np.ndarray)):
            subset_mask = pd.Series(subset_mask, index=df.index)
        
        self.subset_mask = subset_mask.astype(bool)
        self.subset_df = self.df[self.subset_mask]
        self.complement_df = self.df[~self.subset_mask]
        
        self.subset_count = len(self.subset_df)
        self.total_count = len(self.df)
        self.subset_ratio = self.subset_count / self.total_count if self.total_count > 0 else 0

    def analyze(self) -> pd.DataFrame:
        """
        Run the analysis on all columns.

        Returns:
            pd.DataFrame: A ranked table of segments with statistical metrics.
        """
        results = []

        for col in self.df.columns:
            # Skip the mask column if it happens to be in the dataframe (though usually it's external)
            if self.subset_mask.name == col:
                continue

            if pd.api.types.is_numeric_dtype(self.df[col]):
                results.extend(self._analyze_numeric(col))
            else:
                results.extend(self._analyze_categorical(col))

        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df = results_df.sort_values(by='Lift', ascending=False)
        
        return results_df

    def _analyze_categorical(self, col: str, top_n: int = 20) -> List[Dict]:
        """
        Analyze a categorical column.
        """
        results = []
        
        # Get top categories to avoid explosion on high cardinality
        top_categories = self.df[col].value_counts().nlargest(top_n).index.tolist()
        
        for cat in top_categories:
            # Contingency Table:
            #           | In Subset | In Complement
            # Is Cat    | A         | B
            # Not Cat   | C         | D
            
            is_cat = self.df[col] == cat
            
            a = (is_cat & self.subset_mask).sum()
            b = (is_cat & ~self.subset_mask).sum()
            c = (~is_cat & self.subset_mask).sum()
            d = (~is_cat & ~self.subset_mask).sum()
            
            # Simple stats
            segment_total = a + b
            if segment_total == 0:
                continue

            subset_share_in_segment = a / segment_total  # P(Subset | Segment)
            
            # Lift = P(Subset | Segment) / P(Subset)
            lift = subset_share_in_segment / self.subset_ratio if self.subset_ratio > 0 else 0
            
            # Chi-square test
            contingency = [[a, b], [c, d]]
            try:
                chi2, p_value, dof, ex = stats.chi2_contingency(contingency)
            except ValueError:
                p_value = 1.0
                
            results.append({
                'Feature': col,
                'Segment': f"{col} == {cat}",
                'Lift': round(lift, 3),
                'Subset_Share': round(subset_share_in_segment, 3),
                'Global_Share': round(segment_total / self.total_count, 3),
                'P_Value': p_value,
                'Count_In_Subset': a,
                'Count_Total': segment_total
            })
            
        return results

    def _analyze_numeric(self, col: str, n_bins: int = 5) -> List[Dict]:
        """
        Analyze a numeric column by binning it (quantiles).
        """
        results = []
        
        # 1. Distribution Test (KS Test) - Optional, just good to know
        # ks_stat, ks_p = stats.ks_2samp(self.subset_df[col].dropna(), self.complement_df[col].dropna())
        
        # 2. Binning
        # Create bins based on the FULL distribution (or quantiles)
        try:
            # Use qcut for quantiles (e.g. quintiles)
            # We treat NaNs as their own category if possible, or drop them using pandas default behavior
            discretized, bins = pd.qcut(self.df[col], q=n_bins, duplicates='drop', retbins=True)
            
            # Now treat 'discretized' as a categorical feature
            # We need to map the original indices to these bins
            temp_df = pd.DataFrame({'bin': discretized}, index=self.df.index)
            
            # Iterate through unique bins
            unique_bins = temp_df['bin'].unique()
            
            for bin_interval in unique_bins:
                if pd.isna(bin_interval):
                    continue
                    
                is_in_bin = temp_df['bin'] == bin_interval
                
                a = (is_in_bin & self.subset_mask).sum()
                b = (is_in_bin & ~self.subset_mask).sum()
                c = (~is_in_bin & self.subset_mask).sum()
                d = (~is_in_bin & ~self.subset_mask).sum()
                
                segment_total = a + b
                if segment_total == 0:
                    continue

                subset_share_in_segment = a / segment_total
                lift = subset_share_in_segment / self.subset_ratio if self.subset_ratio > 0 else 0
                
                contingency = [[a, b], [c, d]]
                try:
                    chi2, p_value, dof, ex = stats.chi2_contingency(contingency)
                except ValueError:
                    p_value = 1.0

                results.append({
                    'Feature': col,
                    'Segment': f"{col} in {bin_interval}",
                    'Lift': round(lift, 3),
                    'Subset_Share': round(subset_share_in_segment, 3),
                    'Global_Share': round(segment_total / self.total_count, 3),
                    'P_Value': p_value,
                    'Count_In_Subset': a,
                    'Count_Total': segment_total
                })

        except Exception as e:
            # Fallback for constant columns or other errors
            # print(f"Could not bin column {col}: {e}")
            pass

        return results
