"""Unit tests for Trimmed Mean aggregation.

Tests cover:
- Basic aggregation functionality
- Byzantine resilience with outliers
- Edge cases (small n, all identical)
- Frequently trimmed client detection
"""

import numpy as np
import pytest
import torch

from src.defense.robust_agg import TrimmedMeanAggregator, MultiKrumAggregator


def make_update(gradient_values):
    """Helper to create update dict from gradient values."""
    return {'gradients': [np.array(gradient_values, dtype=np.float32)]}


class TestTrimmedMeanAggregator:
    """Tests for TrimmedMeanAggregator class."""
    
    def test_initialization(self):
        """Aggregator initializes with valid trim ratio."""
        agg = TrimmedMeanAggregator(trim_ratio=0.1)
        assert agg.trim_ratio == 0.1
    
    def test_invalid_trim_ratio(self):
        """Invalid trim ratio raises error."""
        with pytest.raises(ValueError):
            TrimmedMeanAggregator(trim_ratio=0.5)
        with pytest.raises(ValueError):
            TrimmedMeanAggregator(trim_ratio=-0.1)
    
    def test_empty_updates(self):
        """Empty updates return None."""
        agg = TrimmedMeanAggregator()
        result = agg.aggregate([])
        assert result is None
    
    def test_single_update(self):
        """Single update returns that update."""
        agg = TrimmedMeanAggregator(trim_ratio=0.1)
        updates = [make_update([1.0, 2.0, 3.0])]
        result = agg.aggregate(updates)
        
        assert result is not None
        np.testing.assert_array_almost_equal(
            result['gradients'], [1.0, 2.0, 3.0], decimal=5
        )
    
    def test_identical_updates(self):
        """Identical updates produce same result."""
        agg = TrimmedMeanAggregator(trim_ratio=0.1)
        updates = [make_update([1.0, 2.0, 3.0]) for _ in range(5)]
        result = agg.aggregate(updates)
        
        np.testing.assert_array_almost_equal(
            result['gradients'], [1.0, 2.0, 3.0], decimal=5
        )
    
    def test_basic_aggregation(self):
        """Basic aggregation of diverse updates."""
        agg = TrimmedMeanAggregator(trim_ratio=0.0)  # No trimming
        updates = [
            make_update([1.0]),
            make_update([2.0]),
            make_update([3.0]),
        ]
        result = agg.aggregate(updates)
        
        np.testing.assert_array_almost_equal(
            result['gradients'], [2.0], decimal=5  # Mean of 1, 2, 3
        )
    
    def test_trimming_removes_outliers(self):
        """Trimming removes extreme values."""
        agg = TrimmedMeanAggregator(trim_ratio=0.2)  # Trim 20% each end
        
        # 10 clients: 8 honest (value ~1.0), 2 malicious (extreme values)
        updates = []
        for i in range(8):
            updates.append(make_update([1.0 + 0.1 * i]))  # Values 1.0 to 1.7
        updates.append(make_update([100.0]))  # Outlier high
        updates.append(make_update([-100.0]))  # Outlier low
        
        result = agg.aggregate(updates)
        
        # After trimming 2 from each end, should be close to honest mean
        assert result['gradients'][0] < 2.0  # Not affected by outliers
        assert result['gradients'][0] > 0.5
    
    def test_byzantine_resilience_30_percent(self):
        """Resilient to 30% Byzantine clients."""
        agg = TrimmedMeanAggregator(trim_ratio=0.3)
        
        n_clients = 10
        n_byzantine = 3  # 30%
        honest_value = 1.0
        byzantine_value = 1000.0
        
        updates = []
        for _ in range(n_clients - n_byzantine):
            updates.append(make_update([honest_value]))
        for _ in range(n_byzantine):
            updates.append(make_update([byzantine_value]))
        
        result = agg.aggregate(updates)
        
        # Should be close to honest value due to trimming
        assert abs(result['gradients'][0] - honest_value) < 1.0
    
    def test_multidimensional_gradients(self):
        """Works with multi-dimensional gradients."""
        agg = TrimmedMeanAggregator(trim_ratio=0.1)
        
        updates = [
            make_update([1.0, 2.0, 3.0, 4.0, 5.0]),
            make_update([1.1, 2.1, 3.1, 4.1, 5.1]),
            make_update([0.9, 1.9, 2.9, 3.9, 4.9]),
        ]
        result = agg.aggregate(updates)
        
        assert len(result['gradients']) == 5
    
    def test_torch_tensor_input(self):
        """Works with torch tensors."""
        agg = TrimmedMeanAggregator(trim_ratio=0.1)
        
        updates = [
            {'gradients': [torch.tensor([1.0, 2.0, 3.0])]},
            {'gradients': [torch.tensor([1.1, 2.1, 3.1])]},
            {'gradients': [torch.tensor([0.9, 1.9, 2.9])]},
        ]
        result = agg.aggregate(updates)
        
        assert result is not None
        assert len(result['gradients']) == 3
    
    def test_multiple_gradient_components(self):
        """Works with multiple gradient components per update."""
        agg = TrimmedMeanAggregator(trim_ratio=0.1)
        
        updates = [
            {'gradients': [np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0])]},
            {'gradients': [np.array([1.1, 2.1]), np.array([3.1, 4.1, 5.1])]},
            {'gradients': [np.array([0.9, 1.9]), np.array([2.9, 3.9, 4.9])]},
        ]
        result = agg.aggregate(updates)
        
        # Total 5 dimensions (2 + 3)
        assert len(result['gradients']) == 5
    
    def test_trimmed_counts(self):
        """Result includes trim count info."""
        agg = TrimmedMeanAggregator(trim_ratio=0.2)
        
        updates = [make_update([float(i)]) for i in range(10)]
        result = agg.aggregate(updates)
        
        assert 'trimmed_counts' in result
        assert result['trimmed_counts']['per_end'] == 2  # 10 * 0.2 = 2
        assert result['trimmed_counts']['total_per_dim'] == 4
        assert result['trimmed_counts']['kept_per_dim'] == 6
    
    def test_get_frequently_trimmed_clients(self):
        """Identifies clients frequently trimmed."""
        agg = TrimmedMeanAggregator(trim_ratio=0.2)
        
        # Client 0 and 9 are outliers
        updates = []
        for i in range(10):
            if i == 0:
                updates.append(make_update([-100.0, -100.0, -100.0]))
            elif i == 9:
                updates.append(make_update([100.0, 100.0, 100.0]))
            else:
                updates.append(make_update([1.0, 1.0, 1.0]))
        
        agg.aggregate(updates)
        frequent = agg.get_frequently_trimmed_clients(threshold=0.5)
        
        # Clients 0 and 9 should be frequently trimmed
        trimmed_ids = [x[0] for x in frequent]
        assert 0 in trimmed_ids or 9 in trimmed_ids


class TestMultiKrumAggregator:
    """Tests for MultiKrumAggregator class (legacy)."""
    
    def test_initialization(self):
        """Aggregator initializes correctly."""
        agg = MultiKrumAggregator(f=2, m=3)
        assert agg.f == 2
        assert agg.m == 3
    
    def test_empty_updates(self):
        """Empty updates return None."""
        agg = MultiKrumAggregator()
        result = agg.aggregate([])
        assert result is None
    
    def test_basic_aggregation(self):
        """Basic Krum aggregation works."""
        agg = MultiKrumAggregator(f=1, m=1)  # Select only 1 best
        
        # Need enough clients: n > 2*f + 2, so n > 4 for f=1
        updates = [
            make_update([1.0, 1.0]),
            make_update([1.1, 1.1]),
            make_update([0.9, 0.9]),
            make_update([1.05, 1.05]),
            make_update([0.95, 0.95]),
            make_update([100.0, 100.0]),  # Outlier at index 5
        ]
        result = agg.aggregate(updates)
        
        assert result is not None
        assert 'selected_indices' in result
        # Outlier (index 5) should not be in selected since m=1
        assert 5 not in result['selected_indices']
    
    def test_selected_indices(self):
        """Returns selected client indices."""
        agg = MultiKrumAggregator(f=1, m=2)
        
        updates = [make_update([float(i)]) for i in range(5)]
        agg.aggregate(updates)
        
        indices = agg.get_selected_indices()
        assert len(indices) == 2
    
    def test_few_clients_fallback(self):
        """Falls back to average with few clients."""
        agg = MultiKrumAggregator(f=2, m=1)
        
        # Only 3 clients, not enough for Krum with f=2
        updates = [
            make_update([1.0]),
            make_update([2.0]),
            make_update([3.0]),
        ]
        result = agg.aggregate(updates)
        
        # Should fall back to average
        assert result is not None
        np.testing.assert_array_almost_equal(
            result['gradients'], [2.0], decimal=5
        )


class TestAggregatorComparison:
    """Compare Trimmed Mean vs Multi-Krum."""
    
    def test_both_handle_same_input(self):
        """Both aggregators handle same input format."""
        updates = [make_update([float(i)]) for i in range(10)]
        
        tm_agg = TrimmedMeanAggregator(trim_ratio=0.1)
        krum_agg = MultiKrumAggregator(f=1, m=5)
        
        tm_result = tm_agg.aggregate(updates)
        krum_result = krum_agg.aggregate(updates)
        
        assert tm_result is not None
        assert krum_result is not None
        assert 'gradients' in tm_result
        assert 'gradients' in krum_result
    
    def test_similar_results_no_outliers(self):
        """Similar results when no outliers present."""
        np.random.seed(42)
        updates = [make_update(np.random.randn(10)) for _ in range(20)]
        
        tm_agg = TrimmedMeanAggregator(trim_ratio=0.1)
        krum_agg = MultiKrumAggregator(f=2, m=16)
        
        tm_result = tm_agg.aggregate(updates)
        krum_result = krum_agg.aggregate(updates)
        
        # Results should be reasonably similar for clean data
        diff = np.abs(tm_result['gradients'] - krum_result['gradients'])
        assert np.mean(diff) < 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
