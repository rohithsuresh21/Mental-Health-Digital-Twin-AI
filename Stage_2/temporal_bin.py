import datetime
import torch
import numpy as np

class TemporalBinning:
    def __init__(self, feature_dim=466):
        """
        Stage 4 — Temporal Binning Router.
        [Morning, Afternoon, Evening] x [Weekday, Weekend]
        """
        self.feature_dim = feature_dim
        
        self.bin_names = [
            "Morning_Weekday", "Afternoon_Weekday", "Evening_Weekday",
            "Morning_Weekend", "Afternoon_Weekend", "Evening_Weekend"
        ]
        
        self.clear_buckets = self.clear_buckets  # Explicit binding
        self.clear_buckets()

    def clear_buckets(self):
        """
        Initializes or resets the storage buffers for all 6 contextual bins. 
        Only called at startup or when explicitly requested to clear the session.
        """
        self.context_buckets = {bin_name: [] for bin_name in self.bin_names}

    def _determine_bin(self, timestamp_str):
        """
        Determines the appropriate bin for a given timestamp string.
        Is called internally by route_vector to route the feature vector to the correct bucket.
        """
        dt = None
        # Comprehensive try-except fallback hierarchy for string parsing
        try:
            # Handles standard ISO-8601 strings (e.g., "2026-07-08T13:17:47Z")
            dt = datetime.datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except ValueError:
            try:
                # Handles standard space-separated datetime strings (e.g., "2026-07-08 13:17:47")
                dt = datetime.datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                try:
                    # Fallback for ISO strings that have a space instead of a 'T'
                    dt = datetime.datetime.fromisoformat(timestamp_str.replace(' ', 'T').replace('Z', '+00:00'))
                except ValueError:
                    raise ValueError(f"Unrecognized timestamp format string: {timestamp_str}")

        # Weekday (0-4) vs Weekend (5-6)
        is_weekend = dt.weekday() >= 5
        day_label = "Weekend" if is_weekend else "Weekday"

        # Morning (05:00-11:59), Afternoon (12:00-16:59), Evening (17:00-04:59)
        hour = dt.hour
        if 5 <= hour < 12:
            time_label = "Morning"
        elif 12 <= hour < 17:
            time_label = "Afternoon"
        else:
            time_label = "Evening"
        return f"{time_label}_{day_label}"

    def route_vector(self, feature_vector, timestamp_str):
        """
        Args:
            feature_vector (Tensor or ndarray): 466-dimensional array slice.
            timestamp_str (str): UTC or local string timestamp associated with the entry.
        """
        # Standardized length/dimension check uniform for PyTorch, NumPy, or basic iterables
        if hasattr(feature_vector, 'shape'):
            incoming_dim = feature_vector.shape[-1]
        elif hasattr(feature_vector, 'size'):
            incoming_dim = feature_vector.size(-1) if callable(feature_vector.size) else feature_vector.size[-1]
        else:
            incoming_dim = len(feature_vector)

        if incoming_dim != self.feature_dim:
            raise ValueError(f"Dimension Mismatch: Expected {self.feature_dim} features, received {incoming_dim}")

        # Safe conversion to NumPy after structure validation
        if isinstance(feature_vector, torch.Tensor):
            feature_vector = feature_vector.detach().cpu().numpy()
        elif isinstance(feature_vector, list):
            feature_vector = np.array(feature_vector)
            
        target_bin = self._determine_bin(timestamp_str)
        self.context_buckets[target_bin].append(feature_vector)
        
        return target_bin

    def get_bucket_matrix(self, bin_name):
        vectors = self.context_buckets.get(bin_name, [])
        if not vectors:
            return np.empty((0, self.feature_dim), dtype=np.float32) 
        
        # Stacks rows together into a unified 2D layout matrix
        return np.vstack(vectors)
