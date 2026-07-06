import datetime
import torch
import numpy as np

class TemporalBinning:
    def __init__(self, feature_dim=466):
        
        self.feature_dim = feature_dim
        
        self.bin_names = [
            "Morning_Weekday", "Afternoon_Weekday", "Evening_Weekday",
            "Morning_Weekend", "Afternoon_Weekend", "Evening_Weekend"
        ]
        
        self.clear_buckets()

    def clear_buckets(self):
        self.context_buckets = {bin_name: [] for bin_name in self.bin_names}

    def _determine_bin(self, timestamp_str):
       
        try:
            dt = datetime.datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except ValueError:
            dt = datetime.datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")

        # Weekday (0-4) vs Weekend (5-6)
        is_weekend = dt.weekday() >= 5
        day_label = "Weekend" if is_weekend else "Weekday"

        #  Morning (05:00-11:59), Afternoon (12:00-16:59), Evening (17:00-04:59)
        hour = dt.hour
        if 5 <= hour < 12:
            time_label = "Morning"
        elif 12 <= hour < 17:
            time_label = "Afternoon"
        else:
            time_label = "Evening"
        return f"{time_label}_{day_label}"

    def route_vector(self, feature_vector, timestamp_str):
        # Convert to numpy if input is a torch tensor
        if isinstance(feature_vector, torch.Tensor):
            feature_vector = feature_vector.detach().cpu().numpy()
            
        if feature_vector.shape[-1] != self.feature_dim:
            raise ValueError(f"Dimension Mismatch: Expected {self.feature_dim} features, received {feature_vector.shape[-1]}")
        
        target_bin = self._determine_bin(timestamp_str)
        self.context_buckets[target_bin].append(feature_vector)
        
        return target_bin

    def get_bucket_matrix(self, bin_name):
        
        vectors = self.context_buckets.get(bin_name, [])
        if not vectors:
            return np.empty((0, self.feature_dim), dtype=np.float32) 
        
        # Stacks rows together into a unified 2D layout matrix
        return np.vstack(vectors)
