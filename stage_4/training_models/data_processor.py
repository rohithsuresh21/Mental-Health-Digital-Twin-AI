from flask import json
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from data_validator import SafePipeline
from unified_pipeline import UnifiedPipeline
import time

class Stage4DataProcessor:

    def __init__(self):
        self.pipeline = UnifiedPipeline()
        self.safe_pipeline = SafePipeline(self.pipeline)

        self.healthy_vectors = []
        self.atrisk_vectors = []

    def process_records(self, records: list) -> dict:
        print("Processing records...")
        time.sleep(2)
        print(f"\nProcessing {len(records)} records through Stages 1-2...")


        healthy_count = 0
        atrisk_count = 0
        rejected_count = 0

        for idx, record in enumerate(records):
            if (idx + 1) % 100 == 0:
                print(f" Progress: {idx + 1}/{len(records)} records processed.")
                time.sleep(1)
            
            try:
                result = self.safe_pipeline.process_entry_safe(
                    user_id=record["user_id"],
                    text=record["text"],
                    audio_path=None,
                    sleep_hours=None,
                    sleep_quality=None,
                    activity_level=None,
                    music_mood_score=None
                )

                if result.get("status") == "REJECTED":
                    rejected_count += 1
                    continue

                if "stage_2_output" not in result:
                    rejected_count += 1
                    continue

                z_vector = result["stage_2_output"]["z_scored_vector"]

                if z_vector is None:
                    rejected_count += 1
                    continue

                z_vector = np.array(z_vector, dtype=np.float32)
                
                label = record.get("label", 0)

                if label == 0:
                    self.healthy_vectors.append(z_vector)
                    healthy_count += 1
                else:
                    self.atrisk_vectors.append(z_vector)
                    atrisk_count += 1

            except Exception as e:
                print(f"Error processing record {idx + 1}: {e}")
                rejected_count += 1
                continue


        print(f"\n Processing complete:")
        print(f"  Healthy vectors: {healthy_count}")
        print(f"  At-risk vectors: {atrisk_count}")
        print(f"  Rejected: {rejected_count}")
        print(f"  Total usable: {healthy_count + atrisk_count}")

        return {
            "healthy": np.array(self.healthy_vectors, dtype=np.float32),
            "atrisk": np.array(self.atrisk_vectors, dtype=np.float32)
        
        }
    
    def save_vectors(self, vectors: dict, output_dir: str):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving vectors to {output_dir}...")

        np.save(output_dir / "healthy_vectors.npy", vectors["healthy"])
        np.save(output_dir / "atrisk_vectors.npy", vectors["atrisk"])

        metadata = {
            "healthy_count": len(vectors["healthy"]),
            "atrisk_count": len(vectors["atrisk"]),
            "total_vectors": len(vectors["healthy"]) + len(vectors["atrisk"]),
            "feature_dimension": 466,
            "ratio": f"{len(vectors['healthy'])} : {len(vectors['atrisk'])}"
        }
    
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   healthy_vectors.npy: {len(vectors['healthy'])} samples")
        print(f"   atrisk_vectors.npy: {len(vectors['atrisk'])} samples")
        print(f"   metadata.json")
        
        return output_dir