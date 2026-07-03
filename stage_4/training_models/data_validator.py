from datetime import datetime
import numpy as np

class SafePipeline:

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def process_entry_safe(
            self,
            user_id: str,
            text: str,
            audio_path = None,
            sleep_hours=None,
            sleep_quality=None,
            activity_level=None,
            music_mood_score=None,
            timestamp=None,
            prev_timestamp=None,
    ) -> dict:
        if not text or not isinstance(text, str) or len(text.strip()) < 5:
            return {"status": "REJECTED", "reason": "text too short or empty"}
        
        if timestamp is None:
            timestamp = datetime.now()

        try:
            result = self.pipeline.process_entry(
                user_id=user_id,
                text=text,
                timestamp=timestamp,
                audio_path=audio_path,
                sleep_hours=sleep_hours,
                sleep_quality=sleep_quality,
                activity_level=activity_level,
                music_mood_score=music_mood_score,
                prev_timestamp=prev_timestamp,
            )

            z_vec = result.get("stage_2_output", {}).get("z_scored_vector")
            if z_vec is None:
                return {"status": "REJECTED", "reason": "stage 2 returned no vector"}
            
            z_arr = np.array(z_vec, dtype=np.float32)
            if np.any(np.isnan(z_arr)) or np.any(np.isinf(z_arr)):
                return {"status": "REJECTED", "reason": "stage 2 returned NaN or Inf values"}
            
            result["status"] = "OK"
            return result
        
        except Exception as e:
            return {"status": "REJECTED", "reason": str(e)}