import numpy as np
from sklearn.preprocessing import StandardScaler


class UserBaseline:

    SBERT_START = 0
    SBERT_END = 384
    TEXT_START = 384
    TEXT_END = 440
    HEALTH_MASK_START = 440
    HEALTH_MASK_END = 444
    AUDIO_START = 444
    AUDIO_END = 455
    AUDIO_MASK_START = 455
    AUDIO_MASK_END = 466

    VECTOR_DIM = 466

    MIN_ENTRIES_TO_FIT = 14
    BASELINE_WINDOW    = 60   # ← NEW: scaler fits on last 60 entries
    INFERENCE_WINDOW   = 30
    REFIT_EVERY        = 30 

    def __init__(self, user_id: str,baseline_window: int= BASELINE_WINDOW,inference_window:int=INFERENCE_WINDOW,min_entries_to_fit: int = MIN_ENTRIES_TO_FIT,refit_every: int= REFIT_EVERY):
        self.user_id=user_id
        self.raw_vecs: list[np.ndarray] = []
        self.scaler=None
        self.entry_count=0
        self.last_refit= 0

        self.baseline_window    = baseline_window
        self.inference_window   = inference_window
        self.min_entries_to_fit = min_entries_to_fit
        self.refit_every        = refit_every


    def add_entry(self,raw_vec:np.ndarray):
        if raw_vec.shape!=(self.VECTOR_DIM,):
            raise ValueError(f"Expected feature vector of shape ({self.VECTOR_DIM},), got {raw_vec.shape}")
        self.raw_vecs.append(raw_vec.copy())
        self.entry_count += 1
        
        if self.entry_count == self.min_entries_to_fit:
            self._fit_scaler()
            self.last_refit = self.entry_count
        elif (self.entry_count > self.min_entries_to_fit and
              self.entry_count - self.last_refit >= self.refit_every):
            self._fit_scaler()
            self.last_refit = self.entry_count
            
        if len(self.raw_vecs) > self.baseline_window:
            self.raw_vecs = self.raw_vecs[-self.baseline_window:]
        
    def _fit_scaler(self):
        recent = np.array(self.raw_vecs)

        to_scale = np.concatenate([
            recent[:, self.TEXT_START:self.TEXT_END],
            recent[:, self.AUDIO_START:self.AUDIO_END]   
        ], axis=1)
        self.scaler = StandardScaler()
        self.scaler.fit(to_scale)    

    def get_inference_window(self) -> np.ndarray | None:
        """Returns last 30 normalised vectors for Stage 3 and Stage 4."""
        if self.scaler is None:
            return None
        last_30 = self.raw_vecs[-self.INFERENCE_WINDOW:]
        return np.array([self.normalise(v) for v in last_30])    

    def normalise(self, raw_vec: np.ndarray) -> np.ndarray | None:

        if raw_vec.shape != (self.VECTOR_DIM,):
            raise ValueError(f"Expected feature vector of shape ({self.VECTOR_DIM},), got {raw_vec.shape}")
        if self.scaler is None:
            return None

        text_part = raw_vec[self.TEXT_START:self.TEXT_END]
        acoustic_vals = raw_vec[self.AUDIO_START:self.AUDIO_END]   
        health_masks = raw_vec[self.HEALTH_MASK_START:self.HEALTH_MASK_END]
        acoustic_masks = raw_vec[self.AUDIO_MASK_START:self.AUDIO_MASK_END]  
        sbert_embed = raw_vec[self.SBERT_START:self.SBERT_END]  

        to_scale = np.concatenate([text_part, acoustic_vals]).reshape(1, -1)
        scaled   = self.scaler.transform(to_scale)[0]
        text_dim = self.TEXT_END - self.TEXT_START
        
        return np.concatenate([
            sbert_embed,           
            scaled[:text_dim],          
            health_masks,          
            scaled[text_dim:],           
            acoustic_masks        
        ])

    @property
    def calibrated(self) -> bool:
        return self.scaler is not None

    def calibration_status(self) -> dict:
        """Returns calibration state for frontend/pipeline consumption."""
        return {
            "calibrated":           self.calibrated,
            "entries_so_far":       self.entry_count,
            "entries_needed":       self.min_entries_to_fit,
            "calibration_progress": f"{min(self.entry_count, self.min_entries_to_fit)}/{self.min_entries_to_fit}",
            "last_refit_at":        self.last_refit,
            "next_refit_at":        self.last_refit + self.refit_every if self.calibrated else self.min_entries_to_fit,
        }    
