import numpy as np
from sklearn.preprocessing import StandardScaler


class UserBaseline:
    SBERT_DIM          = 384
    MASK_START         = 440   
    MIN_ENTRIES_TO_FIT = 14
    REFIT_EVERY        = 30
    MASK_REGIONS = [(440, 444), (455, 466)]

    def __init__(self, user_id:str):
        self.user_id=user_id
        self.raw_vec=[]
        self.scaler=None
        self.entry_count=0

    def add_entry(self,raw_vec:np.ndarray):
        self.raw_vec.append(raw_vec.copy())
        self.entry_count += 1
        
        if len(self.raw_vec)>30:
            self.raw_vec=self.raw_vec[-30:]

        if self.entry_count == self.MIN_ENTRIES_TO_FIT:
            self._fit_scaler()    
        elif (self.entry_count > self.MIN_ENTRIES_TO_FIT and
              self.entry_count % self.REFIT_EVERY == 0):
            self._fit_scaler()    
        
    def _fit_scaler(self):
        recent = np.array(self.raw_vec)

        to_scale = np.concatenate([
            recent[:, 384:440],  
            recent[:, 444:455]    
        ], axis=1)
        self.scaler = StandardScaler()
        self.scaler.fit(to_scale)    

    def normalise(self, raw_vec: np.ndarray) -> np.ndarray | None:
        if self.scaler is None:
            return None

        sbert_part    = raw_vec[384:440]   
        acoustic_vals = raw_vec[444:455]   
        health_masks  = raw_vec[440:444]  
        acoustic_masks= raw_vec[455:466]    
        sbert_embed   = raw_vec[:384]       

        to_scale = np.concatenate([sbert_part, acoustic_vals]).reshape(1, -1)
        scaled   = self.scaler.transform(to_scale)[0]
        
        return np.concatenate([
            sbert_embed,           
            scaled[:56],          
            health_masks,          
            scaled[56:],           
            acoustic_masks        
        ])