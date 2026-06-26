import pickle
import os
from datetime import datetime
import json
import numpy as np


def save_vector(user_id: str, raw_vec: np.ndarray,timestamp: datetime, base_dir: str = "data"):
    folder=f"{base_dir}/vectors/{user_id}"
    os.makedirs(folder,exist_ok=True)

    vec_path =f"{folder}/raw_vecs.npy"
    ts_path=f"{folder}/timestamps.json"

    if os.path.exists(vec_path):
        existing=np.load(vec_path)
        updated=np.vstack([existing, raw_vec.reshape(1,-1)])

    else:
        updated=raw_vec.reshape(1,-1)

    np.save(vec_path,updated)

    if os.path.exists(ts_path):
        with open(ts_path) as f:
            timestamps= json.load(f)
    else:
        timestamps=[]

    timestamps.append(timestamp.isoformat())    

    with open(ts_path,"w") as f:
        json.dump(timestamps,f)


def load_vector(user_id:str, base_dir:str="data"):
    vec_path=f"{base_dir}/vectors/{user_id}/raw_vecs.npy"
    ts_path=f"{base_dir}/vectors/{user_id}/timestamps.json"

    if not os.path.exists(vec_path):
        return np.array([]),[]

    vec=np.load(vec_path)
    with open(ts_path) as f:
        timestamps= json.load(f)

    return vec,timestamps   
    
def save_baseline(baseline, base_dir:str="data"):
    os.makedirs(f"{base_dir}/baselines", exist_ok=True)
    path = f"{base_dir}/baselines/{baseline.user_id}.pkl"

    with open(path,"wb") as f:
        pickle.dump(baseline, f)

def load_baseline(user_id:str, base_dir:str="data"):
    from baseline import UserBaseline

    path=f"{base_dir}/baselines/{user_id}.pkl"
    if not os.path.exists(path):
        return UserBaseline(user_id)    # brand new user
    
    with open(path, "rb") as f:
        return pickle.load(f)



