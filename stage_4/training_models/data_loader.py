import pandas as pd
import numpy as np
import pathlib as Path

class DAICWOZloader:
    def __init__(self, daic_path: str):
        self.daic_path = Path(daic_path)

    def load_labels(self) -> dict:
        candidates = [
            self.daic_path / "Other" / "train_split_Depression_AVEC2017.csv",
            self.daic_path / "other" / "train_split_Depression_AVEC2017.csv",
            self.daic_path / "train_split_Depression_AVEC2017.csv"
        ]

        labels_file =next((p for p in candidates if p.exists()), None)

        if labels_file is None:
            matches = list(self.daic_path.rglob("train_split_Depression_AVEC2017.csv"))
            if matches:
                labels_file = matches[0]

        if labels_file is None:
            print("✗ Labels file not found. Checked:")
            for p in candidates:
                print(f"    {p}")
            return {}
        
        print(f" labels: {labels_file}")
        try:
            df = pd.read_csv(labels_file)
            labels = {}
            for _, row in df.iterrows():
                pid = str(int(float(row["Participant_ID"])))
                labels[pid] = int(float(row["PHQ8_Binary"]))
            print(f"Loaded {len(labels)} DAIC labels")
            return labels
        
        except Exception as e:
            print(f"Error loading labels: {e}")
            return {}
        
    def load_transcripts(self) -> list:
        labels = self.load_labels()
        if not labels:
            print("No labels loaded, cannot load transcripts.")
            return []
        
        records = []
        for transcript_file in sorted(self.daic_path.rglob("*_TRANSCRIPT.csv")):
            participant_id = transcript_file.stem.replace("_TRANSCRIPT", "")
        
            if participant_id not in labels:
                continue

            label = labels[participant_id] 

            try:
                df = pd.read_csv(transcript_file, sep = "\t")

                speaker_col = next(
                    (c for c in df.columns if c.lower() in ("speaker", "role")), None
                )
                text_col = next(
                    (c for c in df.columns if c.lower() in ("value", "transcript", "text", "utterance")), None
                )

                if speaker_col is None or text_col is None:
                    print(f"Unexpected columns in {transcript_file.name}: {list(df.columns)}")
                    continue

                participant_rows = df[
                    df[speaker_col].astype(str).str.strip().str.lower().isin(
                        ["participant", "p"]
                    )
                ]

                for _, row in participant_rows.iterrows():
                    text = str(row[text_col]).strip()
                    if len(text) > 10:
                        records.append({
                            "user_id": f"daic_{participant_id}",
                            "text": text,
                            "label": label,
                            "source": "DAIC-WOZ",
                        })

                    

            except Exception as e:
                print(f"  Error loading {participant_id}: {e}")
                continue
 
        print(f" Loaded {len(records)} DAIC transcript turns")
        return records