import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint
except ImportError:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE


def build_dataframe(patched_data: dict) -> pd.DataFrame:
    rows = []
    for user_id, windows in patched_data.items():
        for window_idx in range(windows.shape[0]):
            for patch_idx in range(windows.shape[1]):
                feature_vec = windows[window_idx, patch_idx].numpy()
                row = {
                    "user_id":   str(user_id),
                    "window_id": f"{user_id}_{window_idx}",
                    "time_idx":  patch_idx + (window_idx * windows.shape[1]),
                    "target":    float(feature_vec.mean()),
                }
                for f_idx, f_val in enumerate(feature_vec):
                    row[f"feature_{f_idx}"] = float(f_val)
                rows.append(row)
    return pd.DataFrame(rows)


def build_dataset(df: pd.DataFrame, feature_dim: int, num_patches: int = 10) -> TimeSeriesDataSet:
    feature_cols = [f"feature_{i}" for i in range(feature_dim)]
    max_prediction_length = 1
    max_encoder_length    = num_patches - max_prediction_length

    return TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="target",
        group_ids=["window_id"],
        min_encoder_length=max_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        time_varying_unknown_reals=feature_cols + ["target"],
        target_normalizer=GroupNormalizer(groups=["window_id"]),
        allow_missing_timesteps=True,
    )


def build_tft(
    dataset: TimeSeriesDataSet,
    hidden_size: int = 64,
    dropout: float = 0.1,
    n_entries: int = 100,
    learning_rate: float = 1e-3,
) -> TemporalFusionTransformer:

    if n_entries < 20:
        dropout = 0.3
        hidden_size = min(hidden_size, 32)
        print(f"[TFT] Small dataset ({n_entries} entries) — dropout set to 0.3, hidden_size capped at {hidden_size}")

    return TemporalFusionTransformer.from_dataset(
        dataset,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=4,
        dropout=dropout,
        hidden_continuous_size=max(16, hidden_size // 2),
        loss=MAE(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )


def train_tft(
    tft: TemporalFusionTransformer,
    train_dataset: TimeSeriesDataSet,
    val_dataset: TimeSeriesDataSet,
    max_epochs: int = 5,
    batch_size: int = 64,
    checkpoint_path: str = "tft_checkpoint.ckpt",
    learning_rate: float = 1e-3,
) -> TemporalFusionTransformer:

    train_loader = train_dataset.to_dataloader(train=True,  batch_size=batch_size, num_workers=0)
    val_loader   = val_dataset.to_dataloader(  train=False, batch_size=batch_size, num_workers=0)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.dirname(os.path.abspath(checkpoint_path)),
        filename=os.path.basename(checkpoint_path).replace(".ckpt", ""),
        monitor="val_loss",
        save_top_k=1,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        gradient_clip_val=0.1,
        enable_model_summary=True,
        enable_progress_bar=True,
        callbacks=[checkpoint_callback],
        accelerator="cpu",
    )

    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)
    return tft


def extract_latent_and_attention(
    tft: TemporalFusionTransformer,
    dataset: TimeSeriesDataSet,
    batch_size: int = 64,
):
    tft = tft.cpu()
    loader        = dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
    all_latents   = []
    all_attention = []
    latent_cache  = {}

    def hook_fn(module, input, output):
        latent_cache["z_t"] = output[1][0][-1].detach()

    hook = tft.rnn.register_forward_hook(hook_fn)
    tft.eval()
    with torch.no_grad():
        for x, _ in loader:
            x         = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in x.items()}
            out       = tft(x)
            attn      = out["attention"]
            attn_mean = attn.mean(dim=1).squeeze(1)
            z_t       = latent_cache["z_t"].reshape(latent_cache["z_t"].shape[0], -1)
            all_latents.append(z_t.cpu())
            all_attention.append(attn_mean.cpu())

    hook.remove()
    latents    = torch.cat(all_latents,   dim=0)
    attentions = torch.cat(all_attention, dim=0)
    return latents, attentions


def project_umap(latents: torch.Tensor, n_components: int = 2, random_state: int = 42):
    from umap import UMAP
    data = latents.numpy()
    if data.shape[0] <= n_components + 1:
        return np.zeros((data.shape[0], n_components), dtype=np.float32)
    n_neighbors = max(2, min(15, data.shape[0] - 1))
    reducer = UMAP(n_components=n_components, n_neighbors=n_neighbors, random_state=random_state)
    return reducer.fit_transform(data)


def run_stage3(
    patched_data: dict,
    feature_dim: int,
    num_patches: int = 10,
    hidden_size: int = 64,
    max_epochs: int = 5,
    batch_size: int = 64,
    checkpoint_path: str = "tft_checkpoint.ckpt",
    n_entries: int = 100,
) -> dict:

    df         = build_dataframe(patched_data)
    window_ids = df["window_id"].unique().tolist()

    user_prefixes  = list(set("_".join(w.split("_")[:-1]) for w in window_ids))
    val_window_ids = set()
    for prefix in user_prefixes:
        user_windows = sorted([w for w in window_ids if "_".join(w.split("_")[:-1]) == prefix])
        if len(user_windows) > 4:
            val_window_ids.update(user_windows[-2:])

    if val_window_ids:
        train_df = df[~df["window_id"].isin(val_window_ids)].reset_index(drop=True)
        val_df   = df[df["window_id"].isin(val_window_ids)].reset_index(drop=True)
        print(f"[TFT] Validation split: {len(val_window_ids)} held-out windows from {len(window_ids)} total")
    else:
        train_df = df
        val_df   = df

    full_dataset  = build_dataset(df,       feature_dim, num_patches=num_patches)
    train_dataset = build_dataset(train_df, feature_dim, num_patches=num_patches)
    val_dataset   = TimeSeriesDataSet.from_dataset(train_dataset, val_df, predict=True)

    if os.path.exists(checkpoint_path):
        print(f"[TFT] Checkpoint found — loading and fine-tuning (frozen encoder).")
        tft = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)
        tft = tft.cpu()

        frozen_count = 0
        for name, param in tft.named_parameters():
            if any(x in name for x in ["lstm_encoder", "input_embeddings", "prescalers", "static_covariates_encoder"]):
                param.requires_grad = False
                frozen_count += 1
        print(f"[TFT] Froze {frozen_count} encoder parameter tensors. Fine-tuning decoder only.")

        tft = train_tft(
            tft, train_dataset, val_dataset,
            max_epochs=max(2, max_epochs // 3),
            batch_size=batch_size,
            checkpoint_path=checkpoint_path,
            learning_rate=1e-4,
        )

    else:
        print("[TFT] No checkpoint found — training from scratch.")
        tft = build_tft(train_dataset, hidden_size=hidden_size, n_entries=n_entries)
        tft = train_tft(
            tft, train_dataset, val_dataset,
            max_epochs=max_epochs,
            batch_size=batch_size,
            checkpoint_path=checkpoint_path,
        )

    latents, attentions = extract_latent_and_attention(tft, full_dataset, batch_size)

    mean_attention = attentions.mean(dim=0).numpy()
    print(f"[TFT] Mean attention weight per patch position (1=most recent):")
    for i, w in enumerate(mean_attention):
        bar = "█" * int(w * 40)
        print(f"  Patch {i+1:2d}: {w:.4f}  {bar}")

    umap_coords = project_umap(latents)

    return {
        "model":                tft,
        "latents":              latents,
        "attention":            attentions,
        "umap_coords":          umap_coords,
        "mean_patch_attention": mean_attention.tolist(),
    }