from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataSyntheticConfig:
    num_assets: int = 8
    num_steps: int = 5000
    feature_dim: int = 12
    regime_switch_prob: float = 0.002


@dataclass
class DataConfig:
    source: str = "synthetic"
    csv_path: Optional[str] = None
    parquet_path: Optional[str] = None
    timestamp_col: str = "timestamp"
    asset_col: str = "asset_id"
    feature_cols: Optional[List[str]] = None
    target_cols: Optional[List[str]] = None
    context_length: int = 128
    horizons: List[int] = field(default_factory=lambda: [1, 5, 20])
    target_window: int = 8
    train_frac: float = 0.7
    val_frac: float = 0.15
    normalize: bool = True
    resample_freq: Optional[str] = None
    synthetic: DataSyntheticConfig = field(default_factory=DataSyntheticConfig)


@dataclass
class ModelConfig:
    d_model: int = 128
    latent_dim: int = 64
    num_assets: int = 8
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1
    backbone: str = "transformer"
    predictor_hidden: int = 128
    stochastic: bool = True
    ema_decay: float = 0.996
    use_target_ema: bool = True
    predictor_mode: str = "split"
    stage_names: List[str] = field(default_factory=lambda: ["macro", "liquidity", "volatility", "residual"])
    use_memory: bool = True
    memory_size: int = 64
    memory_topk: int = 4
    memory_temperature: float = 0.2
    memory_decay: float = 0.99
    memory_write_threshold: float = 0.15
    memory_merge_threshold: float = 0.9
    use_reconstruction_head: bool = True
    use_operator_gate: bool = True


@dataclass
class TrainConfig:
    batch_size: int = 64
    num_workers: int = 0
    lr: float = 3e-4
    weight_decay: float = 1e-2
    max_epochs: int = 20
    grad_clip_norm: float = 1.0
    log_every: int = 20
    validate_every: int = 1
    amp: bool = False
    accumulate_grad_batches: int = 1
    distributed: bool = False
    compile_model: bool = False


@dataclass
class LossConfig:
    multi_horizon_weights: List[float] = field(default_factory=lambda: [1.0, 0.7, 0.5])
    lambda_nll: float = 0.25
    lambda_noarb: float = 0.05
    lambda_vol: float = 0.1
    lambda_liq: float = 0.1
    lambda_macro: float = 0.05
    lambda_reg: float = 0.2
    lambda_recon: float = 0.05
    lambda_gate: float = 0.02
    lambda_mem: float = 0.05
    use_financial_regularizers: bool = True


@dataclass
class ExperimentConfig:
    output_dir: str = "runs/fijepa_base"
    save_best_metric: str = "val_total_loss"
    mode: str = "self_supervised"
    tracker: str = "json"


@dataclass
class FIJEPAConfig:
    seed: int = 7
    device: str = "auto"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
