from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

try:
    import statsmodels.api as sm
except Exception:
    sm = None

from .config import FIJEPAConfig


class FitNormalizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, arr: np.ndarray):
        arr = np.asarray(arr, dtype=np.float32)
        self.mean = arr.mean(axis=0)
        self.std = arr.std(axis=0) + 1e-6
        return self

    def transform(self, arr: np.ndarray):
        if self.mean is None or self.std is None:
            raise ValueError("Normalizer not fit.")
        return (np.asarray(arr, dtype=np.float32) - self.mean) / self.std


def _make_timestamp_index(n: int, freq: str = "Q"):
    if freq == "Q":
        return pd.period_range("1959Q1", periods=n, freq="Q").to_timestamp()
    return pd.date_range("2000-01-01", periods=n, freq=freq)


def generate_synthetic_market(num_assets: int = 8, num_steps: int = 5000, feature_dim: int = 12, regime_switch_prob: float = 0.002, seed: int = 7):
    rng = np.random.default_rng(seed)
    rows = []
    base_time = pd.Timestamp("2010-01-01")
    for asset in range(num_assets):
        regime = np.zeros(num_steps, dtype=np.int64)
        for t in range(1, num_steps):
            if rng.random() < regime_switch_prob:
                regime[t] = 1 - regime[t - 1]
            else:
                regime[t] = regime[t - 1]
        drift = np.where(regime == 0, 0.0005, -0.00025)
        vol_state = np.where(regime == 0, 0.01, 0.03)
        noise = rng.normal(scale=vol_state, size=num_steps)
        returns = drift + noise + 0.2 * np.sin(np.linspace(0, 10, num_steps) + asset)
        price = 100 + np.cumsum(returns)
        price = np.clip(price, 1.0, None)
        volume = 1e6 * (1 + 0.2 * rng.normal(size=num_steps) + 0.15 * regime)
        spread = np.clip(0.01 + 0.02 * vol_state + 0.002 * rng.normal(size=num_steps), 0.001, None)
        imbalance = np.tanh(0.6 * np.roll(returns, 1) + 0.4 * rng.normal(size=num_steps))
        imbalance[0] = 0.0
        realized_vol = pd.Series(returns).rolling(10, min_periods=1).std(ddof=0).fillna(0.0).to_numpy() * np.sqrt(252)
        macro = 0.5 * np.sin(np.linspace(0, 4, num_steps)) + 0.15 * np.cos(np.linspace(0, 17, num_steps))
        features = {
            "timestamp": pd.date_range(base_time, periods=num_steps, freq="D"),
            "asset_id": asset,
            "close": price,
            "return": returns,
            "volume": volume,
            "spread": spread,
            "imbalance": imbalance,
            "realized_vol": realized_vol,
            "macro_1": macro,
            "macro_2": np.sin(np.linspace(0, 12, num_steps)) + 0.1 * rng.normal(size=num_steps),
            "macro_3": regime,
        }
        for j in range(max(0, feature_dim - 9)):
            features[f"feat_{j}"] = 0.2 * rng.normal(size=num_steps) + 0.1 * macro + 0.05 * vol_state
        rows.append(pd.DataFrame(features))
    df = pd.concat(rows, ignore_index=True)
    df["asset_id"] = df["asset_id"].astype(int)
    return df


def load_macrodata_panel() -> pd.DataFrame:
    if sm is None:
        raise ImportError("statsmodels is required for macrodata benchmark.")
    raw = sm.datasets.macrodata.load_pandas().data.copy()
    raw["timestamp"] = _make_timestamp_index(len(raw), freq="Q")
    raw["asset_id"] = 0
    numeric_cols = [c for c in raw.columns if c not in {"timestamp", "asset_id"}]
    raw["gdp_growth"] = raw["realgdp"].pct_change().fillna(0.0)
    raw["infl_change"] = raw["infl"].diff().fillna(0.0)
    raw["unemp_change"] = raw["unemp"].diff().fillna(0.0)
    raw["vol_proxy"] = raw["realgdp"].pct_change().abs().rolling(4, min_periods=1).mean().fillna(0.0)
    return raw[["timestamp", "asset_id"] + numeric_cols + ["gdp_growth", "infl_change", "unemp_change", "vol_proxy"]]


def load_csv_panel(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_parquet_panel(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def _coerce_frame(df: pd.DataFrame, timestamp_col: str, asset_col: str) -> pd.DataFrame:
    df = df.copy()
    if timestamp_col in df.columns:
        ts = pd.to_datetime(df[timestamp_col], errors="coerce")
        if ts.isna().all():
            raise ValueError(f"Could not parse any timestamps from {timestamp_col}")
        df[timestamp_col] = ts.ffill().bfill()
    if asset_col not in df.columns:
        df[asset_col] = 0
    df[asset_col] = pd.factorize(df[asset_col])[0].astype(int)
    return df


def _infer_feature_cols(df: pd.DataFrame, timestamp_col: str, asset_col: str):
    exclude = {timestamp_col, asset_col}
    cols = [c for c in df.columns if c not in exclude]
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]


class MarketWindowDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, feature_cols: Sequence[str], context_length: int, horizons: Sequence[int], target_window: int, timestamp_col: str = "timestamp", asset_col: str = "asset_id", normalize: bool = True, normalizer: Optional[FitNormalizer] = None):
        self.frame = frame.copy()
        self.feature_cols = list(feature_cols)
        self.context_length = int(context_length)
        self.horizons = list(sorted(set(int(h) for h in horizons)))
        self.target_window = int(target_window)
        self.timestamp_col = timestamp_col
        self.asset_col = asset_col
        self.normalize = normalize
        self.normalizer = normalizer
        self.assets = sorted(self.frame[self.asset_col].unique())
        self.asset_to_idx = {a: i for i, a in enumerate(self.assets)}
        if self.normalize and self.normalizer is not None:
            self.frame.loc[:, self.feature_cols] = self.normalizer.transform(self.frame[self.feature_cols].to_numpy()).astype(np.float32)
        self.series = {}
        for asset, grp in self.frame.groupby(self.asset_col, sort=True):
            self.series[asset] = grp.sort_values(self.timestamp_col).reset_index(drop=True)
        self.windows = []
        self._prepare()

    def _prepare(self):
        for asset in self.assets:
            df_a = self.series[asset]
            max_h = max(self.horizons)
            last_start = len(df_a) - (self.context_length + max_h + self.target_window) + 1
            for start in range(max(0, last_start + 1)):
                ctx_start = start
                ctx_end = start + self.context_length
                future_windows = []
                ok = True
                for h in self.horizons:
                    fut_start = ctx_end + h - 1
                    fut_end = fut_start + self.target_window
                    if fut_end > len(df_a):
                        ok = False
                        break
                    future_windows.append((fut_start, fut_end))
                if ok:
                    self.windows.append((asset, ctx_start, ctx_end, future_windows))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        asset, ctx_start, ctx_end, future_windows = self.windows[idx]
        df_a = self.series[asset]
        ctx = df_a.iloc[ctx_start:ctx_end][self.feature_cols].to_numpy(dtype=np.float32)
        futures = []
        for fut_start, fut_end in future_windows:
            fut = df_a.iloc[fut_start:fut_end][self.feature_cols].to_numpy(dtype=np.float32)
            futures.append(torch.from_numpy(fut))
        ts = np.array([pd.Timestamp(df_a.iloc[ctx_end - 1][self.timestamp_col]).value / 1e9], dtype=np.float32)
        asset_idx = np.array([self.asset_to_idx[asset]], dtype=np.int64)
        return {
            "context": torch.from_numpy(ctx),
            "futures": torch.stack(futures, dim=0),
            "asset_id": torch.from_numpy(asset_idx),
            "timestamp": torch.from_numpy(ts),
            "horizons": torch.tensor(self.horizons, dtype=torch.long),
        }


def split_panel_by_time(df: pd.DataFrame, timestamp_col: str, train_frac: float, val_frac: float):
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return df.iloc[:n_train], df.iloc[n_train:n_train+n_val], df.iloc[n_train+n_val:]


def build_datasets(config: FIJEPAConfig, df: Optional[pd.DataFrame] = None):
    if config.data.source == "synthetic":
        df = generate_synthetic_market(config.data.synthetic.num_assets, config.data.synthetic.num_steps, config.data.synthetic.feature_dim, config.data.synthetic.regime_switch_prob, seed=config.seed)
    elif config.data.source == "macrodata":
        df = load_macrodata_panel()
    elif df is None:
        if config.data.csv_path is not None:
            df = load_csv_panel(config.data.csv_path)
        elif config.data.parquet_path is not None:
            df = load_parquet_panel(config.data.parquet_path)
        else:
            raise ValueError("csv_path or parquet_path is required when source is not synthetic/macrodata and no dataframe is provided.")

    df = _coerce_frame(df, config.data.timestamp_col, config.data.asset_col)
    if config.data.resample_freq is not None and config.data.timestamp_col in df.columns:
        pieces = []
        for asset, grp in df.groupby(config.data.asset_col, sort=True):
            grp = grp.sort_values(config.data.timestamp_col).set_index(config.data.timestamp_col)
            grp = grp.resample(config.data.resample_freq).mean(numeric_only=True).interpolate(limit_direction="both")
            grp[config.data.asset_col] = asset
            pieces.append(grp.reset_index())
        df = pd.concat(pieces, ignore_index=True)

    if config.data.feature_cols is None:
        feature_cols = _infer_feature_cols(df, config.data.timestamp_col, config.data.asset_col)
    else:
        feature_cols = list(config.data.feature_cols)

    df = df.sort_values([config.data.asset_col, config.data.timestamp_col]).reset_index(drop=True)
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[feature_cols] = df[feature_cols].ffill().bfill().fillna(0.0).astype(np.float32)

    train_parts = []
    val_parts = []
    test_parts = []
    for asset, grp in df.groupby(config.data.asset_col, sort=True):
        n = len(grp)
        n_train = int(n * config.data.train_frac)
        n_val = int(n * config.data.val_frac)
        train_parts.append(grp.iloc[:n_train])
        val_parts.append(grp.iloc[n_train:n_train+n_val])
        test_parts.append(grp.iloc[n_train+n_val:])
    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    normalizer = FitNormalizer() if config.data.normalize else None
    if normalizer is not None:
        normalizer.fit(train_df[feature_cols].to_numpy())

    train_ds = MarketWindowDataset(train_df, feature_cols, config.data.context_length, config.data.horizons, config.data.target_window, timestamp_col=config.data.timestamp_col, asset_col=config.data.asset_col, normalize=config.data.normalize, normalizer=normalizer)
    val_ds = MarketWindowDataset(val_df, feature_cols, config.data.context_length, config.data.horizons, config.data.target_window, timestamp_col=config.data.timestamp_col, asset_col=config.data.asset_col, normalize=config.data.normalize, normalizer=normalizer)
    test_ds = MarketWindowDataset(test_df, feature_cols, config.data.context_length, config.data.horizons, config.data.target_window, timestamp_col=config.data.timestamp_col, asset_col=config.data.asset_col, normalize=config.data.normalize, normalizer=normalizer)
    return train_ds, val_ds, test_ds, feature_cols, normalizer


def collate_market_batch(batch: List[dict]) -> dict:
    out = {}
    for key in batch[0].keys():
        out[key] = torch.stack([b[key] for b in batch], dim=0)
    return out


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, num_workers: int = 0):
    pin_memory = torch.cuda.is_available()
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_market_batch)


SyntheticMarketData = generate_synthetic_market
