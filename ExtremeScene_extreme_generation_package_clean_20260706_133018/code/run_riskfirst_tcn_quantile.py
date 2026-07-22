from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as sps
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from evaluate_generation import EvalConfig, evaluate_generation
from risk_metrics import soft_core_risk_metrics_torch, soft_risk_metrics_torch
from risk_ranking_utils import add_risk_score, write_risk_tables
from run_model_pool_experiments import ChannelScaler, _set_seed
from run_month_evt_copula_risk_selection import (
    MonthEvtCopulaConfig,
    _add_month_season,
    _compute_monthly_tau,
    _load_split,
    _risk_metrics_for_samples,
    _safe_quantile,
    _safe_scale,
    get_au_season,
)
from run_simple_evt_risk_diffusion import FULL_COMPARE_METRICS


BASE_DIR = Path(__file__).resolve().parent

RISKFIRST_EMPIRICAL = "RiskFirst_TCN_Empirical"
RISKFIRST_EVT = "RiskFirst_TCN_EVT"
RISKFIRST_QUANTILE = "RiskFirst_TCN_QuantileLoss"
CALIBRATED_RISKFIRST_EMPIRICAL = "Calibrated_RiskFirst_TCN_Empirical"
RISKFIRST_VALSELECTED = "RiskFirst_TCN_ValSelectedPrior"
DURATION_AWARE_RISKFIRST = "DurationAware_RiskFirst_TCN_Empirical"
DURATION_AWARE_CALIBRATED_RISKFIRST = "DurationAware_Calibrated_RiskFirst_TCN_Empirical"

RISK_COLS = ["cum_deficit", "core_cum_deficit", "netload_ramp_max", "imbalance_duration"]
RISK_TARGET_ALIASES = {
    "cum_deficit": "target_cum",
    "core_cum_deficit": "target_core",
    "netload_ramp_max": "target_ramp",
    "imbalance_duration": "target_duration",
}


DATASET_PATHS = {
    "singleton": BASE_DIR / "outputs" / "ramp_window_retrain" / "datasets" / "dataset_window_3h",
    "muswellbrook": BASE_DIR / "outputs" / "muswellbrook_ramp_window_retrain" / "datasets" / "dataset_window_3h",
    "cessnock_or_newarea": BASE_DIR / "outputs" / "cessnock_south_ramp_window_retrain" / "datasets" / "dataset_window_3h",
    "openenergyhub_caiso_balanced_relaxed": BASE_DIR / "outputs" / "openenergyhub_caiso_threshold_sweep" / "balanced_relaxed" / "dataset",
}


@dataclass
class RiskFirstConfig:
    """RiskFirst TCN 运行配置。

    这些参数只作用于新增方法，不改变已有 Copula / TCN / ValSelected 的结果复现。
    """

    out_dir: Path = BASE_DIR / "results" / "riskfirst_tcn_quantile"
    dataset: str = "muswellbrook"
    seed: int = 42
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1.0e-3
    k_candidates: int = 50
    hidden_dim: int = 64
    num_layers: int = 3
    kernel_size: int = 3
    dropout: float = 0.10
    noise_dim: int = 16
    lambda_risk: float = 1.0
    lambda_quantile: float = 0.5
    lambda_q99: float = 1.0
    lambda_phy: float = 0.02
    lambda_duration_soft: float = 0.35
    duration_soft_beta: float = 0.08
    duration_temp: float = 12.0
    delta_t_hours: float = 1.0
    min_group_samples: int = 10
    min_event_samples: int = 15
    min_month_samples: int = 15
    min_season_samples: int = 30
    tau_quantile: float = 0.75
    evt_threshold_quantile: float = 0.75
    evt_min_tail: int = 12
    candidate_weights: tuple[float, float, float, float] = (0.40, 0.30, 0.15, 0.15)
    duration_aware_candidate_weights: tuple[float, float, float, float, float] = (0.25, 0.20, 0.20, 0.25, 0.10)
    topk_per_sample: int = 5
    device: str = "cpu"
    methods: tuple[str, ...] = (
        RISKFIRST_EMPIRICAL,
        RISKFIRST_EVT,
        RISKFIRST_QUANTILE,
        CALIBRATED_RISKFIRST_EMPIRICAL,
        DURATION_AWARE_RISKFIRST,
    )


def _device_from_request(text: str) -> torch.device:
    if str(text).lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _is_duration_aware_method(method: str) -> bool:
    return method in {DURATION_AWARE_RISKFIRST, DURATION_AWARE_CALIBRATED_RISKFIRST}


def _ensure_condition(cond: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """合并 cond/meta，保证 month、season、event_type 等条件字段可用。"""

    out = cond.reset_index(drop=True).copy()
    meta = meta.reset_index(drop=True)
    for col in meta.columns:
        if col not in out.columns:
            out[col] = meta[col].to_numpy()
    out = _add_month_season(out, meta)
    if "event_type" not in out.columns:
        out["event_type"] = "unknown"
    if "extreme_prob" not in out.columns:
        out["extreme_prob"] = 0.9
    return out


def _build_train_risk_table_with_event(
    x_train: np.ndarray,
    cond_train: pd.DataFrame,
    mask_train: np.ndarray | None,
    tau_by_month: dict[int, float],
    cfg: RiskFirstConfig,
) -> pd.DataFrame:
    """计算训练集联合失衡风险向量 R，并保留 month/event_type 便于构造风险先验。"""

    risk = _risk_metrics_for_samples(
        x_train,
        cond_train["month"].astype(int).to_numpy(),
        tau_by_month,
        mask_train,
        float(cfg.delta_t_hours),
    )
    risk = _attach_max_imbalance_run(
        risk,
        x_train,
        cond_train["month"].astype(int).to_numpy(),
        tau_by_month,
        float(cfg.delta_t_hours),
    )
    keep = pd.DataFrame(
        {
            "month": cond_train["month"].astype(int).to_numpy(),
            "au_season": cond_train["au_season"].astype(str).to_numpy(),
            "event_type": cond_train["event_type"].astype(str).fillna("unknown").to_numpy(),
        }
    )
    return pd.concat([keep.reset_index(drop=True), risk.reset_index(drop=True)], axis=1)


def _max_true_run(mask: np.ndarray) -> int:
    """计算布尔序列中最长连续 True 段长度，用于刻画持续失衡是否成段出现。"""

    best = 0
    cur = 0
    for item in np.asarray(mask, dtype=bool):
        if item:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def _max_imbalance_run_for_samples(
    x: np.ndarray,
    months: np.ndarray,
    tau_by_month: dict[int, float],
    delta_t: float,
) -> np.ndarray:
    """计算每个样本净负荷连续超过月度阈值的最长持续段。

    max_imbalance_run 是 duration 的形态补充：duration 只看总时长，本指标进一步区分
    “连续成段失衡”和“零散多点失衡”。
    """

    net = x[:, 0, :] - x[:, 1, :] - x[:, 2, :]
    fallback_tau = float(np.nanmean(list(tau_by_month.values()))) if tau_by_month else 0.0
    values = []
    for i in range(x.shape[0]):
        tau = float(tau_by_month.get(int(months[i]), fallback_tau))
        run_steps = _max_true_run(net[i] > tau)
        values.append(float(run_steps) * float(delta_t))
    return np.asarray(values, dtype=float)


def _attach_max_imbalance_run(
    risk: pd.DataFrame,
    x: np.ndarray,
    months: np.ndarray,
    tau_by_month: dict[int, float],
    delta_t: float,
) -> pd.DataFrame:
    out = risk.copy()
    out["max_imbalance_run"] = _max_imbalance_run_for_samples(x, months, tau_by_month, delta_t)
    return out


def _risk_stats(train_risk: pd.DataFrame) -> dict[str, dict[str, float]]:
    """训练集风险向量归一化参数，只由训练集计算。"""

    stats: dict[str, dict[str, float]] = {}
    for col in RISK_COLS:
        values = train_risk[col].to_numpy(dtype=float)
        scale = _safe_scale(values, fallback=float(np.std(values) + 1e-6))
        stats[col] = {"mean": float(np.nanmean(values)), "scale": float(max(scale, 1e-6))}
    return stats


def _normalize_risk_frame(df: pd.DataFrame, stats: dict[str, dict[str, float]]) -> np.ndarray:
    arr = []
    for col in RISK_COLS:
        arr.append((pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(float) - stats[col]["mean"]) / stats[col]["scale"])
    return np.stack(arr, axis=1).astype(np.float32)


class EmpiricalRiskTargetBuilder:
    """根据 month、event_type、extreme_prob 构造目标风险向量 R_target。

    R_target 是 RiskFirst 方法的核心先验：模型不再只看样本级条件，而是显式看见
    [累计缺额、核心段累计缺额、三小时爬坡、持续失衡时长] 的目标水平。
    """

    def __init__(self, train_risk: pd.DataFrame, cfg: RiskFirstConfig, dataset: str):
        self.train_risk = train_risk.reset_index(drop=True)
        self.cfg = cfg
        self.dataset = dataset

    def _quantile_level(self, row: pd.Series) -> float:
        p = pd.to_numeric(pd.Series([row.get("extreme_prob", np.nan)]), errors="coerce").iloc[0]
        if not np.isfinite(p):
            p = 0.9
        return float(np.clip(float(p), 0.50, 0.99))

    def _select_group(self, month: int, event_type: str) -> tuple[pd.DataFrame, str, str, str]:
        month = int(month)
        event_type = str(event_type)
        df = self.train_risk
        group = df[(df["month"].astype(int) == month) & (df["event_type"].astype(str) == event_type)]
        if len(group) >= int(self.cfg.min_group_samples):
            return group, "month_event_type", f"{month}:{event_type}", ""
        group = df[df["event_type"].astype(str) == event_type]
        if len(group) >= int(self.cfg.min_event_samples):
            return group, "event_type", event_type, f"month_event samples insufficient for {month}:{event_type}"
        group = df[df["month"].astype(int) == month]
        if len(group) >= int(self.cfg.min_month_samples):
            return group, "month", str(month), f"event samples insufficient for {event_type}"
        season = get_au_season(month)
        group = df[df["au_season"].astype(str) == season]
        if len(group) >= int(self.cfg.min_season_samples):
            return group, "season", season, f"month samples insufficient for {month}"
        return df, "global", "all", "month/event/season samples insufficient"

    def target_for_row(self, sample_id: int, row: pd.Series) -> tuple[dict[str, float | int | str], dict[str, float]]:
        month = int(row.get("month", 1))
        event_type = str(row.get("event_type", "unknown"))
        q = self._quantile_level(row)
        group, group_type, group_name, reason = self._select_group(month, event_type)
        target: dict[str, float] = {col: _safe_quantile(group[col].to_numpy(float), q) for col in RISK_COLS}
        scales = {
            col: _safe_scale(group[col].to_numpy(float), fallback=_safe_scale(self.train_risk[col].to_numpy(float)))
            for col in RISK_COLS
        }
        if "max_imbalance_run" in group.columns and "max_imbalance_run" in self.train_risk.columns:
            target["max_imbalance_run"] = _safe_quantile(group["max_imbalance_run"].to_numpy(float), q)
            scales["max_imbalance_run"] = _safe_scale(
                group["max_imbalance_run"].to_numpy(float),
                fallback=_safe_scale(self.train_risk["max_imbalance_run"].to_numpy(float)),
            )
        info: dict[str, float | int | str] = {
            "dataset": self.dataset,
            "sample_id": int(sample_id),
            "month": month,
            "season": get_au_season(month),
            "event_type": event_type,
            "extreme_prob": float(row.get("extreme_prob", 0.9)),
            "quantile_level": q,
            "target_cum": target["cum_deficit"],
            "target_core": target["core_cum_deficit"],
            "target_ramp": target["netload_ramp_max"],
            "target_duration": target["imbalance_duration"],
            "target_max_imbalance_run": target.get("max_imbalance_run", target["imbalance_duration"]),
            "group_type": group_type,
            "group_name": group_name,
            "fallback_reason": reason,
        }
        return info, scales

    def build_frame(self, cond: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, float]]]:
        rows = []
        scales = []
        for i, row in cond.reset_index(drop=True).iterrows():
            info, scale = self.target_for_row(i, row)
            rows.append(info)
            scales.append(scale)
        return pd.DataFrame(rows), scales


class CalibratedEmpiricalRiskTargetBuilder(EmpiricalRiskTargetBuilder):
    """带验证集校准的经验风险先验。

    a/b 用于校准 extreme_prob 到风险分位数的映射；
    scale_cum/scale_core 用于把验证集目标 q99 对齐到验证集真实 q99。
    """

    def __init__(
        self,
        train_risk: pd.DataFrame,
        cfg: RiskFirstConfig,
        dataset: str,
        quantile_a: float = 1.0,
        quantile_b: float = 0.0,
        scale_cum: float = 1.0,
        scale_core: float = 1.0,
    ):
        super().__init__(train_risk, cfg, dataset)
        self.quantile_a = float(quantile_a)
        self.quantile_b = float(quantile_b)
        self.scale_cum = float(scale_cum)
        self.scale_core = float(scale_core)

    def _quantile_level(self, row: pd.Series) -> float:
        p = pd.to_numeric(pd.Series([row.get("extreme_prob", np.nan)]), errors="coerce").iloc[0]
        if not np.isfinite(p):
            p = 0.9
        # extreme_prob 到目标风险分位数的验证集校准映射。
        return float(np.clip(self.quantile_a * float(p) + self.quantile_b, 0.50, 0.99))

    def target_for_row(self, sample_id: int, row: pd.Series) -> tuple[dict[str, float | int | str], dict[str, float]]:
        info, scales = super().target_for_row(sample_id, row)
        info["target_cum"] = float(info["target_cum"]) * self.scale_cum
        info["target_core"] = float(info["target_core"]) * self.scale_core
        info["calibration_a"] = self.quantile_a
        info["calibration_b"] = self.quantile_b
        info["scale_cum"] = self.scale_cum
        info["scale_core"] = self.scale_core
        return info, scales


class EVTRiskTargetBuilder(EmpiricalRiskTargetBuilder):
    """对 cum/core 使用 POT-GPD 尾部外推的风险先验。

    如果尾部样本不足或 GPD 拟合失败，自动退回经验分位，避免小样本下不稳定。
    """

    def __init__(self, train_risk: pd.DataFrame, cfg: RiskFirstConfig, dataset: str):
        super().__init__(train_risk, cfg, dataset)
        self.fit_summary: list[dict] = []
        self.evt_models: dict[str, dict[str, float | bool | str]] = {}
        for col in ["cum_deficit", "core_cum_deficit"]:
            self.evt_models[col] = self._fit_evt(col)

    def _fit_evt(self, col: str) -> dict[str, float | bool | str]:
        values = self.train_risk[col].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        threshold = _safe_quantile(values, float(self.cfg.evt_threshold_quantile))
        exceed = values[values > threshold] - threshold
        row: dict[str, float | bool | str] = {
            "dataset": self.dataset,
            "risk_metric": col,
            "threshold": float(threshold),
            "n_tail": int(len(exceed)),
            "shape": float("nan"),
            "scale": float("nan"),
            "fit_status": "fallback_empirical",
            "fallback_used": True,
            "notes": "",
        }
        if len(exceed) < int(self.cfg.evt_min_tail) or np.nanstd(exceed) <= 1e-8:
            row["notes"] = "tail samples too few or nearly constant"
            self.fit_summary.append(row)
            return row
        try:
            shape, loc, scale = sps.genpareto.fit(exceed, floc=0.0)
            if not np.isfinite(shape) or not np.isfinite(scale) or scale <= 0:
                raise ValueError("invalid GPD parameters")
            row.update({"shape": float(shape), "scale": float(scale), "fit_status": "ok", "fallback_used": False, "notes": ""})
        except Exception as exc:  # noqa: BLE001
            row["notes"] = f"fit failed: {exc}"
        self.fit_summary.append(row)
        return row

    def _evt_quantile(self, col: str, q: float, empirical_group: pd.DataFrame) -> float:
        model = self.evt_models.get(col, {})
        if bool(model.get("fallback_used", True)):
            return _safe_quantile(empirical_group[col].to_numpy(float), q)
        threshold_q = float(self.cfg.evt_threshold_quantile)
        if q <= threshold_q:
            return _safe_quantile(empirical_group[col].to_numpy(float), q)
        # POT 条件分位：P(X <= x | X > u) = (q - q_u) / (1 - q_u)
        tail_p = float(np.clip((q - threshold_q) / max(1.0 - threshold_q, 1e-6), 0.0, 0.999))
        y = sps.genpareto.ppf(tail_p, float(model["shape"]), loc=0.0, scale=float(model["scale"]))
        return float(float(model["threshold"]) + max(float(y), 0.0))

    def target_for_row(self, sample_id: int, row: pd.Series) -> tuple[dict[str, float | int | str], dict[str, float]]:
        info, scales = super().target_for_row(sample_id, row)
        month = int(row.get("month", 1))
        event_type = str(row.get("event_type", "unknown"))
        q = float(info["quantile_level"])
        group, _, _, _ = self._select_group(month, event_type)
        info["target_cum"] = self._evt_quantile("cum_deficit", q, group)
        info["target_core"] = self._evt_quantile("core_cum_deficit", q, group)
        info["group_type"] = f"evt_plus_{info['group_type']}"
        return info, scales


class RiskFirstConditionEncoder:
    """RiskFirst 条件编码器。

    条件包括 month_sin/month_cos、event_type、extreme_prob、duration_norm，
    以及归一化后的 R_target 四维风险向量。
    """

    def __init__(self, cond_train: pd.DataFrame, risk_stats: dict[str, dict[str, float]]):
        event = cond_train.get("event_type", pd.Series(["unknown"] * len(cond_train))).astype(str).fillna("unknown")
        self.event_values = sorted(event.unique().tolist())
        self.event_to_idx = {v: i for i, v in enumerate(self.event_values)}
        if "duration_norm" in cond_train.columns:
            duration = pd.to_numeric(cond_train["duration_norm"], errors="coerce").fillna(0.0)
            self.duration_scale = 1.0
            self.duration_col = "duration_norm"
        else:
            duration = pd.to_numeric(cond_train.get("duration_hours", pd.Series([1.0] * len(cond_train))), errors="coerce").fillna(1.0)
            self.duration_scale = float(max(duration.max(), 1.0))
            self.duration_col = "duration_hours"
        self.risk_stats = risk_stats

    @property
    def dim(self) -> int:
        return len(self.event_values) + 4 + 4

    def transform(self, cond: pd.DataFrame, target_df: pd.DataFrame) -> np.ndarray:
        cond = cond.reset_index(drop=True)
        target_df = target_df.reset_index(drop=True)
        event = cond.get("event_type", pd.Series(["unknown"] * len(cond))).astype(str).fillna("unknown")
        onehot = np.zeros((len(cond), len(self.event_values)), dtype=np.float32)
        for i, value in enumerate(event):
            onehot[i, self.event_to_idx.get(str(value), 0)] = 1.0
        month = pd.to_numeric(cond.get("month", 1), errors="coerce").fillna(1).to_numpy(float)
        month_rad = 2.0 * np.pi * (month - 1.0) / 12.0
        duration = pd.to_numeric(cond.get(self.duration_col, 0.0), errors="coerce").fillna(0.0).to_numpy(float)
        if self.duration_col != "duration_norm":
            duration = duration / max(self.duration_scale, 1e-6)
        extreme = pd.to_numeric(cond.get("extreme_prob", 0.9), errors="coerce").fillna(0.9).to_numpy(float)
        scalars = np.stack(
            [
                np.sin(month_rad),
                np.cos(month_rad),
                np.clip(duration, 0.0, 2.0),
                np.clip(extreme, 0.0, 1.0),
            ],
            axis=1,
        ).astype(np.float32)
        risk_norm = []
        for col in RISK_COLS:
            alias = RISK_TARGET_ALIASES[col]
            values = pd.to_numeric(target_df[alias], errors="coerce").fillna(0.0).to_numpy(float)
            risk_norm.append((values - self.risk_stats[col]["mean"]) / self.risk_stats[col]["scale"])
        risk_norm_arr = np.stack(risk_norm, axis=1).astype(np.float32)
        return np.concatenate([onehot, scalars, risk_norm_arr], axis=1).astype(np.float32)


class TCNBlock(nn.Module):
    """轻量 TCN 残差块，用扩张卷积捕捉 36h 内跨时间步风险演化。"""

    def __init__(self, hidden_dim: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        pad = int((kernel_size - 1) * dilation)
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, dilation=dilation, padding=pad)
        self.dropout = nn.Dropout(float(dropout))
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        trim = (self.kernel_size - 1) * self.dilation
        if trim > 0:
            y = y[..., :-trim]
        y = self.dropout(F.silu(y))
        return x + y


class RiskFirstTCN(nn.Module):
    """? R_target ?????????? TCN ????"""

    def __init__(
        self,
        cond_dim: int,
        seq_len: int,
        hidden_dim: int,
        noise_dim: int,
        num_layers: int,
        kernel_size: int,
        dropout: float,
        use_event_mask_channel: bool = False,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.noise_dim = int(noise_dim)
        self.use_event_mask_channel = bool(use_event_mask_channel)
        # event_mask ???????????DurationAware ???????????????
        # ????????????????????????????? 36h ???
        extra_channels = 1 if self.use_event_mask_channel else 0
        self.in_proj = nn.Conv1d(noise_dim + cond_dim + extra_channels, hidden_dim, kernel_size=1)
        dilations = [2**i for i in range(int(num_layers))]
        self.blocks = nn.ModuleList([TCNBlock(hidden_dim, kernel_size, d, dropout) for d in dilations])
        self.out = nn.Conv1d(hidden_dim, 3, kernel_size=1)

    def forward(self, noise: torch.Tensor, cond: torch.Tensor, event_mask: torch.Tensor | None = None) -> torch.Tensor:
        # risk condition ???????????? 36h ??????????????
        cond_seq = cond[:, :, None].expand(-1, -1, noise.shape[-1])
        parts = [noise, cond_seq]
        if self.use_event_mask_channel:
            # event_mask????????????????????????????????
            if event_mask is None or event_mask.ndim != 2 or event_mask.shape[1] != noise.shape[-1]:
                event_mask = torch.ones((noise.shape[0], noise.shape[-1]), dtype=noise.dtype, device=noise.device)
            parts.append(event_mask.to(dtype=noise.dtype, device=noise.device)[:, None, :])
        h = self.in_proj(torch.cat(parts, dim=1))
        for block in self.blocks:
            h = block(h)
        return self.out(h)


def _targets_to_raw_matrix(target_df: pd.DataFrame) -> np.ndarray:
    return target_df[["target_cum", "target_core", "target_ramp", "target_duration"]].to_numpy(dtype=np.float32)


def _raw_to_norm_tensor(raw: torch.Tensor, stats: dict[str, dict[str, float]]) -> torch.Tensor:
    means = torch.tensor([stats[col]["mean"] for col in RISK_COLS], dtype=raw.dtype, device=raw.device)
    scales = torch.tensor([stats[col]["scale"] for col in RISK_COLS], dtype=raw.dtype, device=raw.device)
    return (raw - means.view(1, -1)) / scales.view(1, -1)


def _hard_risk_frame_for_x(
    x: np.ndarray,
    cond: pd.DataFrame,
    mask: np.ndarray | None,
    tau_by_month: dict[int, float],
    delta_t: float,
) -> pd.DataFrame:
    months = pd.to_numeric(cond.get("month", 1), errors="coerce").fillna(1).astype(int).to_numpy()
    risk = _risk_metrics_for_samples(x, months, tau_by_month, mask, delta_t)
    return _attach_max_imbalance_run(risk, x, months, tau_by_month, delta_t)


def _q99(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(np.quantile(arr, 0.99))


def _select_calibration(
    dataset: str,
    train_risk: pd.DataFrame,
    cond_val: pd.DataFrame,
    x_val: np.ndarray,
    mask_val: np.ndarray | None,
    tau_by_month: dict[int, float],
    cfg: RiskFirstConfig,
    out_dir: Path,
) -> tuple[CalibratedEmpiricalRiskTargetBuilder, pd.DataFrame, pd.DataFrame]:
    """使用验证集选择 extreme_prob->quantile 映射，并校准 cum/core 目标尺度。

    验证集只用于选择超参数和目标缩放，不参与模型训练或测试集评价。
    """

    real_val = _hard_risk_frame_for_x(x_val, cond_val, mask_val, tau_by_month, float(cfg.delta_t_hours))
    real_q99_cum = _q99(real_val["cum_deficit"].to_numpy(float))
    real_q99_core = _q99(real_val["core_cum_deficit"].to_numpy(float))
    rows = []
    best: dict | None = None
    for a in [0.6, 0.8, 1.0, 1.2]:
        for b in [-0.10, -0.05, 0.0, 0.05]:
            builder = CalibratedEmpiricalRiskTargetBuilder(train_risk, cfg, dataset, quantile_a=a, quantile_b=b)
            target_val, _ = builder.build_frame(cond_val)
            target_q99_cum = _q99(target_val["target_cum"].to_numpy(float))
            target_q99_core = _q99(target_val["target_core"].to_numpy(float))
            err_cum = abs(real_q99_cum - target_q99_cum) / max(_safe_scale(train_risk["cum_deficit"].to_numpy(float)), 1e-6)
            err_core = abs(real_q99_core - target_q99_core) / max(_safe_scale(train_risk["core_cum_deficit"].to_numpy(float)), 1e-6)
            score = float(err_cum + err_core)
            row = {
                "dataset": dataset,
                "a": float(a),
                "b": float(b),
                "real_val_q99_cum": real_q99_cum,
                "target_val_q99_cum": target_q99_cum,
                "real_val_q99_core": real_q99_core,
                "target_val_q99_core": target_q99_core,
                "val_q99_target_error": score,
                "selected": False,
            }
            rows.append(row)
            if best is None or score < float(best["val_q99_target_error"]):
                best = row
    assert best is not None
    for row in rows:
        row["selected"] = bool(row["a"] == best["a"] and row["b"] == best["b"])
    selection_df = pd.DataFrame(rows)
    selection_df.to_csv(out_dir / "calibrated_quantile_mapping_selection.csv", index=False, encoding="utf-8-sig")
    raw_builder = CalibratedEmpiricalRiskTargetBuilder(
        train_risk,
        cfg,
        dataset,
        quantile_a=float(best["a"]),
        quantile_b=float(best["b"]),
        scale_cum=1.0,
        scale_core=1.0,
    )
    raw_target_val, _ = raw_builder.build_frame(cond_val)
    target_q99_cum = max(_q99(raw_target_val["target_cum"].to_numpy(float)), 1e-6)
    target_q99_core = max(_q99(raw_target_val["target_core"].to_numpy(float)), 1e-6)
    scale_cum = float(np.clip(real_q99_cum / target_q99_cum, 0.6, 1.6))
    scale_core = float(np.clip(real_q99_core / target_q99_core, 0.6, 1.6))
    calibrated = CalibratedEmpiricalRiskTargetBuilder(
        train_risk,
        cfg,
        dataset,
        quantile_a=float(best["a"]),
        quantile_b=float(best["b"]),
        scale_cum=scale_cum,
        scale_core=scale_core,
    )
    scaled_target_val, _ = calibrated.build_frame(cond_val)
    diag = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "split": "val",
                "method": CALIBRATED_RISKFIRST_EMPIRICAL,
                "selected_a": float(best["a"]),
                "selected_b": float(best["b"]),
                "scale_cum": scale_cum,
                "scale_core": scale_core,
                "real_q99_cum": real_q99_cum,
                "target_q99_cum": _q99(scaled_target_val["target_cum"].to_numpy(float)),
                "gen_q99_cum": np.nan,
                "real_q99_core": real_q99_core,
                "target_q99_core": _q99(scaled_target_val["target_core"].to_numpy(float)),
                "gen_q99_core": np.nan,
                "target_error": abs(real_q99_cum - _q99(scaled_target_val["target_cum"].to_numpy(float)))
                + abs(real_q99_core - _q99(scaled_target_val["target_core"].to_numpy(float))),
                "generator_error": np.nan,
            }
        ]
    )
    return calibrated, selection_df, diag


def _physics_project(x: np.ndarray, channel_max: np.ndarray) -> np.ndarray:
    # 物理后处理：风光荷非负，并限制到训练集合理上界附近，避免 TCN 生成离谱尖峰。
    y = np.asarray(x, dtype=np.float32)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.maximum(y, 0.0)
    cap = channel_max.reshape(1, 3, 1) * 1.25
    return np.minimum(y, cap).astype(np.float32)


def train_riskfirst_model(
    method: str,
    x_train: np.ndarray,
    cond_train: pd.DataFrame,
    mask_train: np.ndarray | None,
    tau_by_month: dict[int, float],
    target_df: pd.DataFrame,
    risk_stats: dict[str, dict[str, float]],
    encoder: RiskFirstConditionEncoder,
    cfg: RiskFirstConfig,
    log_path: Path,
) -> tuple[RiskFirstTCN, ChannelScaler]:
    device = _device_from_request(cfg.device)
    scaler = ChannelScaler().fit(x_train)
    x_norm = scaler.transform(x_train)
    cond_feat = encoder.transform(cond_train, target_df)
    target_raw = _targets_to_raw_matrix(target_df)
    target_norm = _normalize_target_raw(target_raw, risk_stats)
    months = cond_train["month"].astype(int).to_numpy(np.int64)
    masks = mask_train if mask_train is not None else np.ones((len(x_train), x_train.shape[-1]), dtype=np.float32)
    ds = TensorDataset(
        torch.from_numpy(x_norm.astype(np.float32)),
        torch.from_numpy(cond_feat.astype(np.float32)),
        torch.from_numpy(months),
        torch.from_numpy(masks.astype(np.float32)),
        torch.from_numpy(target_norm.astype(np.float32)),
        torch.from_numpy(target_raw.astype(np.float32)),
    )
    loader = DataLoader(ds, batch_size=int(cfg.batch_size), shuffle=True)
    model = RiskFirstTCN(
        cond_dim=cond_feat.shape[1],
        seq_len=x_train.shape[-1],
        hidden_dim=int(cfg.hidden_dim),
        noise_dim=int(cfg.noise_dim),
        num_layers=int(cfg.num_layers),
        kernel_size=int(cfg.kernel_size),
        dropout=float(cfg.dropout),
        use_event_mask_channel=_is_duration_aware_method(method),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.lr))
    mean_t = torch.from_numpy(scaler.mean).to(device)
    std_t = torch.from_numpy(scaler.std).to(device)
    channel_max = torch.from_numpy(scaler.channel_max).to(device) * 1.25
    tau_lookup = {int(k): float(v) for k, v in tau_by_month.items()}
    log_rows = []
    for epoch in range(1, int(cfg.epochs) + 1):
        losses = []
        for xb, cb, mb, eb, tb_norm, tb_raw in loader:
            xb, cb, mb, eb, tb_norm, tb_raw = xb.to(device), cb.to(device), mb.to(device), eb.to(device), tb_norm.to(device), tb_raw.to(device)
            noise = torch.randn((xb.shape[0], int(cfg.noise_dim), xb.shape[-1]), device=device)
            pred_norm = model(noise, cb, eb if _is_duration_aware_method(method) else None)
            pred_denorm = pred_norm * std_t + mean_t
            true_denorm = xb * std_t + mean_t
            tau = torch.tensor([tau_lookup.get(int(m.item()), np.median(list(tau_lookup.values()))) for m in mb], dtype=pred_norm.dtype, device=device)
            cum_p, ramp_p, dur_p = soft_risk_metrics_torch(
                pred_denorm,
                tau,
                delta_t_hours=float(cfg.delta_t_hours),
                duration_temp=float(cfg.duration_temp),
                ramp_metric_mode="window_3h",
                ramp_window_hours=3.0,
            )
            core_p, _, _ = soft_core_risk_metrics_torch(
                pred_denorm,
                tau,
                eb,
                delta_t_hours=float(cfg.delta_t_hours),
                duration_temp=float(cfg.duration_temp),
            )
            risk_raw_p = torch.stack([cum_p, core_p, ramp_p, dur_p], dim=1)
            risk_norm_p = _raw_to_norm_tensor(risk_raw_p, risk_stats)
            shape_loss = F.mse_loss(pred_norm, xb)
            risk_loss = F.smooth_l1_loss(risk_norm_p, tb_norm)
            duration_soft_loss = pred_norm.new_tensor(0.0)
            if _is_duration_aware_method(method):
                # soft imbalance duration loss:
                # p_imb(t)=sigmoid((net_load(t)-tau_month)/beta)，用于直接约束持续超过阈值的软时长。
                beta = max(float(cfg.duration_soft_beta), 1e-6)
                net_p = pred_denorm[:, 0, :] - pred_denorm[:, 1, :] - pred_denorm[:, 2, :]
                p_imb = torch.sigmoid((net_p - tau.view(-1, 1)) / beta)
                soft_duration = p_imb.sum(dim=1) * float(cfg.delta_t_hours)
                dur_mean = float(risk_stats["imbalance_duration"]["mean"])
                dur_scale = max(float(risk_stats["imbalance_duration"]["scale"]), 1e-6)
                duration_soft_loss = F.smooth_l1_loss(
                    (soft_duration - dur_mean) / dur_scale,
                    (tb_raw[:, 3] - dur_mean) / dur_scale,
                )
            q_loss = pred_norm.new_tensor(0.0)
            q99_loss = pred_norm.new_tensor(0.0)
            if method == RISKFIRST_QUANTILE:
                with torch.no_grad():
                    cum_t, ramp_t, dur_t = soft_risk_metrics_torch(
                        true_denorm,
                        tau,
                        delta_t_hours=float(cfg.delta_t_hours),
                        duration_temp=float(cfg.duration_temp),
                        ramp_metric_mode="window_3h",
                        ramp_window_hours=3.0,
                    )
                    core_t, _, _ = soft_core_risk_metrics_torch(
                        true_denorm,
                        tau,
                        eb,
                        delta_t_hours=float(cfg.delta_t_hours),
                        duration_temp=float(cfg.duration_temp),
                    )
                q_loss = _quantile_consistency_loss(cum_p, core_p, ramp_p, dur_p, cum_t, core_t, ramp_t, dur_t, risk_stats)
                q99_loss = _q99_tail_loss(cum_p, core_p, cum_t, core_t, risk_stats)
            phy_loss = torch.relu(-pred_denorm).mean() + 0.1 * torch.relu(pred_denorm - channel_max).mean()
            loss = (
                shape_loss
                + float(cfg.lambda_risk) * risk_loss
                + float(cfg.lambda_quantile) * q_loss
                + float(cfg.lambda_q99) * q99_loss
                + float(cfg.lambda_duration_soft) * duration_soft_loss
                + float(cfg.lambda_phy) * phy_loss
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(
                [
                    float(loss.detach().cpu()),
                    float(shape_loss.detach().cpu()),
                    float(risk_loss.detach().cpu()),
                    float(q_loss.detach().cpu()),
                    float(q99_loss.detach().cpu()),
                    float(duration_soft_loss.detach().cpu()),
                    float(phy_loss.detach().cpu()),
                ]
            )
        mean_losses = np.asarray(losses, dtype=float).mean(axis=0)
        log_rows.append(
            {
                "method": method,
                "epoch": epoch,
                "loss": mean_losses[0],
                "shape_loss": mean_losses[1],
                "risk_loss": mean_losses[2],
                "quantile_loss": mean_losses[3],
                "q99_loss": mean_losses[4],
                "duration_soft_loss": mean_losses[5],
                "physics_loss": mean_losses[6],
            }
        )
    pd.DataFrame(log_rows).to_csv(log_path, index=False, encoding="utf-8-sig")
    model.eval()
    return model, scaler


def _normalize_target_raw(raw: np.ndarray, stats: dict[str, dict[str, float]]) -> np.ndarray:
    out = []
    for j, col in enumerate(RISK_COLS):
        out.append((raw[:, j] - stats[col]["mean"]) / stats[col]["scale"])
    return np.stack(out, axis=1).astype(np.float32)


def _quantile_diff(a: torch.Tensor, b: torch.Tensor, q: float, scale: float) -> torch.Tensor:
    if a.numel() == 0 or b.numel() == 0:
        return a.new_tensor(0.0)
    return torch.abs(torch.quantile(a, q) - torch.quantile(b, q)) / max(float(scale), 1e-6)


def _quantile_consistency_loss(
    cum_p: torch.Tensor,
    core_p: torch.Tensor,
    ramp_p: torch.Tensor,
    dur_p: torch.Tensor,
    cum_t: torch.Tensor,
    core_t: torch.Tensor,
    ramp_t: torch.Tensor,
    dur_t: torch.Tensor,
    stats: dict[str, dict[str, float]],
) -> torch.Tensor:
    qs = [0.90, 0.95, 0.99] if cum_p.numel() >= 16 else [0.90, 0.95]
    loss = cum_p.new_tensor(0.0)
    for q in qs:
        loss = loss + _quantile_diff(cum_p, cum_t, q, stats["cum_deficit"]["scale"])
        loss = loss + _quantile_diff(core_p, core_t, q, stats["core_cum_deficit"]["scale"])
    loss = loss + _quantile_diff(ramp_p, ramp_t, 0.95, stats["netload_ramp_max"]["scale"])
    loss = loss + _quantile_diff(dur_p, dur_t, 0.95, stats["imbalance_duration"]["scale"])
    return loss


def _q99_tail_loss(
    cum_p: torch.Tensor,
    core_p: torch.Tensor,
    cum_t: torch.Tensor,
    core_t: torch.Tensor,
    stats: dict[str, dict[str, float]],
) -> torch.Tensor:
    q = 0.99 if cum_p.numel() >= 16 else 0.95
    return _quantile_diff(cum_p, cum_t, q, stats["cum_deficit"]["scale"]) + _quantile_diff(core_p, core_t, q, stats["core_cum_deficit"]["scale"])


@torch.no_grad()
def generate_with_selection(
    method: str,
    model: RiskFirstTCN,
    scaler: ChannelScaler,
    encoder: RiskFirstConditionEncoder,
    target_builder: EmpiricalRiskTargetBuilder,
    cond: pd.DataFrame,
    mask: np.ndarray | None,
    tau_by_month: dict[int, float],
    cfg: RiskFirstConfig,
    out_dir: Path,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """每个测试条件生成 K 个候选，再按 R_target 风险距离筛选最终样本。"""

    device = next(model.parameters()).device
    channel_max = scaler.channel_max.reshape(3)
    selected = []
    logs = []
    target_df, scales_list = target_builder.build_frame(cond)
    for i, row in cond.reset_index(drop=True).iterrows():
        row_df = pd.DataFrame([row] * int(cfg.k_candidates))
        target_i = pd.DataFrame([target_df.iloc[i].to_dict()] * int(cfg.k_candidates))
        cond_feat = encoder.transform(row_df, target_i)
        c = torch.from_numpy(cond_feat.astype(np.float32)).to(device)
        noise = torch.randn((int(cfg.k_candidates), int(cfg.noise_dim), int(model.seq_len)), device=device)
        event_for_model = None
        if getattr(model, "use_event_mask_channel", False):
            if mask is not None:
                event_for_model = torch.from_numpy(np.repeat(mask[i : i + 1], int(cfg.k_candidates), axis=0).astype(np.float32)).to(device)
            else:
                event_for_model = torch.ones((int(cfg.k_candidates), int(model.seq_len)), dtype=torch.float32, device=device)
        pred_norm = model(noise, c, event_for_model).cpu().numpy()
        candidates = scaler.inverse(pred_norm)
        candidates = _physics_project(candidates, channel_max)
        mask_i = np.repeat(mask[i : i + 1], int(cfg.k_candidates), axis=0) if mask is not None else None
        cand_cond = pd.DataFrame([row] * int(cfg.k_candidates)).reset_index(drop=True)
        cand_metrics = _hard_risk_frame_for_x(candidates, cand_cond, mask_i, tau_by_month, float(cfg.delta_t_hours))
        if _is_duration_aware_method(method):
            score = _duration_aware_candidate_score(cand_metrics, target_df.iloc[i].to_dict(), scales_list[i], cfg.duration_aware_candidate_weights)
        else:
            score = _candidate_score(cand_metrics, target_df.iloc[i].to_dict(), scales_list[i], cfg.candidate_weights)
        j = int(np.nanargmin(score)) if len(score) else 0
        selected.append(candidates[j])
        log = {
            "method": method,
            "sample_id": int(i),
            "month": int(row.get("month", 1)),
            "event_type": str(row.get("event_type", "unknown")),
            "selected_candidate_index": j,
            "selected_score": float(score[j]),
            "selected_cum": float(cand_metrics.loc[j, "cum_deficit"]),
            "selected_core_cum": float(cand_metrics.loc[j, "core_cum_deficit"]),
            "selected_ramp": float(cand_metrics.loc[j, "netload_ramp_max"]),
            "selected_duration": float(cand_metrics.loc[j, "imbalance_duration"]),
            "selected_max_imbalance_run": float(cand_metrics.loc[j, "max_imbalance_run"]) if "max_imbalance_run" in cand_metrics.columns else np.nan,
            "target_cum": float(target_df.loc[i, "target_cum"]),
            "target_core": float(target_df.loc[i, "target_core"]),
            "target_ramp": float(target_df.loc[i, "target_ramp"]),
            "target_duration": float(target_df.loc[i, "target_duration"]),
            "target_max_imbalance_run": float(target_df.loc[i, "target_max_imbalance_run"]) if "target_max_imbalance_run" in target_df.columns else np.nan,
        }
        logs.append(log)
    arr = np.stack(selected, axis=0).astype(np.float32)
    pd.DataFrame(logs).to_csv(out_dir / f"candidate_selection_log_{method}.csv", index=False, encoding="utf-8-sig")
    return arr, pd.DataFrame(logs), target_df


@torch.no_grad()
def generate_with_calibrated_two_layer_selection(
    method: str,
    model: RiskFirstTCN,
    scaler: ChannelScaler,
    encoder: RiskFirstConditionEncoder,
    target_builder: CalibratedEmpiricalRiskTargetBuilder,
    cond: pd.DataFrame,
    mask: np.ndarray | None,
    tau_by_month: dict[int, float],
    cfg: RiskFirstConfig,
    out_dir: Path,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Calibrated RiskFirst 的两层候选筛选。

    第一层：每个样本按 R_target 误差保留 top-5。
    第二层：在 top-5 组合空间中贪心调整，使全测试集 q99_cum/q99_core 更接近目标分布。
    """

    device = next(model.parameters()).device
    channel_max = scaler.channel_max.reshape(3)
    target_df, scales_list = target_builder.build_frame(cond)
    top_candidates: list[np.ndarray] = []
    top_metrics: list[pd.DataFrame] = []
    top_scores: list[np.ndarray] = []
    logs = []
    k_keep = max(1, int(cfg.topk_per_sample))
    for i, row in cond.reset_index(drop=True).iterrows():
        row_df = pd.DataFrame([row] * int(cfg.k_candidates))
        target_i = pd.DataFrame([target_df.iloc[i].to_dict()] * int(cfg.k_candidates))
        cond_feat = encoder.transform(row_df, target_i)
        c = torch.from_numpy(cond_feat.astype(np.float32)).to(device)
        noise = torch.randn((int(cfg.k_candidates), int(cfg.noise_dim), int(model.seq_len)), device=device)
        pred_norm = model(noise, c).cpu().numpy()
        candidates = _physics_project(scaler.inverse(pred_norm), channel_max)
        mask_i = np.repeat(mask[i : i + 1], int(cfg.k_candidates), axis=0) if mask is not None else None
        cand_cond = pd.DataFrame([row] * int(cfg.k_candidates)).reset_index(drop=True)
        cand_metrics = _hard_risk_frame_for_x(candidates, cand_cond, mask_i, tau_by_month, float(cfg.delta_t_hours))
        score = _candidate_score(cand_metrics, target_df.iloc[i].to_dict(), scales_list[i], cfg.candidate_weights)
        order = np.argsort(score)[: min(k_keep, len(score))]
        top_candidates.append(candidates[order])
        top_metrics.append(cand_metrics.iloc[order].reset_index(drop=True))
        top_scores.append(score[order])
        logs.append(
            {
                "method": method,
                "sample_id": int(i),
                "month": int(row.get("month", 1)),
                "event_type": str(row.get("event_type", "unknown")),
                "topk_kept": int(len(order)),
                "best_first_layer_score": float(score[order[0]]) if len(order) else np.nan,
                "target_cum": float(target_df.loc[i, "target_cum"]),
                "target_core": float(target_df.loc[i, "target_core"]),
                "target_ramp": float(target_df.loc[i, "target_ramp"]),
                "target_duration": float(target_df.loc[i, "target_duration"]),
            }
        )
    selected_idx = _global_q99_greedy_selection(top_metrics, top_scores, target_df, scales_list)
    selected = []
    final_logs = []
    for i, j in enumerate(selected_idx):
        selected.append(top_candidates[i][j])
        m = top_metrics[i].iloc[j]
        row = dict(logs[i])
        row.update(
            {
                "selected_topk_index": int(j),
                "selected_score": float(top_scores[i][j]),
                "selected_cum": float(m["cum_deficit"]),
                "selected_core_cum": float(m["core_cum_deficit"]),
                "selected_ramp": float(m["netload_ramp_max"]),
                "selected_duration": float(m["imbalance_duration"]),
            }
        )
        final_logs.append(row)
    arr = np.stack(selected, axis=0).astype(np.float32)
    log_df = pd.DataFrame(final_logs)
    log_df.to_csv(out_dir / f"candidate_selection_log_{method}.csv", index=False, encoding="utf-8-sig")
    return arr, log_df, target_df


def _global_q99_greedy_selection(
    top_metrics: list[pd.DataFrame],
    top_scores: list[np.ndarray],
    target_df: pd.DataFrame,
    scales_list: list[dict[str, float]],
) -> list[int]:
    if not top_metrics:
        return []
    selected = [0 for _ in top_metrics]
    target_q99_cum = _q99(target_df["target_cum"].to_numpy(float))
    target_q99_core = _q99(target_df["target_core"].to_numpy(float))
    scale_cum = max(np.nanmean([s.get("cum_deficit", 1.0) for s in scales_list]), 1e-6)
    scale_core = max(np.nanmean([s.get("core_cum_deficit", 1.0) for s in scales_list]), 1e-6)

    def objective(indices: list[int]) -> float:
        cum = np.asarray([top_metrics[i].iloc[j]["cum_deficit"] for i, j in enumerate(indices)], dtype=float)
        core = np.asarray([top_metrics[i].iloc[j]["core_cum_deficit"] for i, j in enumerate(indices)], dtype=float)
        score_mean = float(np.mean([top_scores[i][j] for i, j in enumerate(indices)]))
        return (
            abs(_q99(cum) - target_q99_cum) / scale_cum
            + abs(_q99(core) - target_q99_core) / scale_core
            + 0.05 * score_mean
        )

    best_obj = objective(selected)
    for _ in range(4):
        improved = False
        for i in range(len(top_metrics)):
            local_best = selected[i]
            local_obj = best_obj
            for j in range(len(top_metrics[i])):
                trial = list(selected)
                trial[i] = j
                obj = objective(trial)
                if obj + 1e-9 < local_obj:
                    local_best = j
                    local_obj = obj
            if local_best != selected[i]:
                selected[i] = local_best
                best_obj = local_obj
                improved = True
        if not improved:
            break
    return selected


def _candidate_score(
    cand_metrics: pd.DataFrame,
    target: dict,
    scales: dict[str, float],
    weights: tuple[float, float, float, float],
) -> np.ndarray:
    w_cum, w_core, w_ramp, w_dur = weights
    err_cum = np.abs(cand_metrics["cum_deficit"].to_numpy(float) - float(target["target_cum"])) / max(scales["cum_deficit"], 1e-6)
    err_core = np.abs(cand_metrics["core_cum_deficit"].to_numpy(float) - float(target["target_core"])) / max(scales["core_cum_deficit"], 1e-6)
    err_ramp = np.abs(cand_metrics["netload_ramp_max"].to_numpy(float) - float(target["target_ramp"])) / max(scales["netload_ramp_max"], 1e-6)
    err_dur = np.abs(cand_metrics["imbalance_duration"].to_numpy(float) - float(target["target_duration"])) / max(scales["imbalance_duration"], 1e-6)
    return w_cum * err_cum + w_core * err_core + w_ramp * err_ramp + w_dur * err_dur


def _duration_aware_candidate_score(
    cand_metrics: pd.DataFrame,
    target: dict,
    scales: dict[str, float],
    weights: tuple[float, float, float, float, float],
) -> np.ndarray:
    """DurationAware 两层思想中的单样本候选筛选分数。

    除 cum/core/ramp/duration 外，额外加入 max_imbalance_run：
    它刻画净负荷连续超过阈值的最长段，避免只匹配总失衡时长但生成成零散点。
    """

    w_cum, w_core, w_ramp, w_dur, w_run = weights
    base = _candidate_score(cand_metrics, target, scales, (w_cum, w_core, w_ramp, w_dur))
    if "max_imbalance_run" not in cand_metrics.columns:
        return base
    target_run = float(target.get("target_max_imbalance_run", target.get("target_duration", 0.0)))
    scale_run = max(float(scales.get("max_imbalance_run", scales.get("imbalance_duration", 1.0))), 1e-6)
    err_run = np.abs(cand_metrics["max_imbalance_run"].to_numpy(float) - target_run) / scale_run
    return base + w_run * err_run


def _write_calibrated_diagnostic(
    dataset: str,
    split: str,
    method: str,
    real_x: np.ndarray,
    gen_x: np.ndarray | None,
    cond: pd.DataFrame,
    mask: np.ndarray | None,
    target_df: pd.DataFrame,
    tau_by_month: dict[int, float],
    calibration_diag: pd.DataFrame | None,
    out_dir: Path,
    cfg: RiskFirstConfig,
) -> pd.DataFrame:
    real_risk = _hard_risk_frame_for_x(real_x, cond, mask, tau_by_month, float(cfg.delta_t_hours))
    gen_q99_cum = np.nan
    gen_q99_core = np.nan
    if gen_x is not None:
        gen_risk = _hard_risk_frame_for_x(gen_x, cond, mask, tau_by_month, float(cfg.delta_t_hours))
        gen_q99_cum = _q99(gen_risk["cum_deficit"].to_numpy(float))
        gen_q99_core = _q99(gen_risk["core_cum_deficit"].to_numpy(float))
    real_q99_cum = _q99(real_risk["cum_deficit"].to_numpy(float))
    real_q99_core = _q99(real_risk["core_cum_deficit"].to_numpy(float))
    target_q99_cum = _q99(target_df["target_cum"].to_numpy(float))
    target_q99_core = _q99(target_df["target_core"].to_numpy(float))
    row = {
        "dataset": dataset,
        "split": split,
        "method": method,
        "real_q99_cum": real_q99_cum,
        "target_q99_cum": target_q99_cum,
        "gen_q99_cum": gen_q99_cum,
        "real_q99_core": real_q99_core,
        "target_q99_core": target_q99_core,
        "gen_q99_core": gen_q99_core,
        "target_error": abs(real_q99_cum - target_q99_cum) + abs(real_q99_core - target_q99_core),
        "generator_error": np.nan
        if gen_x is None
        else abs(real_q99_cum - gen_q99_cum) + abs(real_q99_core - gen_q99_core),
    }
    if calibration_diag is not None and len(calibration_diag):
        for col in ["selected_a", "selected_b", "scale_cum", "scale_core"]:
            if col in calibration_diag.columns:
                row[col] = calibration_diag.iloc[0][col]
    df = pd.DataFrame([row])
    path = out_dir / "calibrated_q99_diagnostic.csv"
    if path.exists():
        old = pd.read_csv(path)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df


def _evaluate_generated(method: str, gen_path: Path, data_dir: Path, out_dir: Path, tau_by_month: dict[int, float] | None = None, delta_t: float = 1.0) -> dict:
    return _evaluate_generated_split(method, gen_path, data_dir, out_dir, split="test", tau_by_month=tau_by_month, delta_t=delta_t)


def _evaluate_generated_split(
    method: str,
    gen_path: Path,
    data_dir: Path,
    out_dir: Path,
    split: str = "test",
    tau_by_month: dict[int, float] | None = None,
    delta_t: float = 1.0,
) -> dict:
    event_mask_path = data_dir / f"event_mask_{split}.npy"
    eval_out_dir = out_dir / method if split == "test" else out_dir / f"{split}_evaluation" / method
    summary = evaluate_generation(
        EvalConfig(
            real=str(data_dir / f"X_{split}.npy"),
            generated=str(gen_path),
            cond=str(data_dir / f"cond_{split}.csv"),
            meta=str(data_dir / f"meta_{split}.csv"),
            out_dir=str(eval_out_dir),
            model_name=method,
            event_mask=str(event_mask_path) if event_mask_path.exists() else None,
            ramp_metric_mode="window_3h",
            ramp_window_hours=3.0,
        )
    )
    metrics = dict(summary.get("metrics", {}))
    if tau_by_month is not None and gen_path.exists():
        try:
            real_x = np.load(data_dir / f"X_{split}.npy")
            gen_x = np.load(gen_path)
            cond_raw = pd.read_csv(data_dir / f"cond_{split}.csv")
            meta_path = data_dir / f"meta_{split}.csv"
            meta = pd.read_csv(meta_path) if meta_path.exists() else pd.DataFrame(index=np.arange(len(cond_raw)))
            cond = _ensure_condition(cond_raw, meta)
            months = pd.to_numeric(cond.get("month", 1), errors="coerce").fillna(1).astype(int).to_numpy()
            real_run = _max_imbalance_run_for_samples(real_x, months, tau_by_month, float(delta_t))
            gen_run = _max_imbalance_run_for_samples(gen_x, months, tau_by_month, float(delta_t))
            metrics["max_imbalance_run_mae"] = float(np.mean(np.abs(gen_run - real_run)))
        except Exception as exc:  # noqa: BLE001
            metrics["max_imbalance_run_mae"] = np.nan
            metrics["max_imbalance_run_error"] = str(exc)
    metrics["method"] = method
    metrics["eval_split"] = split
    return metrics


def _load_baseline_rows(dataset: str) -> pd.DataFrame:
    path = BASE_DIR / "results" / "final_full_baseline_compare" / dataset / "compare_all_methods.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    keep_methods = [
        "traditional_gaussian_copula",
        "TailWeighted_Month_EVT_Copula_Risk_Selection_Fixed",
        "Conditional_TCN_Risk_Generator",
        "RiskKNN_Bootstrap_Generator",
        "ValSelected_Model_Ensemble_Margin_0.05",
    ]
    return df[df["method"].isin(keep_methods)].copy()


def run_one_dataset(dataset: str, cfg: RiskFirstConfig, quick: bool = True) -> tuple[pd.DataFrame, Path]:
    data_dir = DATASET_PATHS[dataset]
    quick_name = "muswellbrook_quick"
    if quick and dataset == "muswellbrook" and abs(float(cfg.lambda_q99) - 1.0) > 1e-9:
        quick_name = f"muswellbrook_quick_q99x{str(float(cfg.lambda_q99)).replace('.', 'p')}"
    out_dir = cfg.out_dir / quick_name if quick and dataset == "muswellbrook" else cfg.out_dir / "formal" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    diag_path = out_dir / "calibrated_q99_diagnostic.csv"
    if diag_path.exists():
        diag_path.unlink()
    x_train, cond_train_raw, meta_train, mask_train = _load_split(data_dir, "train")
    x_val, cond_val_raw, meta_val, mask_val = _load_split(data_dir, "val")
    x_test, cond_test_raw, meta_test, mask_test = _load_split(data_dir, "test")
    cond_train = _ensure_condition(cond_train_raw, meta_train)
    cond_val = _ensure_condition(cond_val_raw, meta_val)
    cond_test = _ensure_condition(cond_test_raw, meta_test)
    tau_cfg = MonthEvtCopulaConfig(
        out_dir=out_dir,
        tau_quantile=float(cfg.tau_quantile),
        min_month_samples=int(cfg.min_month_samples),
        min_season_samples=int(cfg.min_season_samples),
        delta_t_hours=float(cfg.delta_t_hours),
        seed=int(cfg.seed),
    )
    tau_by_month, tau_df = _compute_monthly_tau(x_train, cond_train, tau_cfg, dataset, out_dir)
    train_risk = _build_train_risk_table_with_event(x_train, cond_train, mask_train, tau_by_month, cfg)
    train_risk.to_csv(out_dir / "train_risk_table.csv", index=False, encoding="utf-8-sig")
    stats = _risk_stats(train_risk)
    (out_dir / "risk_normalization_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    encoder = RiskFirstConditionEncoder(cond_train, stats)

    empirical_builder = EmpiricalRiskTargetBuilder(train_risk, cfg, dataset)
    evt_builder = EVTRiskTargetBuilder(train_risk, cfg, dataset)
    calibrated_builder, calibration_grid, calibration_diag_val = _select_calibration(
        dataset,
        train_risk,
        cond_val,
        x_val,
        mask_val,
        tau_by_month,
        cfg,
        out_dir,
    )

    target_train_emp, _ = empirical_builder.build_frame(cond_train)
    target_test_emp, _ = empirical_builder.build_frame(cond_test)
    target_test_emp.to_csv(out_dir / "risk_target_empirical_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(evt_builder.fit_summary).to_csv(out_dir / "evt_risk_prior_summary.csv", index=False, encoding="utf-8-sig")

    rows = []
    val_rows = []
    test_rows_by_method: dict[str, dict] = {}
    quantile_logs = []
    method_builders = {
        RISKFIRST_EMPIRICAL: empirical_builder,
        RISKFIRST_EVT: evt_builder,
        RISKFIRST_QUANTILE: empirical_builder,
        CALIBRATED_RISKFIRST_EMPIRICAL: calibrated_builder,
        DURATION_AWARE_RISKFIRST: empirical_builder,
        DURATION_AWARE_CALIBRATED_RISKFIRST: calibrated_builder,
    }
    method_builders = {k: v for k, v in method_builders.items() if k in set(cfg.methods)}
    for method, builder in method_builders.items():
        _set_seed(int(cfg.seed))
        train_target, _ = builder.build_frame(cond_train)
        log_path = out_dir / ("quantile_loss_training_log.csv" if method == RISKFIRST_QUANTILE else f"training_log_{method}.csv")
        model, scaler = train_riskfirst_model(
            method,
            x_train,
            cond_train,
            mask_train,
            tau_by_month,
            train_target,
            stats,
            encoder,
            cfg,
            log_path,
        )
        val_log_dir = out_dir / "val_candidate_selection"
        val_log_dir.mkdir(parents=True, exist_ok=True)
        if method == CALIBRATED_RISKFIRST_EMPIRICAL:
            gen_val, _val_selection_log, val_target_df = generate_with_calibrated_two_layer_selection(
                method,
                model,
                scaler,
                encoder,
                calibrated_builder,
                cond_val,
                mask_val,
                tau_by_month,
                cfg,
                val_log_dir,
            )
        else:
            gen_val, _val_selection_log, val_target_df = generate_with_selection(
                method,
                model,
                scaler,
                encoder,
                builder,
                cond_val,
                mask_val,
                tau_by_month,
                cfg,
                val_log_dir,
            )
        val_gen_path = out_dir / f"generated_val_samples_{method}.npy"
        np.save(val_gen_path, gen_val.astype(np.float32))
        val_target_df.to_csv(out_dir / f"risk_target_val_{method}.csv", index=False, encoding="utf-8-sig")
        val_rows.append(_evaluate_generated_split(method, val_gen_path, data_dir, out_dir, split="val", tau_by_month=tau_by_month, delta_t=float(cfg.delta_t_hours)))

        if method == CALIBRATED_RISKFIRST_EMPIRICAL:
            gen, selection_log, target_df = generate_with_calibrated_two_layer_selection(
                method,
                model,
                scaler,
                encoder,
                calibrated_builder,
                cond_test,
                mask_test,
                tau_by_month,
                cfg,
                out_dir,
            )
        else:
            gen, selection_log, target_df = generate_with_selection(
                method,
                model,
                scaler,
                encoder,
                builder,
                cond_test,
                mask_test,
                tau_by_month,
                cfg,
                out_dir,
            )
        gen_path = out_dir / f"generated_samples_{method}.npy"
        np.save(gen_path, gen.astype(np.float32))
        target_df.to_csv(out_dir / f"risk_target_{method}.csv", index=False, encoding="utf-8-sig")
        if method == CALIBRATED_RISKFIRST_EMPIRICAL:
            _write_calibrated_diagnostic(
                dataset,
                "val",
                method,
                x_val,
                None,
                cond_val,
                mask_val,
                calibrated_builder.build_frame(cond_val)[0],
                tau_by_month,
                calibration_diag_val,
                out_dir,
                cfg,
            )
            _write_calibrated_diagnostic(
                dataset,
                "test",
                method,
                x_test,
                gen,
                cond_test,
                mask_test,
                target_df,
                tau_by_month,
                calibration_diag_val,
                out_dir,
                cfg,
            )
        test_metrics = _evaluate_generated(method, gen_path, data_dir, out_dir, tau_by_month=tau_by_month, delta_t=float(cfg.delta_t_hours))
        rows.append(test_metrics)
        test_rows_by_method[method] = dict(test_metrics)

    val_compare = add_risk_score(pd.DataFrame(val_rows))
    val_compare.to_csv(out_dir / "riskfirst_val_selection_metrics.csv", index=False, encoding="utf-8-sig")
    selected_method = str(val_compare.sort_values(["risk_score", "risk_rank"]).iloc[0]["method"])
    selected_test = dict(test_rows_by_method[selected_method])
    selected_test["method"] = RISKFIRST_VALSELECTED
    selected_test["selected_source_method"] = selected_method
    rows.append(selected_test)
    src_gen = out_dir / f"generated_samples_{selected_method}.npy"
    if src_gen.exists():
        selected_arr = np.load(src_gen).astype(np.float32)
        np.save(out_dir / f"generated_samples_{RISKFIRST_VALSELECTED}.npy", selected_arr)
    pd.DataFrame(
        [
            {
                "dataset": dataset,
                "selected_method": selected_method,
                "selected_val_risk_score": float(val_compare.sort_values(["risk_score", "risk_rank"]).iloc[0]["risk_score"]),
                "selected_val_risk_rank": int(val_compare.sort_values(["risk_score", "risk_rank"]).iloc[0]["risk_rank"]),
            }
        ]
    ).to_csv(out_dir / "riskfirst_val_selected_prior_summary.csv", index=False, encoding="utf-8-sig")

    new_df = pd.DataFrame(rows)
    base_df = _load_baseline_rows(dataset)
    compare = pd.concat([base_df, new_df], ignore_index=True, sort=False)
    compare = add_risk_score(compare)
    compare.to_csv(out_dir / "compare_all_methods.csv", index=False, encoding="utf-8-sig")
    write_risk_tables(compare, out_dir)
    _write_fairness_report(dataset, data_dir, out_dir, cfg, bool(not base_df.empty))
    _write_method_report(dataset, out_dir, compare)
    return compare, out_dir


def _write_fairness_report(dataset: str, data_dir: Path, out_dir: Path, cfg: RiskFirstConfig, baseline_loaded: bool) -> None:
    text = f"""# RiskFirst TCN Fairness Check

- dataset: {dataset}
- dataset_path: `{data_dir}`
- train/val/test: 使用当前 `dataset_window_3h` 固定划分。
- monthly tau: 仅由训练集计算，测试集只复用训练集 tau。
- risk target: 仅由训练集风险分布和测试条件中的 month/event_type/extreme_prob 决定。
- candidate selection: 只使用 R_target 和候选自身风险，不使用测试真实曲线。
- ramp metric: `window_3h`，即三小时窗口最大正向净负荷爬坡。
- baseline source: {"loaded from current final_full_baseline_compare table" if baseline_loaded else "not found in final_full_baseline_compare"}。

本 quick 版本的 traditional / TailWeighted / TCN / RiskKNN / ValSelected 对照行来自当前项目已生成的统一对照表；
RiskFirst 三个新方法在本脚本中重新训练和评价，所有新增 risk_score 在本表内重新统一归一化。
"""
    (cfg.out_dir / "fairness_check_report.md").parent.mkdir(parents=True, exist_ok=True)
    (cfg.out_dir / "fairness_check_report.md").write_text(text, encoding="utf-8")
    (out_dir / "fairness_check_report.md").write_text(text, encoding="utf-8")


def _write_method_report(dataset: str, out_dir: Path, compare: pd.DataFrame) -> None:
    risk_cols = ["method", "q99_cum_deficit_error", "core_q99_cum_deficit_error", "netload_ramp_max_mae", "imbalance_duration_mae", "max_imbalance_run_mae", "risk_score", "risk_rank", "extreme_degree_match_rate"]
    table = compare[[c for c in risk_cols if c in compare.columns]].copy()
    best = table.sort_values("risk_score").iloc[0].to_dict() if "risk_score" in table.columns and len(table) else {}
    trad = table[table["method"] == "traditional_gaussian_copula"].iloc[0].to_dict() if (table["method"] == "traditional_gaussian_copula").any() else {}
    qloss = table[table["method"] == RISKFIRST_QUANTILE].iloc[0].to_dict() if (table["method"] == RISKFIRST_QUANTILE).any() else {}
    calibrated = table[table["method"] == CALIBRATED_RISKFIRST_EMPIRICAL].iloc[0].to_dict() if (table["method"] == CALIBRATED_RISKFIRST_EMPIRICAL).any() else {}
    lines = [
        f"# RiskFirst TCN QuantileLoss Report - {dataset}",
        "",
        "## 方法说明",
        "- RiskFirst_TCN_Empirical：由 month/event_type/extreme_prob 映射到经验 R_target，再条件生成。",
        "- RiskFirst_TCN_EVT：cum/core 使用 POT-GPD 尾部先验，尾部不足时回退经验分位。",
        "- RiskFirst_TCN_QuantileLoss：在 Empirical R_target 条件上加入 batch 分位数一致性和 q99 加强项。",
        "- Calibrated_RiskFirst_TCN_Empirical：在验证集上校准 extreme_prob->quantile 映射和 cum/core 目标尺度，并使用 top-5 + 全局 q99/core 二层候选筛选。",
        "",
        "## 关键结论",
        f"- 当前 risk_score 最优方法：{best.get('method', 'NA')}",
        f"- traditional q99：{trad.get('q99_cum_deficit_error', 'NA')}",
        f"- RiskFirst_TCN_QuantileLoss q99：{qloss.get('q99_cum_deficit_error', 'NA')}",
        f"- Calibrated_RiskFirst_TCN_Empirical q99：{calibrated.get('q99_cum_deficit_error', 'NA')}",
        f"- traditional core_q99：{trad.get('core_q99_cum_deficit_error', 'NA')}",
        f"- RiskFirst_TCN_QuantileLoss core_q99：{qloss.get('core_q99_cum_deficit_error', 'NA')}",
        f"- Calibrated_RiskFirst_TCN_Empirical core_q99：{calibrated.get('core_q99_cum_deficit_error', 'NA')}",
        "",
        "## 主风险指标表",
        table.to_string(index=False),
    ]
    (out_dir / "method_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RiskFirst TCN target-risk and quantile-loss experiments.")
    parser.add_argument(
        "--dataset",
        default="muswellbrook",
        choices=["muswellbrook", "singleton", "cessnock_or_newarea", "openenergyhub_caiso_balanced_relaxed", "all"],
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-candidates", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lambda-q99", type=float, default=1.0)
    parser.add_argument("--lambda-duration-soft", type=float, default=0.35)
    parser.add_argument("--duration-soft-beta", type=float, default=0.08)
    parser.add_argument(
        "--duration-aware-candidate-weights",
        default="0.25,0.20,0.20,0.25,0.10",
        help="DurationAware candidate weights: cum,core,ramp,duration,max_run.",
    )
    parser.add_argument(
        "--methods",
        default="empirical,evt,quantile,calibrated,duration_aware,duration_aware_calibrated",
        help="Comma-separated methods: empirical, evt, quantile, calibrated, duration_aware, duration_aware_calibrated, all.",
    )
    parser.add_argument("--out-dir", default=str(BASE_DIR / "results" / "riskfirst_tcn_quantile"))
    return parser.parse_args()


def _parse_method_list(text: str) -> tuple[str, ...]:
    aliases = {
        "empirical": RISKFIRST_EMPIRICAL,
        "riskfirst_tcn_empirical": RISKFIRST_EMPIRICAL,
        "evt": RISKFIRST_EVT,
        "quantile": RISKFIRST_QUANTILE,
        "quantile_loss": RISKFIRST_QUANTILE,
        "calibrated": CALIBRATED_RISKFIRST_EMPIRICAL,
        "duration": DURATION_AWARE_RISKFIRST,
        "duration_aware": DURATION_AWARE_RISKFIRST,
        "durationaware": DURATION_AWARE_RISKFIRST,
        "duration_calibrated": DURATION_AWARE_CALIBRATED_RISKFIRST,
        "duration_aware_calibrated": DURATION_AWARE_CALIBRATED_RISKFIRST,
        "durationaware_calibrated": DURATION_AWARE_CALIBRATED_RISKFIRST,
    }
    default = (
        RISKFIRST_EMPIRICAL,
        RISKFIRST_EVT,
        RISKFIRST_QUANTILE,
        CALIBRATED_RISKFIRST_EMPIRICAL,
        DURATION_AWARE_RISKFIRST,
        DURATION_AWARE_CALIBRATED_RISKFIRST,
    )
    if str(text).strip().lower() == "all":
        return default
    methods = []
    for item in str(text).split(","):
        key = item.strip().lower()
        if not key:
            continue
        methods.append(aliases.get(key, item.strip()))
    return tuple(dict.fromkeys(methods)) or default


def _parse_duration_weights(text: str) -> tuple[float, float, float, float, float]:
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if len(vals) != 5:
        raise ValueError("--duration-aware-candidate-weights must contain five comma-separated values.")
    total = sum(vals)
    if total <= 0:
        raise ValueError("--duration-aware-candidate-weights sum must be positive.")
    vals = [v / total for v in vals]
    return float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3]), float(vals[4])


def main() -> None:
    args = parse_args()
    cfg = RiskFirstConfig(
        out_dir=Path(args.out_dir),
        dataset=args.dataset,
        epochs=int(args.epochs),
        seed=int(args.seed),
        k_candidates=int(args.k_candidates),
        batch_size=int(args.batch_size),
        device=str(args.device),
        lambda_q99=float(args.lambda_q99),
        lambda_duration_soft=float(args.lambda_duration_soft),
        duration_soft_beta=float(args.duration_soft_beta),
        duration_aware_candidate_weights=_parse_duration_weights(args.duration_aware_candidate_weights),
        methods=_parse_method_list(args.methods),
    )
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    datasets = ["singleton", "muswellbrook", "cessnock_or_newarea", "openenergyhub_caiso_balanced_relaxed"] if args.dataset == "all" else [args.dataset]
    all_rows = []
    for ds in datasets:
        compare, out_dir = run_one_dataset(ds, cfg, quick=(ds == "muswellbrook" and args.dataset != "all"))
        tmp = compare.copy()
        if "dataset" in tmp.columns:
            tmp["dataset"] = tmp["dataset"].fillna(ds)
        else:
            tmp.insert(0, "dataset", ds)
        all_rows.append(tmp)
        print(f"[RiskFirst] {ds} done -> {out_dir}")
    if all_rows:
        all_df = pd.concat(all_rows, ignore_index=True, sort=False)
        all_df.to_csv(cfg.out_dir / "all_datasets_risk_summary.csv", index=False, encoding="utf-8-sig")
        rank_rows = []
        for method, g in all_df.groupby("method"):
            rank_rows.append(
                {
                    "method": method,
                    "mean_risk_rank": float(pd.to_numeric(g["risk_rank"], errors="coerce").mean()),
                    "mean_risk_score": float(pd.to_numeric(g["risk_score"], errors="coerce").mean()),
                    "available_dataset_count": int(g["dataset"].nunique()),
                }
            )
        pd.DataFrame(rank_rows).sort_values(["mean_risk_rank", "mean_risk_score"]).to_csv(
            cfg.out_dir / "all_datasets_rank_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )


if __name__ == "__main__":
    main()
