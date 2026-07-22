from __future__ import annotations

import argparse
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from run_copula_guided_residual_diffusion import DatasetSpec, _dataset_specs
from run_gan_augmented_tailweighted_copula import (
    METHOD_NAME as GAN_METHOD,
    GanAugmentedTailWeightedConfig,
    ensure_gan_augmented_val_test_paths,
)
from run_model_pool_experiments import (
    FLOW_METHOD,
    TRANSFORMER_METHOD,
    VAE_METHOD,
    ChannelScaler,
    ConditionEncoder,
    ModelPoolConfig,
    _candidate_metrics_for_pool,
    _evaluate_test,
    _load_existing_gan_paths,
    _margin_method_name,
    _run_vae_aug,
    _select_with_margin,
    _set_seed,
    _sample_flow_candidates,
    _sample_transformer_candidates,
    _train_flow,
    _train_transformer,
)
from run_month_evt_copula_risk_selection import (
    MonthEvtCopulaConfig,
    _add_month_season,
    _build_train_risk_table,
    _choose_copula_model,
    _compute_monthly_tau,
    _copula_cfg,
    _load_split,
    _parse_weights,
    _score_candidates,
    _select_candidates,
    _target_for_condition,
)
from run_simple_evt_risk_diffusion import FULL_COMPARE_METRICS
from run_tailweighted_month_evt_copula import (
    TAIL_FIXED_METHOD,
    TailWeightedConfig,
    TailWeightedGaussianCopulaGroupModel,
    _compute_tail_scores,
    _fit_tailweighted_group_copulas,
)
from traditional_statistical_extreme_baseline.traditional_copula_baseline import (
    CopulaConfig,
    EmpiricalMarginal,
    GaussianCopulaGroupModel,
    physical_projection_np,
    temporal_correction,
)


BASE_DIR = Path(__file__).resolve().parent

STUDENT_T_METHOD = "StudentT_TailWeighted_Month_EVT_Copula_Risk_Selection_Fixed"
RISK_KNN_METHOD = "RiskKNN_Bootstrap_Generator"
QUANTILE_RESAMPLER_METHOD = "QuantileRisk_Resampler"
TCN_METHOD = "Conditional_TCN_Risk_Generator"
EXPANDED_ENSEMBLE_METHOD = "Expanded_ValSelected_Model_Ensemble"
STACKED_METHOD = "Stacked_Risk_Calibrated_Selection"


@dataclass
class ExpandedModelPoolConfig(ModelPoolConfig):
    out_dir: Path = BASE_DIR / "results" / "expanded_model_pool_formal50"
    smoke_out_dir: Path = BASE_DIR / "results" / "expanded_model_pool_smoke"
    student_t_df: int = 6
    risk_knn_neighbors: int = 12
    bootstrap_jitter_scale: float = 0.05
    quantile_jitter_scale: float = 0.02
    duration_temp: float = 12.0
    tcn_epochs: int = 5
    tcn_hidden_dim: int = 64
    tcn_num_layers: int = 3
    tcn_kernel_size: int = 3
    tcn_dropout: float = 0.10
    tcn_noise_dim: int = 16
    lambda_cum: float = 0.6
    lambda_core: float = 0.4
    lambda_ramp: float = 0.2
    lambda_dur: float = 0.2
    lambda_phy: float = 0.02
    stacked_candidate_methods: tuple[str, ...] = (
        TAIL_FIXED_METHOD,
        STUDENT_T_METHOD,
        RISK_KNN_METHOD,
        QUANTILE_RESAMPLER_METHOD,
        TRANSFORMER_METHOD,
        FLOW_METHOD,
        TCN_METHOD,
    )
    run_margins: tuple[float, ...] = (0.0, 0.05, 0.10)


@dataclass
class DatasetContext:
    spec: DatasetSpec
    cfg: ExpandedModelPoolConfig
    out_dir: Path
    x_train: np.ndarray
    cond_train_raw: pd.DataFrame
    meta_train: pd.DataFrame
    mask_train: np.ndarray | None
    cond_train: pd.DataFrame
    tau_by_month: dict[int, float]
    train_risk: pd.DataFrame
    cond_encoder: ConditionEncoder
    cond_train_features: np.ndarray


class BaseScenarioGenerator(ABC):
    """统一候选生成器接口。

    fit: 在训练集上拟合生成器。
    generate: 给定目标条件，生成 [N, K, 3, 36] 的候选场景池。
    generate_selected: 根据统一风险目标筛选最终样本。
    evaluate: 统一调用评价函数生成指标。
    get_name: 返回方法名。
    """

    candidate_ready: bool = True
    priority: int = 9

    def __init__(self, ctx: DatasetContext) -> None:
        self.ctx = ctx

    @abstractmethod
    def fit(self, X_train, cond_train, X_val=None, cond_val=None) -> None:
        pass

    @abstractmethod
    def generate(self, cond_target: pd.DataFrame, meta_target: pd.DataFrame, event_mask: np.ndarray | None, n_candidates: int = 20) -> np.ndarray:
        pass

    def generate_selected(self, split: str, K_candidates: int = 20) -> tuple[Path, pd.DataFrame | None]:
        _, cond_raw, meta, mask = _load_split(self.ctx.spec.data_dir, split)
        cond = _add_month_season(cond_raw, meta)
        candidates = self.generate(cond, meta, mask, n_candidates=K_candidates)
        if candidates.ndim != 4:
            raise ValueError(f"{self.get_name()} generate must return [N,K,3,T], got {candidates.shape}")
        selected, log = _select_candidates_from_pool(
            candidates,
            cond,
            mask,
            self.ctx.train_risk,
            self.ctx.tau_by_month,
            self.ctx.cfg,
            base_method_name=self.get_name(),
        )
        path = _generated_path(self.ctx.out_dir, self.get_name(), split)
        np.save(path, selected.astype(np.float32))
        if log is not None:
            log.to_csv(_selection_log_path(self.ctx.out_dir, self.get_name(), split), index=False, encoding="utf-8-sig")
        return path, log

    def evaluate(self, generated_path: Path, split: str = "test") -> dict:
        return _evaluate_test(self.get_name(), generated_path, self.ctx.spec, self.ctx.out_dir, split=split)

    @abstractmethod
    def get_name(self) -> str:
        pass


class LegacySelectedGenerator(BaseScenarioGenerator):
    candidate_ready = False

    def __init__(self, ctx: DatasetContext, method_name: str, split_runner):
        super().__init__(ctx)
        self.method_name = method_name
        self.split_runner = split_runner

    def fit(self, X_train, cond_train, X_val=None, cond_val=None) -> None:
        return None

    def generate(self, cond_target: pd.DataFrame, meta_target: pd.DataFrame, event_mask: np.ndarray | None, n_candidates: int = 20) -> np.ndarray:
        raise NotImplementedError(f"{self.method_name} is a selected-only legacy generator.")

    def generate_selected(self, split: str, K_candidates: int = 20) -> tuple[Path, pd.DataFrame | None]:
        path = self.split_runner(self.ctx.spec, self.ctx.cfg, self.ctx.out_dir, split)
        return path, None

    def get_name(self) -> str:
        return self.method_name


class TailWeightedCopulaGenerator(BaseScenarioGenerator):
    priority = 0

    def fit(self, X_train, cond_train, X_val=None, cond_val=None) -> None:
        method_dir = self.ctx.out_dir / self.get_name()
        method_dir.mkdir(parents=True, exist_ok=True)
        tail_df = _compute_tail_scores(self.ctx.train_risk, self.ctx.cond_train, self.ctx.cfg, self.ctx.spec.out_name, method_dir)
        self.models, _, _ = _fit_tailweighted_group_copulas(
            self.ctx.x_train,
            self.ctx.cond_train,
            tail_df,
            self.ctx.cfg,
            method_dir,
            self.ctx.spec.out_name,
        )
        self.cop_cfg = _copula_cfg(self.ctx.cfg, int(self.ctx.x_train.shape[2]), method_dir / "copula_groups")
        self.rng = np.random.default_rng(int(self.ctx.cfg.seed))

    def generate(self, cond_target: pd.DataFrame, meta_target: pd.DataFrame, event_mask: np.ndarray | None, n_candidates: int = 20) -> np.ndarray:
        pools = []
        for _, row in cond_target.reset_index(drop=True).iterrows():
            month = int(row.get("month", 1))
            model, _, _ = _choose_copula_model(self.models, month)
            raw = model.sample_raw(int(n_candidates), self.rng)
            corrected = temporal_correction(raw, model, self.cop_cfg)
            cond_rows = pd.DataFrame([row] * int(n_candidates))
            cand = physical_projection_np(corrected, cond_rows=cond_rows, cfg=self.cop_cfg).astype(np.float32)
            pools.append(cand)
        return np.stack(pools, axis=0).astype(np.float32)

    def get_name(self) -> str:
        return TAIL_FIXED_METHOD


class StudentTCopulaGroupModel:
    """Student-t Copula 分组模型。

    相比 Gaussian Copula，Student-t Copula 能显式保留尾部相关性。
    这里采用经验边缘分布 + t 潜变量采样的轻量实现，用于极端场景候选池。
    """

    def __init__(self, group_name: str, x_group: np.ndarray, sample_weight: np.ndarray, cfg: CopulaConfig, df: int = 6):
        self.group_name = group_name
        self.n_samples = int(x_group.shape[0])
        self.n_channels = int(x_group.shape[1])
        self.seq_len = int(x_group.shape[2])
        self.dim = int(self.n_channels * self.seq_len)
        self.df = int(df)
        x_flat = x_group.reshape(self.n_samples, -1)
        self.marginals = [EmpiricalMarginal(x_flat[:, j], cfg.quantile_grid_size) for j in range(self.dim)]
        z = np.zeros_like(x_flat, dtype=np.float64)
        for j, marginal in enumerate(self.marginals):
            u = marginal.cdf(x_flat[:, j])
            u = np.clip(u, 1.0 / (self.n_samples + 2), 1.0 - 1.0 / (self.n_samples + 2))
            z[:, j] = sps.norm.ppf(u)
        z = np.nan_to_num(z, nan=0.0, posinf=4.75, neginf=-4.75)
        w = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        if w.shape[0] != self.n_samples:
            w = np.ones((self.n_samples,), dtype=np.float64)
        w = np.clip(w, 1e-6, None)
        w_sum = float(w.sum())
        self.mean = (z * w[:, None]).sum(axis=0) / max(w_sum, 1e-6)
        centered = z - self.mean[None, :]
        cov = (centered * w[:, None]).T @ centered / max(w_sum, 1e-6)
        cov = np.atleast_2d(cov)
        if cov.shape != (self.dim, self.dim):
            cov = np.eye(self.dim)
        shrink = float(np.clip(cfg.covariance_shrinkage, 0.0, 0.95))
        diag = np.diag(np.maximum(np.diag(cov), 1e-5))
        cov = (1.0 - shrink) * cov + shrink * diag
        cov = 0.5 * (cov + cov.T)
        vals, vecs = np.linalg.eigh(cov)
        vals = np.clip(vals, 1e-5, None)
        self.cov = (vecs * vals[None, :]) @ vecs.T
        self.cov_sqrt = vecs * np.sqrt(vals)[None, :]
        self.channel_min = np.quantile(x_group, 0.001, axis=(0, 2))
        self.channel_max = np.quantile(x_group, 0.999, axis=(0, 2))
        self.ramp_abs_q = self._estimate_ramp_quantile(x_group, cfg.ramp_clip_quantile)
        self.mean_profile = np.mean(x_group, axis=0)

    def _estimate_ramp_quantile(self, x: np.ndarray, q: float) -> np.ndarray:
        out = np.ones(self.n_channels, dtype=np.float64)
        for c in range(self.n_channels):
            r = np.abs(np.diff(x[:, c, :], axis=1)).reshape(-1)
            out[c] = float(np.quantile(r, np.clip(q, 0.5, 0.9999))) if r.size else np.inf
            out[c] = max(out[c], 1e-6)
        return out

    def sample_raw(self, n: int, rng: np.random.Generator) -> np.ndarray:
        normal = rng.standard_normal((int(n), self.dim)) @ self.cov_sqrt.T + self.mean[None, :]
        scale = np.sqrt(rng.chisquare(self.df, size=int(n)) / float(self.df))[:, None]
        t_latent = normal / np.maximum(scale, 1e-6)
        u = sps.t.cdf(t_latent, df=self.df)
        x_flat = np.zeros_like(u, dtype=np.float64)
        for j, marginal in enumerate(self.marginals):
            x_flat[:, j] = marginal.ppf(u[:, j])
        return x_flat.reshape(int(n), self.n_channels, self.seq_len)


class StudentTCopulaGenerator(BaseScenarioGenerator):
    priority = 1

    def fit(self, X_train, cond_train, X_val=None, cond_val=None) -> None:
        tail_df = _compute_tail_scores(self.ctx.train_risk, self.ctx.cond_train, self.ctx.cfg, self.ctx.spec.out_name, self.ctx.out_dir)
        weights = tail_df["sample_weight"].to_numpy(float)
        cop_cfg = _copula_cfg(self.ctx.cfg, int(self.ctx.x_train.shape[2]), self.ctx.out_dir / f"{self.get_name()}_copula_groups")
        self.cop_cfg = cop_cfg
        self.models = {}
        for key_type, key_name in [("global", "all")] + [("season", s) for s in ["summer", "autumn", "winter", "spring"]] + [("month", str(m)) for m in range(1, 13)]:
            if key_type == "global":
                idx = np.arange(len(self.ctx.x_train))
                use = True
            elif key_type == "season":
                idx = np.where(self.ctx.cond_train["au_season"].astype(str).to_numpy() == key_name)[0]
                use = len(idx) >= int(self.ctx.cfg.min_season_samples)
            else:
                idx = np.where(self.ctx.cond_train["month"].astype(int).to_numpy() == int(key_name))[0]
                use = len(idx) >= int(self.ctx.cfg.min_month_samples)
            if use:
                self.models[(key_type, key_name)] = StudentTCopulaGroupModel(
                    f"{key_type}:{key_name}",
                    self.ctx.x_train[idx],
                    weights[idx],
                    cop_cfg,
                    df=int(self.ctx.cfg.student_t_df),
                )
        self.rng = np.random.default_rng(int(self.ctx.cfg.seed))

    def generate(self, cond_target: pd.DataFrame, meta_target: pd.DataFrame, event_mask: np.ndarray | None, n_candidates: int = 20) -> np.ndarray:
        pools = []
        for _, row in cond_target.reset_index(drop=True).iterrows():
            month = int(row.get("month", 1))
            model, _, _ = _choose_copula_model(self.models, month)
            raw = model.sample_raw(int(n_candidates), self.rng)
            corrected = temporal_correction(raw, model, self.cop_cfg)
            cond_rows = pd.DataFrame([row] * int(n_candidates))
            cand = physical_projection_np(corrected, cond_rows=cond_rows, cfg=self.cop_cfg).astype(np.float32)
            pools.append(cand)
        return np.stack(pools, axis=0).astype(np.float32)

    def get_name(self) -> str:
        return STUDENT_T_METHOD


class RiskKNNBootstrapGenerator(BaseScenarioGenerator):
    priority = 2

    def fit(self, X_train, cond_train, X_val=None, cond_val=None) -> None:
        self.train_x = self.ctx.x_train.astype(np.float32)
        self.channel_std = self.train_x.std(axis=(0, 2), keepdims=True).astype(np.float32) + 1e-6

    def _neighbor_frame(self, month: int) -> pd.DataFrame:
        month_df = self.ctx.train_risk[self.ctx.train_risk["month"].astype(int) == int(month)]
        if len(month_df) >= max(8, self.ctx.cfg.risk_knn_neighbors):
            return month_df.reset_index()
        season = _add_month_season(pd.DataFrame({"month": [month]})).iloc[0]["au_season"]
        season_df = self.ctx.train_risk[self.ctx.train_risk["au_season"].astype(str) == str(season)]
        if len(season_df) >= max(8, self.ctx.cfg.risk_knn_neighbors):
            return season_df.reset_index()
        return self.ctx.train_risk.reset_index()

    def generate(self, cond_target: pd.DataFrame, meta_target: pd.DataFrame, event_mask: np.ndarray | None, n_candidates: int = 20) -> np.ndarray:
        rng = np.random.default_rng(int(self.ctx.cfg.seed))
        pools = []
        for sample_id, row in cond_target.reset_index(drop=True).iterrows():
            target_info, scales = _target_for_condition(sample_id, row, self.ctx.train_risk, self.ctx.cfg)
            month = int(row.get("month", 1))
            group = self._neighbor_frame(month)
            month_rad = 2.0 * np.pi * (month - 1.0) / 12.0
            extreme = float(pd.to_numeric(pd.Series([row.get("extreme_prob", 0.0)]), errors="coerce").iloc[0] or 0.0)
            d = (
                np.abs(group["cum_deficit"].to_numpy(float) - float(target_info["target_cum_deficit"])) / max(scales["cum_deficit"], 1e-6)
                + np.abs(group["core_cum_deficit"].to_numpy(float) - float(target_info["target_core_cum_deficit"])) / max(scales["core_cum_deficit"], 1e-6)
                + np.abs(group["netload_ramp_max"].to_numpy(float) - float(target_info["target_ramp"])) / max(scales["netload_ramp_max"], 1e-6)
                + np.abs(group["imbalance_duration"].to_numpy(float) - float(target_info["target_duration"])) / max(scales["imbalance_duration"], 1e-6)
                + 0.15 * np.abs(np.sin(month_rad) - np.sin(2.0 * np.pi * (group["month"].to_numpy(float) - 1.0) / 12.0))
                + 0.15 * np.abs(extreme - np.clip(pd.to_numeric(self.ctx.cond_train_raw.iloc[group["index"].to_numpy(int)].get("extreme_prob", 0.0), errors="coerce").fillna(0.0).to_numpy(float), 0.0, 1.0))
            )
            top_idx = group.iloc[np.argsort(d)[: max(3, min(int(self.ctx.cfg.risk_knn_neighbors), len(group)))]]["index"].to_numpy(int)
            sample_idx = rng.choice(top_idx, size=int(n_candidates), replace=True)
            x = self.train_x[sample_idx].copy()
            jitter = rng.standard_normal(size=x.shape).astype(np.float32) * self.ctx.cfg.bootstrap_jitter_scale * self.channel_std
            x = np.maximum(x + jitter, 0.0)
            x = physical_projection_np(x, cond_rows=pd.DataFrame([row] * int(n_candidates)), cfg=_copula_cfg(self.ctx.cfg, int(x.shape[2]), self.ctx.out_dir / "tmp_knn")).astype(np.float32)
            pools.append(x)
        return np.stack(pools, axis=0).astype(np.float32)

    def get_name(self) -> str:
        return RISK_KNN_METHOD


class QuantileRiskResamplerGenerator(BaseScenarioGenerator):
    priority = 3

    def fit(self, X_train, cond_train, X_val=None, cond_val=None) -> None:
        self.train_x = self.ctx.x_train.astype(np.float32)
        self.channel_std = self.train_x.std(axis=(0, 2), keepdims=True).astype(np.float32) + 1e-6

    def generate(self, cond_target: pd.DataFrame, meta_target: pd.DataFrame, event_mask: np.ndarray | None, n_candidates: int = 20) -> np.ndarray:
        rng = np.random.default_rng(int(self.ctx.cfg.seed))
        pools = []
        for sample_id, row in cond_target.reset_index(drop=True).iterrows():
            target_info, scales = _target_for_condition(sample_id, row, self.ctx.train_risk, self.ctx.cfg)
            month = int(row.get("month", 1))
            group_df = self.ctx.train_risk[self.ctx.train_risk["month"].astype(int) == month]
            if len(group_df) < 6:
                group_df = self.ctx.train_risk
            group_df = group_df.reset_index()
            d = (
                np.abs(group_df["cum_deficit"].to_numpy(float) - float(target_info["target_cum_deficit"])) / max(scales["cum_deficit"], 1e-6)
                + np.abs(group_df["core_cum_deficit"].to_numpy(float) - float(target_info["target_core_cum_deficit"])) / max(scales["core_cum_deficit"], 1e-6)
                + np.abs(group_df["netload_ramp_max"].to_numpy(float) - float(target_info["target_ramp"])) / max(scales["netload_ramp_max"], 1e-6)
                + np.abs(group_df["imbalance_duration"].to_numpy(float) - float(target_info["target_duration"])) / max(scales["imbalance_duration"], 1e-6)
            )
            base_idx = group_df.iloc[np.argsort(d)[: max(1, min(int(n_candidates), len(group_df)))]]["index"].to_numpy(int)
            sample_idx = rng.choice(base_idx, size=int(n_candidates), replace=True)
            x = self.train_x[sample_idx].copy()
            jitter = rng.standard_normal(size=x.shape).astype(np.float32) * self.ctx.cfg.quantile_jitter_scale * self.channel_std
            x = np.maximum(x + jitter, 0.0)
            x = physical_projection_np(x, cond_rows=pd.DataFrame([row] * int(n_candidates)), cfg=_copula_cfg(self.ctx.cfg, int(x.shape[2]), self.ctx.out_dir / "tmp_quantile")).astype(np.float32)
            pools.append(x)
        return np.stack(pools, axis=0).astype(np.float32)

    def get_name(self) -> str:
        return QUANTILE_RESAMPLER_METHOD


class TCNBlock(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding, dilation=dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.conv1(x)[..., : x.shape[-1]]
        y = F.silu(y)
        y = self.dropout(y)
        y = self.conv2(y)[..., : x.shape[-1]]
        y = self.dropout(F.silu(y))
        return residual + y


class DirectTCNGenerator(nn.Module):
    """轻量 TCN 条件生成器，用于小样本时序建模。

    该模块通过扩张卷积捕捉 36h 窗口内的局部和跨步依赖，
    目标是比 Transformer 更轻量、更稳地刻画净负荷爬坡过程。
    """

    def __init__(self, cond_dim: int, hidden_dim: int = 64, noise_dim: int = 16, num_layers: int = 3, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.noise_dim = noise_dim
        self.cond_dim = cond_dim
        self.in_proj = nn.Conv1d(3 + noise_dim + cond_dim, hidden_dim, kernel_size=1)
        dilations = [2**i for i in range(num_layers)]
        self.blocks = nn.ModuleList([TCNBlock(hidden_dim, kernel_size, d, dropout) for d in dilations])
        self.out = nn.Conv1d(hidden_dim, 3, kernel_size=1)

    def forward(self, base: torch.Tensor, noise: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        cond_seq = cond[:, :, None].expand(-1, -1, base.shape[-1])
        x = torch.cat([base, noise, cond_seq], dim=1)
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.out(h)


class ConditionalTCNGenerator(BaseScenarioGenerator):
    priority = 4

    def fit(self, X_train, cond_train, X_val=None, cond_val=None) -> None:
        self.device = torch.device(self.ctx.cfg.device)
        self.scaler = ChannelScaler().fit(self.ctx.x_train)
        x_norm = self.scaler.transform(self.ctx.x_train)
        months = self.ctx.cond_train["month"].astype(int).to_numpy(np.int64)
        masks = self.ctx.mask_train if self.ctx.mask_train is not None else np.ones((len(x_norm), x_norm.shape[-1]), dtype=np.float32)
        ds = TensorDataset(
            torch.from_numpy(x_norm.astype(np.float32)),
            torch.from_numpy(self.ctx.cond_train_features.astype(np.float32)),
            torch.from_numpy(months),
            torch.from_numpy(masks.astype(np.float32)),
        )
        loader = DataLoader(ds, batch_size=int(self.ctx.cfg.batch_size), shuffle=True)
        self.model = DirectTCNGenerator(
            cond_dim=self.ctx.cond_train_features.shape[1],
            hidden_dim=int(self.ctx.cfg.tcn_hidden_dim),
            noise_dim=int(self.ctx.cfg.tcn_noise_dim),
            num_layers=int(self.ctx.cfg.tcn_num_layers),
            kernel_size=int(self.ctx.cfg.tcn_kernel_size),
            dropout=float(self.ctx.cfg.tcn_dropout),
        ).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=float(self.ctx.cfg.model_lr))
        mean_t = torch.from_numpy(self.scaler.mean).to(self.device)
        std_t = torch.from_numpy(self.scaler.std).to(self.device)
        tau_lookup = {int(k): float(v) for k, v in self.ctx.tau_by_month.items()}
        for _ in range(int(self.ctx.cfg.tcn_epochs)):
            for xb, cb, mb, eb in loader:
                xb, cb, mb, eb = xb.to(self.device), cb.to(self.device), mb.to(self.device), eb.to(self.device)
                noisy = xb + 0.10 * torch.randn_like(xb)
                noise = torch.randn((xb.shape[0], int(self.ctx.cfg.tcn_noise_dim), xb.shape[-1]), device=self.device)
                pred = self.model(noisy, noise, cb)
                recon_loss = F.mse_loss(pred, xb)
                pred_denorm = pred * std_t + mean_t
                true_denorm = xb * std_t + mean_t
                tau = torch.tensor([tau_lookup.get(int(m.item()), np.median(list(tau_lookup.values()))) for m in mb], dtype=pred.dtype, device=self.device)
                cum_p, ramp_p, dur_p = soft_risk_metrics_torch(pred_denorm, tau, delta_t_hours=float(self.ctx.cfg.delta_t_hours), duration_temp=float(self.ctx.cfg.duration_temp), ramp_metric_mode="window_3h", ramp_window_hours=3.0)
                cum_t, ramp_t, dur_t = soft_risk_metrics_torch(true_denorm.detach(), tau, delta_t_hours=float(self.ctx.cfg.delta_t_hours), duration_temp=float(self.ctx.cfg.duration_temp), ramp_metric_mode="window_3h", ramp_window_hours=3.0)
                core_cum_p, _, _ = soft_core_risk_metrics_torch(pred_denorm, tau, eb, delta_t_hours=float(self.ctx.cfg.delta_t_hours), duration_temp=float(self.ctx.cfg.duration_temp))
                core_cum_t, _, _ = soft_core_risk_metrics_torch(true_denorm.detach(), tau, eb, delta_t_hours=float(self.ctx.cfg.delta_t_hours), duration_temp=float(self.ctx.cfg.duration_temp))
                loss = (
                    recon_loss
                    + float(self.ctx.cfg.lambda_cum) * F.smooth_l1_loss(cum_p, cum_t)
                    + float(self.ctx.cfg.lambda_core) * F.smooth_l1_loss(core_cum_p, core_cum_t)
                    + float(self.ctx.cfg.lambda_ramp) * F.smooth_l1_loss(ramp_p, ramp_t)
                    + float(self.ctx.cfg.lambda_dur) * F.smooth_l1_loss(dur_p, dur_t)
                    + float(self.ctx.cfg.lambda_phy) * torch.relu(-pred_denorm).mean()
                )
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.model.eval()
        self.train_x_norm = x_norm.astype(np.float32)

    @torch.no_grad()
    def generate(self, cond_target: pd.DataFrame, meta_target: pd.DataFrame, event_mask: np.ndarray | None, n_candidates: int = 20) -> np.ndarray:
        cond_feat = self.ctx.cond_encoder.transform(_condition_frame(cond_target, meta_target))
        out = []
        for i in range(len(cond_feat)):
            base_idx = np.random.choice(len(self.train_x_norm), size=int(n_candidates), replace=True)
            base = torch.from_numpy(self.train_x_norm[base_idx]).to(self.device)
            c = torch.from_numpy(np.repeat(cond_feat[i : i + 1], int(n_candidates), axis=0).astype(np.float32)).to(self.device)
            noise = torch.randn((int(n_candidates), int(self.ctx.cfg.tcn_noise_dim), self.train_x_norm.shape[2]), device=self.device)
            pred = self.model(base, noise, c).cpu().numpy()
            out.append(self.scaler.inverse(pred))
        return np.stack(out, axis=0).astype(np.float32)

    def get_name(self) -> str:
        return TCN_METHOD


def _condition_frame(cond: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    out = cond.reset_index(drop=True).copy()
    meta = meta.reset_index(drop=True)
    for col in meta.columns:
        if col not in out.columns:
            out[col] = meta[col].to_numpy()
    if "month" not in out.columns:
        out["month"] = 1
    return out


def _generated_path(out_dir: Path, method: str, split: str) -> Path:
    safe = method.replace("/", "_")
    return out_dir / (f"generated_samples_{safe}.npy" if split == "test" else f"generated_val_{safe}.npy")


def _selection_log_path(out_dir: Path, method: str, split: str) -> Path:
    safe = method.replace("/", "_")
    return out_dir / f"candidate_selection_log_{safe}_{split}.csv"


def _select_candidates_from_pool(
    candidates: np.ndarray,
    cond: pd.DataFrame,
    mask: np.ndarray | None,
    train_risk: pd.DataFrame,
    tau_by_month: dict[int, float],
    cfg: ExpandedModelPoolConfig,
    base_method_name: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    metrics, targets, base = _candidate_metrics_for_pool(candidates, cond, mask, train_risk, tau_by_month, cfg)
    for row in base:
        row["copula_group_used"] = base_method_name
    selected, log = _select_candidates(candidates, metrics, targets, base, train_risk, cond, _parse_weights(cfg.fixed_weights), cfg)
    return selected, log


def _build_context(spec: DatasetSpec, cfg: ExpandedModelPoolConfig, out_dir: Path) -> DatasetContext:
    x_train, cond_train_raw, meta_train, mask_train = _load_split(spec.data_dir, "train")
    cond_train = _add_month_season(cond_train_raw, meta_train)
    tau_by_month, _ = _compute_monthly_tau(x_train, cond_train, cfg, spec.out_name, out_dir)
    train_risk = _build_train_risk_table(x_train, cond_train, mask_train, tau_by_month, cfg)
    cond_encoder = ConditionEncoder(_condition_frame(cond_train_raw, meta_train))
    cond_train_features = cond_encoder.transform(_condition_frame(cond_train_raw, meta_train))
    return DatasetContext(
        spec=spec,
        cfg=cfg,
        out_dir=out_dir,
        x_train=x_train,
        cond_train_raw=cond_train_raw,
        meta_train=meta_train,
        mask_train=mask_train,
        cond_train=cond_train,
        tau_by_month=tau_by_month,
        train_risk=train_risk,
        cond_encoder=cond_encoder,
        cond_train_features=cond_train_features,
    )


def _tailweighted_split_runner(spec: DatasetSpec, cfg: ExpandedModelPoolConfig, out_dir: Path, split: str) -> Path:
    from run_model_pool_experiments import _generate_tailweighted_split

    return _generate_tailweighted_split(spec, cfg, out_dir, split, TAIL_FIXED_METHOD)


def _vae_split_runner(spec: DatasetSpec, cfg: ExpandedModelPoolConfig, out_dir: Path, split: str) -> Path:
    return _run_vae_aug(spec, cfg, out_dir, split)


def _gan_split_runner(spec: DatasetSpec, cfg: ExpandedModelPoolConfig, out_dir: Path, split: str) -> Path:
    gan_val, gan_test = _load_existing_gan_paths(spec, out_dir, cfg)
    if gan_val is None or gan_test is None:
        gan_cfg = GanAugmentedTailWeightedConfig(
            out_dir=BASE_DIR / "results" / "gan_augmented_tailweighted_copula_valprotected_formal50",
            augmented_data_root=cfg.augmented_data_root / "gan_augmented_tailweighted_pool",
            seed=cfg.seed,
            k_candidates=cfg.k_candidates,
            min_month_samples=cfg.min_month_samples,
            min_season_samples=cfg.min_season_samples,
            tau_quantile=cfg.tau_quantile,
            alpha_tail=cfg.alpha_tail,
            tail_w_cum=cfg.tail_w_cum,
            tail_w_core=cfg.tail_w_core,
            tail_w_ramp=cfg.tail_w_ramp,
            tail_w_duration=cfg.tail_w_duration,
            fixed_weights=cfg.fixed_weights,
            delta_t_hours=cfg.delta_t_hours,
        )
        gan_val, gan_test = ensure_gan_augmented_val_test_paths(spec, gan_cfg, gan_cfg.out_dir / spec.out_name)
    source = gan_test if split == "test" else gan_val
    target = _generated_path(out_dir, GAN_METHOD, split)
    target.parent.mkdir(parents=True, exist_ok=True)
    if source != target:
        import shutil

        shutil.copy2(source, target)
    return target


def _fit_legacy_direct_generator(ctx: DatasetContext, method_name: str):
    if method_name == TRANSFORMER_METHOD:
        model, scaler = _train_transformer(ctx.x_train, ctx.cond_train_features, ctx.cfg)
        return model, scaler
    if method_name == FLOW_METHOD:
        model, scaler = _train_flow(ctx.x_train, ctx.cond_train_features, ctx.cfg)
        return model, scaler
    raise ValueError(method_name)


class LegacyDirectCandidateGenerator(BaseScenarioGenerator):
    def __init__(self, ctx: DatasetContext, method_name: str):
        super().__init__(ctx)
        self.method_name = method_name
        self.priority = 3 if method_name == TRANSFORMER_METHOD else 4

    def fit(self, X_train, cond_train, X_val=None, cond_val=None) -> None:
        self.model, self.scaler = _fit_legacy_direct_generator(self.ctx, self.method_name)

    def generate(self, cond_target: pd.DataFrame, meta_target: pd.DataFrame, event_mask: np.ndarray | None, n_candidates: int = 20) -> np.ndarray:
        cond_feat = self.ctx.cond_encoder.transform(_condition_frame(cond_target, meta_target))
        if self.method_name == TRANSFORMER_METHOD:
            return _sample_transformer_candidates(self.model, self.scaler, self.ctx.x_train, cond_feat, int(n_candidates), self.ctx.cfg)
        return _sample_flow_candidates(self.model, self.scaler, cond_feat, int(n_candidates), self.ctx.cfg)

    def get_name(self) -> str:
        return self.method_name


GENERATOR_REGISTRY: dict[str, Any] = {
    TAIL_FIXED_METHOD: lambda ctx: TailWeightedCopulaGenerator(ctx),
    STUDENT_T_METHOD: lambda ctx: StudentTCopulaGenerator(ctx),
    RISK_KNN_METHOD: lambda ctx: RiskKNNBootstrapGenerator(ctx),
    QUANTILE_RESAMPLER_METHOD: lambda ctx: QuantileRiskResamplerGenerator(ctx),
    TRANSFORMER_METHOD: lambda ctx: LegacyDirectCandidateGenerator(ctx, TRANSFORMER_METHOD),
    FLOW_METHOD: lambda ctx: LegacyDirectCandidateGenerator(ctx, FLOW_METHOD),
    TCN_METHOD: lambda ctx: ConditionalTCNGenerator(ctx),
    VAE_METHOD: lambda ctx: LegacySelectedGenerator(ctx, VAE_METHOD, _vae_split_runner),
    GAN_METHOD: lambda ctx: LegacySelectedGenerator(ctx, GAN_METHOD, _gan_split_runner),
}


def _instantiate_generators(ctx: DatasetContext, method_names: list[str]) -> list[BaseScenarioGenerator]:
    generators = []
    for name in method_names:
        factory = GENERATOR_REGISTRY.get(name)
        if factory is None:
            raise KeyError(f"Unknown generator name: {name}")
        generators.append(factory(ctx))
    return generators


def _write_failed_methods(out_dir: Path, failed_rows: list[dict]) -> None:
    lines = ["# Failed Methods Summary", ""]
    if not failed_rows:
        lines.append("- no failed methods")
    else:
        for row in failed_rows:
            lines.extend(
                [
                    f"## {row['dataset']} - {row['method']}",
                    "",
                    f"- stage: `{row['stage']}`",
                    f"- reason: `{row['error']}`",
                    "",
                    "```text",
                    str(row["traceback"]),
                    "```",
                    "",
                ]
            )
    (out_dir / "failed_methods_summary.md").write_text("\n".join(lines), encoding="utf-8-sig")


def _run_stacked_selection(
    ctx: DatasetContext,
    generators: dict[str, BaseScenarioGenerator],
    split: str,
    candidate_methods: list[str],
) -> tuple[Path, pd.DataFrame]:
    _, cond_raw, meta, mask = _load_split(ctx.spec.data_dir, split)
    cond = _add_month_season(cond_raw, meta)
    candidate_blocks = []
    source_names = []
    for method in candidate_methods:
        gen = generators.get(method)
        if gen is None or not gen.candidate_ready:
            continue
        try:
            cand = gen.generate(cond, meta, mask, n_candidates=int(ctx.cfg.k_candidates))
        except Exception:
            continue
        candidate_blocks.append(cand.astype(np.float32))
        source_names.extend([method] * cand.shape[1])
    if not candidate_blocks:
        raise RuntimeError("No candidate-ready generators were available for stacked selection.")
    all_candidates = np.concatenate(candidate_blocks, axis=1).astype(np.float32)
    metrics, _, _ = _candidate_metrics_for_pool(all_candidates, cond, mask, ctx.train_risk, ctx.tau_by_month, ctx.cfg)
    selected = []
    logs = []
    weights = _parse_weights(ctx.cfg.fixed_weights)
    for i, row in cond.reset_index(drop=True).iterrows():
        target_info, scales = _target_for_condition(i, row, ctx.train_risk, ctx.cfg)
        sub = metrics[metrics["sample_id"].astype(int) == int(i)].reset_index(drop=True)
        scores = _score_candidates(sub, target_info, scales, weights)
        j = int(np.nanargmin(scores)) if len(scores) else 0
        selected.append(all_candidates[i, j])
        logs.append(
            {
                "dataset": ctx.spec.out_name,
                "sample_id": int(i),
                "selected_base_method": str(source_names[j]),
                "selected_score": float(scores[j]),
                "target_cum": float(target_info["target_cum_deficit"]),
                "target_core": float(target_info["target_core_cum_deficit"]),
                "target_ramp": float(target_info["target_ramp"]),
                "target_duration": float(target_info["target_duration"]),
                "selected_cum": float(sub.loc[j, "cum_deficit"]),
                "selected_core": float(sub.loc[j, "core_cum_deficit"]),
                "selected_ramp": float(sub.loc[j, "netload_ramp_max"]),
                "selected_duration": float(sub.loc[j, "imbalance_duration"]),
            }
        )
    path = _generated_path(ctx.out_dir, STACKED_METHOD, split)
    np.save(path, np.asarray(selected, dtype=np.float32))
    log_df = pd.DataFrame(logs)
    log_df.to_csv(ctx.out_dir / f"stacked_candidate_selection_log_{split}.csv", index=False, encoding="utf-8-sig")
    return path, log_df


def run_dataset(spec: DatasetSpec, cfg: ExpandedModelPoolConfig, method_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    out_dir = cfg.out_dir / spec.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    _set_seed(cfg.seed)
    ctx = _build_context(spec, cfg, out_dir)
    generators = {g.get_name(): g for g in _instantiate_generators(ctx, method_names)}
    failed_rows: list[dict] = []
    val_rows = []
    test_rows = []
    val_paths: dict[str, Path] = {}
    test_paths: dict[str, Path] = {}
    for name, gen in generators.items():
        try:
            gen.fit(ctx.x_train, ctx.cond_train_raw)
            val_path, _ = gen.generate_selected("val", K_candidates=int(cfg.k_candidates))
            test_path, _ = gen.generate_selected("test", K_candidates=int(cfg.k_candidates))
            val_paths[name] = val_path
            test_paths[name] = test_path
            val_rows.append(gen.evaluate(val_path, split="val"))
            test_rows.append(gen.evaluate(test_path, split="test"))
        except Exception as exc:
            failed_rows.append(
                {
                    "dataset": spec.out_name,
                    "method": name,
                    "stage": "fit/generate/evaluate",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
    if cfg.stacked_candidate_methods:
        try:
            stacked_val, _ = _run_stacked_selection(ctx, generators, "val", list(cfg.stacked_candidate_methods))
            stacked_test, _ = _run_stacked_selection(ctx, generators, "test", list(cfg.stacked_candidate_methods))
            val_paths[STACKED_METHOD] = stacked_val
            test_paths[STACKED_METHOD] = stacked_test
            val_rows.append(_evaluate_test(STACKED_METHOD, stacked_val, spec, out_dir, split="val"))
            test_rows.append(_evaluate_test(STACKED_METHOD, stacked_test, spec, out_dir, split="test"))
        except Exception as exc:
            failed_rows.append(
                {
                    "dataset": spec.out_name,
                    "method": STACKED_METHOD,
                    "stage": "stacked-selection",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
    if not val_rows or not test_rows:
        raise RuntimeError(f"No successful generators for dataset {spec.out_name}.")
    val_df = add_risk_score(pd.DataFrame(val_rows))
    priority_map = {name: getattr(gen, "priority", 9) for name, gen in generators.items()}
    priority_map[STACKED_METHOD] = 8
    val_df["priority"] = val_df["method"].map(priority_map).fillna(9)
    baseline_method = TAIL_FIXED_METHOD if TAIL_FIXED_METHOD in val_df["method"].astype(str).tolist() else str(
        val_df.sort_values(["risk_score", "priority"], ascending=[True, True]).iloc[0]["method"]
    )
    selections = []
    for margin in cfg.run_margins:
        if baseline_method == TAIL_FIXED_METHOD:
            selected_method, best_candidate_method, tail_score, best_candidate_score, reason = _select_with_margin(val_df, float(margin))
        else:
            best_row = val_df.sort_values(["risk_score", "priority"], ascending=[True, True]).iloc[0]
            selected_method = str(best_row["method"])
            best_candidate_method = selected_method
            tail_score = float(best_row["risk_score"])
            best_candidate_score = float(best_row["risk_score"])
            reason = f"TailWeighted unavailable; fallback baseline is current best available method {selected_method}"
        if selected_method not in test_paths:
            selected_method = baseline_method
            reason = f"{reason}; selected method missing in successful pool, fallback to {baseline_method}"
        selections.append(
            {
                "margin": float(margin),
                "dataset": spec.out_name,
                "tailweighted_val_score": tail_score,
                "best_candidate_method": best_candidate_method,
                "best_candidate_val_score": best_candidate_score,
                "selected_method": selected_method,
                "reason": reason,
            }
        )
        val_df[f"selected_margin_{float(margin):.2f}"] = val_df["method"].astype(str).eq(selected_method)
    val_df = val_df.rename(
        columns={
            "q99_cum_deficit_error": "val_q99",
            "core_q99_cum_deficit_error": "val_core_q99",
            "netload_ramp_max_mae": "val_ramp",
            "imbalance_duration_mae": "val_duration",
            "risk_score": "val_risk_score",
        }
    )
    val_df.insert(0, "dataset", spec.out_name)
    val_df.to_csv(out_dir / "val_model_selection_summary.csv", index=False, encoding="utf-8-sig")

    rows = list(test_rows)
    for selection in selections:
        margin = float(selection["margin"])
        selected_method = str(selection["selected_method"])
        method_name = f"Expanded_ValSelected_Model_Ensemble_Margin_{margin:.2f}"
        selected_path = test_paths[selected_method]
        target_path = _generated_path(out_dir, method_name, "test")
        if selected_path != target_path:
            import shutil

            shutil.copy2(selected_path, target_path)
        rows.append(_evaluate_test(method_name, target_path, spec, out_dir, split="test"))
        if abs(margin - 0.05) < 1e-12:
            legacy_path = _generated_path(out_dir, EXPANDED_ENSEMBLE_METHOD, "test")
            if selected_path != legacy_path:
                import shutil

                shutil.copy2(selected_path, legacy_path)
            rows.append(_evaluate_test(EXPANDED_ENSEMBLE_METHOD, legacy_path, spec, out_dir, split="test"))

    compare_df = add_risk_score(pd.DataFrame(rows))
    compare_df.to_csv(out_dir / "compare_all_methods.csv", index=False, encoding="utf-8-sig")
    selection_df = pd.DataFrame(selections)
    selection_df.to_csv(out_dir / "expanded_val_selection_summary.csv", index=False, encoding="utf-8-sig")
    risk_main, aux = write_risk_tables(compare_df, out_dir)
    lines = [
        f"# {spec.name} - Expanded Model Pool",
        "",
        "## Validation Selection",
        "",
        val_df.to_markdown(index=False),
        "",
        "## Expanded Margin Selection",
        "",
        selection_df.to_markdown(index=False),
        "",
        "## Main Risk Table",
        "",
        risk_main.to_markdown(index=False),
        "",
        "## Auxiliary Realism",
        "",
        aux.to_markdown(index=False),
    ]
    (out_dir / "method_report.md").write_text("\n".join(lines), encoding="utf-8-sig")
    return risk_main, aux, selection_df, failed_rows


def _write_global(root: Path, risk_tables: dict[str, pd.DataFrame], aux_tables: dict[str, pd.DataFrame], selection_tables: dict[str, pd.DataFrame], failed_rows: list[dict]) -> None:
    risk_rows = []
    for ds, table in risk_tables.items():
        df = table.copy()
        df.insert(0, "dataset", ds)
        risk_rows.append(df)
    risk_summary = pd.concat(risk_rows, ignore_index=True)
    risk_summary.to_csv(root / "all_datasets_risk_summary.csv", index=False, encoding="utf-8-sig")
    aux_rows = []
    for ds, table in aux_tables.items():
        df = table.copy()
        df.insert(0, "dataset", ds)
        aux_rows.append(df)
    pd.concat(aux_rows, ignore_index=True).to_csv(root / "all_datasets_auxiliary_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(selection_tables.values(), ignore_index=True).to_csv(root / "expanded_val_selection_summary.csv", index=False, encoding="utf-8-sig")

    methods = risk_summary["method"].astype(str).unique().tolist()
    rows = []
    datasets = list(risk_tables.keys())
    for method in methods:
        row = {"method": method}
        ranks, scores = [], []
        for ds in datasets:
            sub = risk_summary[(risk_summary["dataset"] == ds) & (risk_summary["method"] == method)]
            if sub.empty:
                row[f"{ds}_risk_rank"] = np.nan
                row[f"{ds}_risk_score"] = np.nan
                continue
            rank = float(sub["risk_rank"].iloc[0])
            score = float(sub["risk_score"].iloc[0])
            row[f"{ds}_risk_rank"] = rank
            row[f"{ds}_risk_score"] = score
            ranks.append(rank)
            scores.append(score)
        row["mean_risk_rank"] = float(np.mean(ranks)) if ranks else np.nan
        row["mean_risk_score"] = float(np.mean(scores)) if scores else np.nan
        row["wins_count"] = int(sum(1 for rank in ranks if rank == 1.0))
        row["top3_count"] = int(sum(1 for rank in ranks if rank <= 3.0))
        rows.append(row)
    rank_df = pd.DataFrame(rows).sort_values(["mean_risk_rank", "mean_risk_score"], na_position="last")
    rank_df.to_csv(root / "all_datasets_rank_summary.csv", index=False, encoding="utf-8-sig")
    _write_failed_methods(root, failed_rows)

    expanded_rows = rank_df[rank_df["method"].astype(str).str.startswith("Expanded_ValSelected_Model_Ensemble_Margin_")].copy()
    stacked_row = rank_df[rank_df["method"] == STACKED_METHOD]
    tail_row = rank_df[rank_df["method"] == TAIL_FIXED_METHOD]
    best_row = rank_df.iloc[0] if len(rank_df) else None
    lines = [
        "# Expanded Model Pool Report",
        "",
        "## Rank Summary",
        "",
        rank_df.to_markdown(index=False),
        "",
        "## Expanded Val Selection",
        "",
        pd.concat(selection_tables.values(), ignore_index=True).to_markdown(index=False),
        "",
        "## Experiment Conclusion",
    ]
    lines.append(f"- 是否有新方法超过当前 ValSelected_Margin_0.05：{'是' if best_row is not None and str(best_row['method']) != 'ValSelected_Model_Ensemble_Margin_0.05' else '否或未比较到旧目录'}")
    lines.append(f"- 最优方法：{best_row['method'] if best_row is not None else 'NA'}")
    lines.append(f"- 是否推荐替换当前主方法：{'是' if best_row is not None and 'Expanded_ValSelected_Model_Ensemble_Margin_0.05' in str(best_row['method']) or str(best_row['method']) == STACKED_METHOD else '暂不'}")
    lines.extend(["", "## Failed Methods", ""])
    if failed_rows:
        fail_df = pd.DataFrame(failed_rows)[["dataset", "method", "stage", "error"]]
        lines.append(fail_df.to_markdown(index=False))
    else:
        lines.append("- no failed methods")
    (root / "final_expanded_model_pool_report.md").write_text("\n".join(lines), encoding="utf-8-sig")


def run_all(cfg: ExpandedModelPoolConfig, datasets: list[str] | None, method_names: list[str]) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.augmented_data_root.mkdir(parents=True, exist_ok=True)
    specs = [s for s in _dataset_specs() if s.data_dir.exists()]
    if datasets:
        specs = [s for s in specs if s.out_name in set(datasets)]
    risk_tables: dict[str, pd.DataFrame] = {}
    aux_tables: dict[str, pd.DataFrame] = {}
    selection_tables: dict[str, pd.DataFrame] = {}
    failed_rows: list[dict] = []
    for spec in specs:
        print(f"\n=== Dataset: {spec.out_name} ===")
        try:
            risk, aux, sel, failed = run_dataset(spec, cfg, method_names)
            risk_tables[spec.out_name] = risk
            aux_tables[spec.out_name] = aux
            selection_tables[spec.out_name] = sel
            failed_rows.extend(failed)
        except Exception as exc:
            failed_rows.append(
                {
                    "dataset": spec.out_name,
                    "method": "ALL_DATASET_RUN",
                    "stage": "dataset-run",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
    if risk_tables:
        _write_global(cfg.out_dir, risk_tables, aux_tables, selection_tables, failed_rows)
    else:
        _write_failed_methods(cfg.out_dir, failed_rows)


def parse_args() -> tuple[ExpandedModelPoolConfig, list[str] | None, list[str]]:
    parser = argparse.ArgumentParser(description="Run expanded scenario-generator model pool experiments.")
    parser.add_argument("--out-dir", type=Path, default=BASE_DIR / "results" / "expanded_model_pool_formal50")
    parser.add_argument("--datasets", type=str, default="")
    parser.add_argument("--methods", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--tcn-epochs", type=int, default=5)
    parser.add_argument("--k-candidates", type=int, default=10)
    parser.add_argument("--candidate-count", type=int, default=20)
    parser.add_argument("--selection-margins", type=str, default="0.00,0.05,0.10")
    parser.add_argument("--alpha-tail", type=float, default=1.0)
    parser.add_argument("--fixed-weights", type=str, default="0.35,0.25,0.20,0.20")
    args = parser.parse_args()
    cfg = ExpandedModelPoolConfig(
        out_dir=args.out_dir,
        seed=args.seed,
        model_epochs=args.epochs,
        vae_epochs=args.epochs,
        transformer_epochs=args.epochs,
        flow_epochs=args.epochs,
        tcn_epochs=args.tcn_epochs,
        k_candidates=args.k_candidates,
        direct_k_candidates=args.k_candidates,
        candidate_count=args.candidate_count,
        alpha_tail=args.alpha_tail,
        fixed_weights=args.fixed_weights,
        selection_margins=tuple(float(item.strip()) for item in args.selection_margins.split(",") if item.strip()),
        run_margins=tuple(float(item.strip()) for item in args.selection_margins.split(",") if item.strip()),
    )
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()] or None
    default_methods = [
        TAIL_FIXED_METHOD,
        STUDENT_T_METHOD,
        RISK_KNN_METHOD,
        QUANTILE_RESAMPLER_METHOD,
        TRANSFORMER_METHOD,
        FLOW_METHOD,
        TCN_METHOD,
        VAE_METHOD,
        GAN_METHOD,
    ]
    methods = [item.strip() for item in args.methods.split(",") if item.strip()] or default_methods
    return cfg, datasets, methods


if __name__ == "__main__":
    cfg, datasets, methods = parse_args()
    run_all(cfg, datasets, methods)
