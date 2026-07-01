#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v90_pollution_integrated_comparison.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
整合版本：9 个基线模型 + 主模型（CatBoost + Optuna）
  - 主模型来源：v89_pollution_main_explain.py
  - 基线模型来源：v87_pollution_baseline_comparison_v3.py
  - 已移除：全部 SHAP 可解释性层
  - 已移除：混淆矩阵输出
  - 保留：所有评价指标（逐窗口明细 + 跨窗口均值）
  - 输出：Excel（逐窗口逐模型明细 + 跨窗口均值汇总 + 配置说明）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# ════════════════════════════════════════════════════════════════════════
# 依赖导入
# ════════════════════════════════════════════════════════════════════════
import os
import warnings
import time
import logging
import platform
from datetime import datetime
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    roc_auc_score, precision_recall_fscore_support, average_precision_score,
)

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── 可选依赖 ────────────────────────────────────────────────────────────
HAS_LGB = False
try:
    import lightgbm as lgb
    HAS_LGB = True
    logging.getLogger("lightgbm").setLevel(logging.CRITICAL + 1)
except ImportError:
    pass

HAS_XGB = False
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    pass

HAS_CB = False
try:
    from catboost import CatBoostClassifier
    HAS_CB = True
except ImportError:
    pass

HAS_IMBLEARN = False
try:
    from imblearn.ensemble import (
        BalancedRandomForestClassifier,
        EasyEnsembleClassifier,
    )
    from imblearn.over_sampling import SMOTE
    HAS_IMBLEARN = True
except ImportError:
    pass

HAS_OPTUNA = False
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    pass

print(f"[ENV] LGB={HAS_LGB}  XGB={HAS_XGB}  CB={HAS_CB}"
      f"  IMBLEARN={HAS_IMBLEARN}  OPTUNA={HAS_OPTUNA}"
      f"  OS={platform.system()}\n")


# ════════════════════════════════════════════════════════════════════════
# 全局配置
# ════════════════════════════════════════════════════════════════════════
DATA_PATH   = r"C:\Users\ADMIM\Desktop\论文实验1.xlsx"
SHEET_NAME  = "Sheet1"          # 按需修改
OUTPUT_PATH = "v90_integrated_comparison_results.xlsx"

YEAR_COL     = "year"
ID_COL       = "id"
INDUSTRY_COL = "industry"
TARGET_COL   = "Wind E"

RANDOM_STATE = 42

# ── CPU 线程配置（平台自适应） ──────────────────────────────────────
if platform.system() == "Windows":
    N_JOBS          = 1
    N_JOBS_CATBOOST = 8
else:
    N_JOBS          = 16
    N_JOBS_CATBOOST = 16

# ── 预处理参数 ─────────────────────────────────────────────────────────
CAP_LOWER_PCT      = 0.01
CAP_UPPER_PCT      = 0.99
FEAT_IMP_THRESHOLD = 0.001
FEAT_MIN_KEEP      = 50
FEAT_TOPK          = 100
CORR_THRESHOLD     = 0.85
PCA_VARIANCE       = 0.95

# ── Optuna 参数 ────────────────────────────────────────────────────────
OPTUNA_N_TRIALS_MAX = 40
OPTUNA_N_TRIALS_MIN = 15
OPTUNA_TRIALS_PER_N = 200
OPTUNA_VAL_RATIO    = 0.5
CW_MODE             = "sqrt"
OPTUNA_L2_MAX       = 15.0

# ── 特征与污染物配置 ────────────────────────────────────────────────────
POLL_ANOMALY_GASES      = ["CH4", "NO2", "CO", "SO2"]
RAW_POLLUTION_COLS_DROP = {"mean_NO2", "mean_SO2", "mean_CH4", "mean_CO"}
POLL_TREND_COLS = [
    "mean_SO2", "mean_NO2", "mean_CH4", "mean_CO",
    "SO2_chang", "NO2_chang", "CH4_chang", "CO_chang",
]
CARBON_NEUTRAL_YEAR = 2021

CORE_NUM_FEATURES = [
    "owner", "independent_rate", "diet", "recognize",
    "prize", "punish_num", "punish_money", "patent",
    "CH4_CV", "NO2_CV", "CO_CV", "SO2_CV",
    "mean_NO2", "mean_SO2", "mean_CH4", "mean_CO",
    "NO2_chang", "SO2_chang", "CH4_chang", "CO_chang",
    "Top1", "Lnage", "Lev", "ROA", "Size", "Growth",
]

STRICT_WHITELIST = [
    "SO2_exceed_diff", "CH4_zscore",
    "Lev_x_SO2_exceed", "Size_x_CO_zscore",
]

IND_STAT_CURRENT_COLS = {
    "ind_change_rate", "ind_up_rate", "ind_down_rate",
}

BASE_DROP_ROOTS: Set[str] = {"mean_CH4", "mean_NO2", "mean_CO", "mean_SO2"}

# ── 模型分流集合 ────────────────────────────────────────────────────────
LINEAR_MODELS:   set = {"LR", "LinearSVM"}
NO_SMOTE_MODELS: set = {"BalancedRF", "EasyEnsemble"}
TREE_MODELS:     set = {
    "RF", "XGBoost", "LightGBM",
    "CatBoost", "HistGradBoost",
}

# ── 模型列表 ────────────────────────────────────────────────────────────
BASELINE_MODEL_NAMES: List[str] = [
    "LR",
    "RF",
    "BalancedRF",
    "EasyEnsemble",
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "LinearSVM",
    "HistGradBoost",
]
MAIN_MODEL_NAME = "10_CatBoost_Main"
ALL_MODEL_NAMES = BASELINE_MODEL_NAMES + [MAIN_MODEL_NAME]

# ── 标签体系 ────────────────────────────────────────────────────────────
LABEL_MAP  = {-1: 0, 0: 1, 1: 2}
INV_LABEL  = {0: -1, 1: 0, 2: 1}
LABEL_LIST = [-1, 0, 1]

# ── 指标字段名常量 ──────────────────────────────────────────────────────
_MK_F1      = "Macro-F1（宏平均F1分数）"
_MK_AUC_ROC = "Macro-AUC-ROC（宏平均受试者工作特征曲线下面积）"
_MK_AUC_PR  = "AUC-PR（宏平均精确率-召回率曲线下面积）"
_MK_REC_DN  = "Recall-下降(↓)（下降类别召回率）"
_MK_REC_ST  = "Recall-稳定(平)（稳定类别召回率）"
_MK_REC_UP  = "Recall-上升(↑)（上升类别召回率）"
_MK_REC_MAC = "Macro-Recall（宏平均召回率）"
_MK_ACC     = "Accuracy（准确率）"
_MK_PRE_MAC = "Macro-Precision（宏平均精确率）"
_MK_F1_DN   = "F1-下降(↓)（下降类别F1分数）"
_MK_F1_ST   = "F1-稳定(平)（稳定类别F1分数）"
_MK_F1_UP   = "F1-上升(↑)（上升类别F1分数）"
_MK_PRE_DN  = "Precision-下降(↓)（下降类别精确率）"
_MK_PRE_ST  = "Precision-稳定(平)（稳定类别精确率）"
_MK_PRE_UP  = "Precision-上升(↑)（上升类别精确率）"
_MK_SUP_DN  = "Support-下降(↓)（下降类别样本数）"
_MK_SUP_ST  = "Support-稳定(平)（稳定类别样本数）"
_MK_SUP_UP  = "Support-上升(↑)（上升类别样本数）"

SUMMARY_METRIC_COLS = [
    _MK_F1, _MK_AUC_ROC, _MK_AUC_PR,
    _MK_REC_DN, _MK_REC_ST, _MK_REC_UP,
    _MK_REC_MAC, _MK_ACC, _MK_PRE_MAC,
]

ALL_METRIC_COLS = SUMMARY_METRIC_COLS + [
    _MK_F1_DN, _MK_F1_ST, _MK_F1_UP,
    _MK_PRE_DN, _MK_PRE_ST, _MK_PRE_UP,
    _MK_SUP_DN, _MK_SUP_ST, _MK_SUP_UP,
]

# ── 固定滚动窗口 ────────────────────────────────────────────────────────
FIXED_WINDOWS: List[Dict] = [
    {
        "window_id":   "W1_train2018-2020_test2022",
        "train_years": [2018, 2019, 2020],
        "val_year":    2021,
        "test_year":   2022,
    },
    {
        "window_id":   "W2_train2019-2021_test2023",
        "train_years": [2019, 2020, 2021],
        "val_year":    2022,
        "test_year":   2023,
    },
    {
        "window_id":   "W3_train2020-2022_test2024",
        "train_years": [2020, 2021, 2022],
        "val_year":    2023,
        "test_year":   2024,
    },
]


# ════════════════════════════════════════════════════════════════════════
# 标签映射工具
# ════════════════════════════════════════════════════════════════════════
def map_labels(y):
    return np.array([LABEL_MAP[int(v)] for v in y])


def inv_labels(y):
    return np.array([INV_LABEL[int(v)] for v in y])


# ════════════════════════════════════════════════════════════════════════
# 标签完整性检查
# ════════════════════════════════════════════════════════════════════════
def _check_label_integrity(
        dtr: pd.DataFrame,
        dte: pd.DataFrame,
        window_id: str,
) -> Tuple[bool, str]:
    def _label3(df_):
        return np.where(df_[TARGET_COL] > 0,  1,
               np.where(df_[TARGET_COL] < 0, -1, 0)).astype(int)

    ytr = _label3(dtr)
    yte = _label3(dte)
    utr = set(np.unique(ytr).tolist())
    ute = set(np.unique(yte).tolist())
    need   = {-1, 0, 1}
    prefix = f"  [标签校验][{window_id}]"

    missing_train = need - utr
    if missing_train:
        reason = f"训练集缺少类别 {missing_train}，跳过"
        print(f"{prefix} ❌ {reason}")
        return False, reason

    if len(ute) < 2:
        reason = f"测试集只有 1 个类别 {ute}，跳过"
        print(f"{prefix} ❌ {reason}")
        return False, reason

    from collections import Counter
    cnt = Counter(ytr.tolist())
    min_cls, min_n = min(cnt.items(), key=lambda x: x[1])
    lnm = {-1: "Down(↓)", 0: "Stable(平)", 1: "Up(↑)"}
    if min_n < 5:
        reason = f"训练集类别 {lnm[min_cls]} 仅有 {min_n} 条，跳过"
        print(f"{prefix} ❌ {reason}")
        return False, reason

    dist_tr = "  ".join(f"{lnm[k]}:{cnt[k]}" for k in [-1, 0, 1])
    nte     = Counter(yte.tolist())
    dist_te = "  ".join(f"{lnm[k]}:{nte.get(k, 0)}" for k in [-1, 0, 1])
    print(f"{prefix} ✅ 训练集={len(ytr)}行 [{dist_tr}]"
          f"  测试集={len(yte)}行 [{dist_te}]")
    return True, "ok"


# ════════════════════════════════════════════════════════════════════════
# 污染特征列匹配与删除
# ════════════════════════════════════════════════════════════════════════
def drop_pollution_cols_from_avail(
        avail: List[str],
        drop_roots: Set[str],
) -> List[str]:
    if not drop_roots:
        return []
    to_drop: Set[str] = set()
    for col in avail:
        for root in drop_roots:
            if col.startswith(root):
                to_drop.add(col); break
            if f"_{root}_" in col or col.endswith(f"_{root}"):
                to_drop.add(col); break
            if f"_x_{root}" in col:
                to_drop.add(col); break
            gas_name = root.split("_")[0]
            if (gas_name in POLL_ANOMALY_GASES
                    and f"_x_{gas_name}" in col
                    and any(r.startswith(gas_name) for r in drop_roots)):
                to_drop.add(col); break
    return [c for c in avail if c in to_drop]


def get_avail_after_base_drop(
        avail: List[str],
        base_drop_roots: Set[str],
) -> List[str]:
    if not base_drop_roots:
        return avail
    dropped = drop_pollution_cols_from_avail(avail, base_drop_roots)
    kept    = [c for c in avail if c not in set(dropped)]
    print(f"    [BASE_DROP] 删除 {len(dropped)} 列 → 剩余 {len(kept)} 列")
    return kept


# ════════════════════════════════════════════════════════════════════════
# 行业统计安全计算（防数据泄漏）
# ════════════════════════════════════════════════════════════════════════
def compute_industry_stats_from_train(
        dtr: pd.DataFrame,
        dv:  pd.DataFrame,
        dte: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if INDUSTRY_COL not in dtr.columns:
        return dtr, dv, dte

    def _up(s):
        ch = s[s["y_bin"] == 1]
        return float((ch[TARGET_COL] > 0).mean()) if len(ch) > 0 else 0.0

    def _dn(s):
        ch = s[s["y_bin"] == 1]
        return float((ch[TARGET_COL] < 0).mean()) if len(ch) > 0 else 0.0

    ist = (dtr.groupby([INDUSTRY_COL, YEAR_COL])["y_bin"]
           .agg(["mean", "count"]).reset_index())
    ist.columns = [INDUSTRY_COL, YEAR_COL, "ind_change_rate_train", "ind_n_train"]

    ur = dtr.groupby([INDUSTRY_COL, YEAR_COL]).apply(_up).reset_index(
        name="ind_up_rate_train")
    dr = dtr.groupby([INDUSTRY_COL, YEAR_COL]).apply(_dn).reset_index(
        name="ind_down_rate_train")
    ist = ist.merge(ur, on=[INDUSTRY_COL, YEAR_COL], how="left")
    ist = ist.merge(dr, on=[INDUSTRY_COL, YEAR_COL], how="left")
    ist.sort_values([INDUSTRY_COL, YEAR_COL], inplace=True)

    for col in ["ind_change_rate_train", "ind_up_rate_train", "ind_down_rate_train"]:
        g = ist.groupby(INDUSTRY_COL)[col]
        ist[f"{col}_lag1"]     = g.shift(1)
        ist[f"{col}_ma3_lag1"] = (g.rolling(3, min_periods=2).mean()
                                   .reset_index(level=0, drop=True).shift(1))

    max_train_year = int(dtr[YEAR_COL].max())
    ist_latest     = ist[ist[YEAR_COL] == max_train_year].copy()
    rename_map     = {c: c.replace("_train", "_safe") for c in ist.columns}
    ist            = ist.rename(columns=rename_map)
    ist_latest     = ist_latest.rename(columns=rename_map)

    def _is_leaky(col_name: str) -> bool:
        for stem in ["ind_change_rate", "ind_up_rate", "ind_down_rate"]:
            if stem in col_name and "_safe" not in col_name:
                return True
        return False

    dtr_c = dtr.drop(columns=[c for c in dtr.columns if _is_leaky(c)],
                     errors="ignore").copy()
    dv_c  = dv.drop(columns=[c for c in dv.columns  if _is_leaky(c)],
                    errors="ignore").copy()
    dte_c = dte.drop(columns=[c for c in dte.columns if _is_leaky(c)],
                     errors="ignore").copy()

    train_join = [INDUSTRY_COL, YEAR_COL] + [
        c for c in ist.columns if c not in [INDUSTRY_COL, YEAR_COL, "ind_n_safe"]]
    dtr_c = dtr_c.merge(ist[train_join], on=[INDUSTRY_COL, YEAR_COL], how="left")

    safe_lag_cols = [INDUSTRY_COL] + [
        c for c in ist_latest.columns
        if "_safe_lag1" in c or "_safe_ma3_lag1" in c]
    ist_j = ist_latest[[c for c in safe_lag_cols if c in ist_latest.columns]].copy()
    dv_c  = dv_c.merge(ist_j,  on=INDUSTRY_COL, how="left")
    dte_c = dte_c.merge(ist_j, on=INDUSTRY_COL, how="left")

    for df_ in [dtr_c, dv_c, dte_c]:
        sc = [c for c in df_.columns if "_safe" in c
              and df_[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
        df_[sc] = df_[sc].fillna(0)

    added = [c for c in dtr_c.columns if "_safe" in c]
    print(f"    [FIX-2] 安全行业特征={len(added)}列  max_train_year={max_train_year}")
    return dtr_c, dv_c, dte_c


# ════════════════════════════════════════════════════════════════════════
# 动态类权重
# ════════════════════════════════════════════════════════════════════════
def compute_class_weights_3c(y_raw, mode=CW_MODE):
    from collections import Counter
    counts  = Counter(int(v) for v in y_raw)
    n_major = max(counts.values())
    if mode == "log":
        cw = {lbl: float(np.log(1.0 + n_major / max(counts.get(lbl, 1), 1)))
              for lbl in [-1, 0, 1]}
    else:
        cw = {lbl: float(np.sqrt(n_major / max(counts.get(lbl, 1), 1)))
              for lbl in [-1, 0, 1]}
    scale = max(cw.get(0, 1.0), 1e-9)
    return {lbl: round(v / scale, 4) for lbl, v in cw.items()}


def build_sample_weights(y_raw, class_weight_dict):
    sw = np.array([class_weight_dict.get(int(v), 1.0) for v in y_raw],
                  dtype=np.float32)
    msw = sw.mean()
    if msw > 1e-9:
        sw = sw / msw
    return sw


# ════════════════════════════════════════════════════════════════════════
# 特征工程
# ════════════════════════════════════════════════════════════════════════
def feature_engineering(df: pd.DataFrame, core_features=None) -> pd.DataFrame:
    if core_features is None:
        core_features = CORE_NUM_FEATURES
    df = df.copy()
    df.sort_values([ID_COL, YEAR_COL], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df["y_bin"] = (df[TARGET_COL] != 0).astype(int)
    df["y_dir"] = np.where(df[TARGET_COL] > 0, 1, 0)

    grp = df.groupby(ID_COL)
    for f in core_features:
        if f not in df.columns: continue
        df[f"{f}_lag1"] = grp[f].shift(1)
    df["y_lag1"]     = grp[TARGET_COL].shift(1)
    df["y_bin_lag1"] = grp["y_bin"].shift(1)

    for f in core_features:
        if f not in df.columns: continue
        r = grp[f].rolling(3, min_periods=2)
        df[f"{f}_ma3"]    = r.mean().reset_index(level=0, drop=True)
        df[f"{f}_std3"]   = r.std().reset_index(level=0, drop=True)
        df[f"{f}_trend2"] = grp[f].diff(1)

    for f in core_features:
        if f not in df.columns: continue
        yg = df.groupby(YEAR_COL)[f]
        df[f"{f}_yr_rank"]   = yg.rank(pct=True)
        ym_ = yg.transform("mean")
        ys_ = yg.transform("std").replace(0, 1)
        df[f"{f}_yr_zscore"] = (df[f] - ym_) / ys_
        if INDUSTRY_COL in df.columns:
            yi = df.groupby([YEAR_COL, INDUSTRY_COL])[f]
            df[f"{f}_yi_rank"]   = yi.rank(pct=True)
            yim = yi.transform("mean")
            yis = yi.transform("std").replace(0, 1)
            df[f"{f}_yi_zscore"] = (df[f] - yim) / yis

    for sx in ["_yr_rank", "_yr_zscore", "_yi_rank", "_yi_zscore"]:
        for f in core_features:
            c = f"{f}{sx}"
            if c in df.columns:
                df[f"{c}_lag1"] = df.groupby(ID_COL)[c].shift(1)

    pc  = [c for c in ["NO2_chang", "SO2_chang", "CH4_chang", "CO_chang"]
           if c in df.columns and c in core_features]
    pcv = [c for c in ["CH4_CV", "NO2_CV", "CO_CV", "SO2_CV"]
           if c in df.columns and c in core_features]
    pm  = [c for c in ["mean_NO2", "mean_SO2", "mean_CH4", "mean_CO"]
           if c in df.columns and c in core_features]
    if pc:  df["poll_abs_sum"]  = df[pc].abs().sum(axis=1)
    if pcv: df["poll_cv_mean"]  = df[pcv].mean(axis=1)
    if pm:  df["poll_mean_sum"] = df[pm].sum(axis=1)

    for comp in ["poll_abs_sum", "poll_cv_mean", "poll_mean_sum"]:
        if comp in df.columns and INDUSTRY_COL in df.columns:
            df[f"{comp}_yi_rank"] = df.groupby(
                [YEAR_COL, INDUSTRY_COL])[comp].rank(pct=True)
            df[f"{comp}_yi_rank_lag1"] = df.groupby(
                ID_COL)[f"{comp}_yi_rank"].shift(1)

    if "punish_num" in df.columns and "punish_money" in df.columns:
        df["punish_interact"]  = df["punish_num"] * df["punish_money"]
    if "punish_num" in df.columns and "recognize" in df.columns:
        df["punish_recognize"] = df["punish_num"] * df["recognize"]
    if "prize" in df.columns and "patent" in df.columns:
        df["prize_patent"]     = df["prize"] * df["patent"]

    if "Size_yr_zscore" in df.columns:
        for poll in POLL_TREND_COLS:
            tc = f"{poll}_trend2"
            if tc in df.columns:
                df[f"{poll}_trend2_x_size_zscore"] = df[tc] * df["Size_yr_zscore"]
        for cv_col in ["CH4_CV", "NO2_CV", "CO_CV", "SO2_CV"]:
            if cv_col in df.columns:
                df[f"{cv_col}_x_size_zscore"] = df[cv_col] * df["Size_yr_zscore"]

    gas_grp = df.groupby(ID_COL)
    for gas in POLL_ANOMALY_GASES:
        zs_col = f"{gas}_zscore"
        ex_col = f"{gas}_exceed"
        if zs_col in df.columns:
            df[f"{gas}_zscore_lag1"] = gas_grp[zs_col].shift(1)
            df[f"{gas}_zscore_sq"]   = df[zs_col] ** 2
        if ex_col in df.columns:
            df[f"{gas}_exceed_diff"] = gas_grp[ex_col].diff(1)
            df[f"{gas}_exceed_sum2"] = (
                gas_grp[ex_col].rolling(2, min_periods=1).sum()
                               .reset_index(level=0, drop=True))
        if zs_col in df.columns and "Size_yr_zscore" in df.columns:
            df[f"{gas}_zscore_x_size"] = df[zs_col] * df["Size_yr_zscore"]
        if ex_col in df.columns and "Lev" in df.columns:
            df[f"{gas}_exceed_x_lev"] = df[ex_col] * df["Lev"]

    if "SO2_exceed_x_lev" in df.columns and "Lev_x_SO2_exceed" not in df.columns:
        df["Lev_x_SO2_exceed"] = df["SO2_exceed_x_lev"]
    if "CO_zscore_x_size" in df.columns and "Size_x_CO_zscore" not in df.columns:
        df["Size_x_CO_zscore"] = df["CO_zscore_x_size"]

    df["After_Carbon_Neutral_Goal"] = (
        df[YEAR_COL] >= CARBON_NEUTRAL_YEAR).astype(np.int8)
    for gas in POLL_ANOMALY_GASES:
        zs_col = f"{gas}_zscore"
        if zs_col in df.columns:
            df[f"{gas}_zscore_x_carbon"] = df[zs_col] * df["After_Carbon_Neutral_Goal"]
    if "poll_abs_sum" in df.columns:
        df["poll_abs_sum_x_carbon"] = df["poll_abs_sum"] * df["After_Carbon_Neutral_Goal"]

    if INDUSTRY_COL in df.columns:
        def _up2(s):
            ch = s[s["y_bin"] == 1]
            return float((ch[TARGET_COL] > 0).mean()) if len(ch) > 0 else 0.0

        def _dn2(s):
            ch = s[s["y_bin"] == 1]
            return float((ch[TARGET_COL] < 0).mean()) if len(ch) > 0 else 0.0

        ist = (df.groupby([INDUSTRY_COL, YEAR_COL])["y_bin"]
               .agg(["mean", "sum", "count"]).reset_index())
        ist.columns = [INDUSTRY_COL, YEAR_COL,
                       "ind_change_rate", "ind_change_count", "ind_total"]
        ur = df.groupby([INDUSTRY_COL, YEAR_COL]).apply(_up2).reset_index(
            name="ind_up_rate")
        dr = df.groupby([INDUSTRY_COL, YEAR_COL]).apply(_dn2).reset_index(
            name="ind_down_rate")
        ist = ist.merge(ur, on=[INDUSTRY_COL, YEAR_COL], how="left")
        ist = ist.merge(dr, on=[INDUSTRY_COL, YEAR_COL], how="left")
        ist.sort_values([INDUSTRY_COL, YEAR_COL], inplace=True)
        for col in ["ind_change_rate", "ind_up_rate", "ind_down_rate"]:
            g = ist.groupby(INDUSTRY_COL)[col]
            ist[f"{col}_lag1"]     = g.shift(1)
            ist[f"{col}_ma3_lag1"] = (
                g.rolling(3, min_periods=2).mean()
                 .reset_index(level=0, drop=True).shift(1))
        df = df.merge(
            ist.drop(columns=["ind_change_count", "ind_total"], errors="ignore"),
            on=[INDUSTRY_COL, YEAR_COL], how="left")

    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    df[str_cols] = df[str_cols].fillna("")
    return df


def onehot_industry(df: pd.DataFrame) -> pd.DataFrame:
    if INDUSTRY_COL not in df.columns:
        return df
    return pd.concat(
        [df, pd.get_dummies(df[INDUSTRY_COL], prefix="ind", dtype=np.int8)],
        axis=1)


def get_feature_columns_fixed(df: pd.DataFrame) -> List[str]:
    exclude = {YEAR_COL, ID_COL, INDUSTRY_COL, TARGET_COL, "y_bin", "y_dir"}
    exclude.update(RAW_POLLUTION_COLS_DROP)
    for stem in IND_STAT_CURRENT_COLS:
        for suffix in ["", "_lag1", "_ma3_lag1"]:
            exclude.add(f"{stem}{suffix}")
    return [c for c in df.columns
            if c not in exclude
            and df[c].dtype in [np.float64, np.float32,
                                  np.int64, np.int32, np.int8]]


# ════════════════════════════════════════════════════════════════════════
# 分位数截断
# ════════════════════════════════════════════════════════════════════════
def cap_fit(X: np.ndarray) -> Dict:
    return {j: (float(np.percentile(X[:, j], CAP_LOWER_PCT * 100)),
                float(np.percentile(X[:, j], CAP_UPPER_PCT * 100)))
            for j in range(X.shape[1])}


def cap_transform(X: np.ndarray, bounds: Dict) -> np.ndarray:
    X = X.copy()
    for j, (lo, hi) in bounds.items():
        X[:, j] = np.clip(X[:, j], lo, hi)
    return X


# ════════════════════════════════════════════════════════════════════════
# 特征选择
# ════════════════════════════════════════════════════════════════════════
def select_features_rf(Xtr, ytr, names):
    if len(np.unique(ytr)) < 2:
        return np.ones(len(names), dtype=bool), list(names)
    rf = RandomForestClassifier(
        n_estimators=150, max_depth=10, class_weight="balanced",
        n_jobs=N_JOBS, random_state=RANDOM_STATE)
    rf.fit(Xtr, ytr)
    imp  = rf.feature_importances_
    mask = imp >= FEAT_IMP_THRESHOLD
    if mask.sum() < FEAT_MIN_KEEP:
        ti   = np.argsort(imp)[::-1][:FEAT_MIN_KEEP]
        mask = np.zeros(len(imp), dtype=bool)
        mask[ti] = True
    return mask, [f for f, v in zip(names, mask) if v]


def select_features_catboost(Xtr, ytr, names):
    strict_present = [f for f in STRICT_WHITELIST if f in names]
    if HAS_CB:
        try:
            selector = CatBoostClassifier(
                iterations=500, depth=6, l2_leaf_reg=10,
                random_strength=2, bootstrap_type="Bernoulli",
                subsample=0.8, verbose=0,
                random_seed=RANDOM_STATE, thread_count=N_JOBS_CATBOOST)
            selector.fit(Xtr, ytr)
            importances = np.array(selector.get_feature_importance())
            actual_k    = min(FEAT_TOPK, len(names))
            topk_idx    = set(np.argsort(importances)[::-1][:actual_k])
            mask = np.zeros(len(names), dtype=bool)
            mask[list(topk_idx)] = True
            for f in strict_present:
                if f in names and not mask[names.index(f)]:
                    mask[names.index(f)] = True
            return mask, [f for f, s in zip(names, mask) if s]
        except Exception as e:
            print(f"    [WARN] CatBoost特征选择失败（{e}），降级→RF")
    mask, sel = select_features_rf(Xtr, ytr, names)
    for f in strict_present:
        if f in names and not mask[names.index(f)]:
            mask[names.index(f)] = True
    return mask, [f for f, s in zip(names, mask) if s]


def filter_high_correlation(
        Xtr: np.ndarray,
        feat_names: List[str],
        threshold: float = CORR_THRESHOLD,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    n_feat = Xtr.shape[1]
    if n_feat <= 1:
        return Xtr, feat_names, np.ones(n_feat, dtype=bool)
    var_order = np.argsort(-Xtr.var(axis=0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr_mat = np.corrcoef(Xtr.T)
    corr_mat  = np.nan_to_num(corr_mat, nan=0.0)
    keep_set  = []
    drop_mask = np.zeros(n_feat, dtype=bool)
    for idx in var_order:
        if drop_mask[idx]: continue
        if any(abs(corr_mat[idx, k]) > threshold for k in keep_set):
            drop_mask[idx] = True
        else:
            keep_set.append(idx)
    keep_mask    = ~drop_mask
    kept_names   = [f for f, k in zip(feat_names, keep_mask) if k]
    Xtr_filtered = Xtr[:, keep_mask]
    print(f"    [共线性过滤] |r|>{threshold:.2f}  "
          f"删除={int(drop_mask.sum())}  保留={len(kept_names)}")
    return Xtr_filtered, kept_names, keep_mask


# ════════════════════════════════════════════════════════════════════════
# FIX-A：线性模型 StandardScaler + PCA
# ════════════════════════════════════════════════════════════════════════
def apply_pca_for_linear(
        Xtr: np.ndarray,
        Xv:  np.ndarray,
        Xte: np.ndarray,
        var: float = PCA_VARIANCE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_feat = Xtr.shape[1]
    if n_feat < 2:
        sc = StandardScaler()
        return (sc.fit_transform(Xtr).astype(np.float32),
                sc.transform(Xv).astype(np.float32),
                sc.transform(Xte).astype(np.float32))
    scaler  = StandardScaler()
    Xtr_sc  = scaler.fit_transform(Xtr)
    Xv_sc   = scaler.transform(Xv)
    Xte_sc  = scaler.transform(Xte)
    pca     = PCA(n_components=var, random_state=RANDOM_STATE)
    Xtr_pca = pca.fit_transform(Xtr_sc)
    Xv_pca  = pca.transform(Xv_sc)
    Xte_pca = pca.transform(Xte_sc)
    print(f"    [FIX-A 线性PCA] {n_feat}维 → {pca.n_components_}主成分"
          f"（解释方差 ≥ {var:.0%}）")
    return (Xtr_pca.astype(np.float32),
            Xv_pca.astype(np.float32),
            Xte_pca.astype(np.float32))


# ════════════════════════════════════════════════════════════════════════
# FIX-B：SMOTE
# ════════════════════════════════════════════════════════════════════════
def try_smote_3class(X: np.ndarray, y: np.ndarray):
    if not HAS_IMBLEARN:
        return X, y, "skip: no imblearn"
    mc = pd.Series(y).value_counts().min()
    k  = min(5, mc - 1)
    if k < 1:
        return X, y, f"skip: minority={mc}"
    try:
        sm = SMOTE(k_neighbors=k, random_state=RANDOM_STATE)
        Xr, yr = sm.fit_resample(X, y)
        if len(np.unique(yr)) < len(np.unique(y)):
            return X, y, "skip: collapsed"
        return Xr, yr, f"ok: {len(y)}→{len(Xr)}"
    except Exception as e:
        return X, y, f"fail: {str(e)[:60]}"


# ════════════════════════════════════════════════════════════════════════
# 验证集拆分（供 Optuna 使用）
# ════════════════════════════════════════════════════════════════════════
def split_val_for_optuna(
        Xv: np.ndarray, yv: np.ndarray,
        dv: pd.DataFrame,
        ratio: float = OPTUNA_VAL_RATIO,
        rs: int = RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng       = np.random.RandomState(rs)
    companies = np.array(sorted(dv[ID_COL].unique()))
    rng.shuffle(companies)
    n_tune    = max(1, int(len(companies) * ratio))
    tune_ids  = set(companies[:n_tune])
    tune_mask = dv[ID_COL].isin(tune_ids).values
    eval_mask = ~tune_mask
    if tune_mask.sum() == 0 or eval_mask.sum() == 0:
        return Xv, yv, Xv, yv
    print(f"    [验证集拆分] tune={tune_mask.sum()}行  eval={eval_mask.sum()}行")
    return (Xv[tune_mask], yv[tune_mask], Xv[eval_mask], yv[eval_mask])


# ════════════════════════════════════════════════════════════════════════
# 指标计算
# ════════════════════════════════════════════════════════════════════════
def compute_metrics(y_true, y_pred, proba3=None) -> Dict:
    m      = {}
    labels = LABEL_LIST
    m[_MK_ACC]     = round(accuracy_score(y_true, y_pred), 5)
    m[_MK_F1]      = round(f1_score(y_true, y_pred, average="macro",
                                     zero_division=0, labels=labels), 5)
    m[_MK_PRE_MAC] = round(precision_score(y_true, y_pred, average="macro",
                                             zero_division=0, labels=labels), 5)
    m[_MK_REC_MAC] = round(recall_score(y_true, y_pred, average="macro",
                                          zero_division=0, labels=labels), 5)
    pc_, rc_, fc_, sc_ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    for i, (pk, rk, fk, sk) in enumerate([
        (_MK_PRE_DN, _MK_REC_DN, _MK_F1_DN, _MK_SUP_DN),
        (_MK_PRE_ST, _MK_REC_ST, _MK_F1_ST, _MK_SUP_ST),
        (_MK_PRE_UP, _MK_REC_UP, _MK_F1_UP, _MK_SUP_UP),
    ]):
        m[pk] = round(pc_[i], 5)
        m[rk] = round(rc_[i], 5)
        m[fk] = round(fc_[i], 5)
        m[sk] = int(sc_[i])

    if proba3 is not None and proba3.shape[1] == 3:
        try:
            yb = label_binarize(y_true, classes=labels)
            m[_MK_AUC_ROC] = round(
                roc_auc_score(yb, proba3, multi_class="ovr", average="macro"), 5)
        except Exception:
            m[_MK_AUC_ROC] = np.nan
        try:
            yb      = label_binarize(y_true, classes=labels)
            ap_list = [average_precision_score(yb[:, ci], proba3[:, ci])
                       for ci in range(3) if yb[:, ci].sum() > 0]
            m[_MK_AUC_PR] = (round(float(np.mean(ap_list)), 5)
                              if ap_list else np.nan)
        except Exception:
            m[_MK_AUC_PR] = np.nan
    else:
        m[_MK_AUC_ROC] = np.nan
        m[_MK_AUC_PR]  = np.nan
    return m


def _safe_normalize_proba(p3: np.ndarray) -> np.ndarray:
    p3 = np.where(np.isfinite(p3), p3, 0.0).astype(np.float32)
    rs = p3.sum(axis=1, keepdims=True)
    rs = np.where(rs > 1e-9, rs, 1.0)
    return p3 / rs


def _align_proba_to_label_list(mdl, X: np.ndarray) -> np.ndarray:
    proba_raw     = mdl.predict_proba(X)
    model_classes = list(mdl.classes_)
    n             = len(X)
    aligned       = np.zeros((n, 3), dtype=np.float32)
    for ti, lbl in enumerate(LABEL_LIST):
        if lbl in model_classes:
            si = model_classes.index(lbl)
            if si < proba_raw.shape[1]:
                aligned[:, ti] = proba_raw[:, si]
    return aligned


# ════════════════════════════════════════════════════════════════════════
# Optuna 目标函数（主模型专用）
# ════════════════════════════════════════════════════════════════════════
def _catboost_optuna_objective(
        trial, Xtr, ytr, Xv_tune, yv_tune, sample_sw,
) -> float:
    if not HAS_CB:
        return 0.0
    bootstrap_type = trial.suggest_categorical(
        "bootstrap_type", ["Bernoulli", "Bayesian"])
    params = {
        "iterations":        trial.suggest_int("iterations", 200, 800),
        "depth":             trial.suggest_int("depth", 4, 8),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "l2_leaf_reg":       trial.suggest_float("l2_leaf_reg", 1.0, OPTUNA_L2_MAX, log=True),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "random_strength":   trial.suggest_float("random_strength", 0.1, 5.0),
        "min_data_in_leaf":  trial.suggest_int("min_data_in_leaf", 5, 50),
        "bootstrap_type":    bootstrap_type,
        "loss_function":     "MultiClass",
        "eval_metric":       "TotalF1:average=Macro",
        "verbose":           0,
        "random_seed":       RANDOM_STATE,
        "thread_count":      N_JOBS_CATBOOST,
    }
    if bootstrap_type == "Bernoulli":
        params["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
    else:
        params["bagging_temperature"] = trial.suggest_float(
            "bagging_temperature", 0.0, 2.0)
    try:
        mdl = CatBoostClassifier(**params)
        mdl.fit(Xtr, ytr, sample_weight=sample_sw)
        pred = inv_labels(mdl.predict(Xv_tune).flatten().astype(int))
        return f1_score(yv_tune, pred, average="macro", zero_division=0)
    except Exception:
        return 0.0


# ════════════════════════════════════════════════════════════════════════
# 主模型训练（CatBoost + Optuna）
# ════════════════════════════════════════════════════════════════════════
def run_catboost_main_window(
        Xtr_tree: np.ndarray, ytr: np.ndarray,
        Xtr_smote: np.ndarray, ytr_smote: np.ndarray,
        Xv_tree: np.ndarray, yv: np.ndarray,
        Xte_tree: np.ndarray, yte: np.ndarray,
        dv: pd.DataFrame,
) -> Tuple[Dict, Dict]:
    """
    返回 (metrics, debug_info)
    """
    if not HAS_CB:
        return {}, {"skip_reason": "no_catboost"}

    debug_info = {}
    Xv_tune, yv_tune, Xv_eval, yv_eval = split_val_for_optuna(
        Xv_tree, yv, dv, ratio=OPTUNA_VAL_RATIO)

    cw_smote  = compute_class_weights_3c(ytr_smote)
    sample_sw = build_sample_weights(ytr_smote, cw_smote).astype(np.float32)

    ytr_mapped = map_labels(ytr_smote)
    n_trials   = int(np.clip(len(Xtr_smote) // OPTUNA_TRIALS_PER_N,
                              OPTUNA_N_TRIALS_MIN, OPTUNA_N_TRIALS_MAX))
    print(f"    [Optuna] n_trials={n_trials}（训练集{len(Xtr_smote)}行）")
    debug_info["n_trials_actual"] = n_trials

    if HAS_OPTUNA:
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5, n_warmup_steps=0))
        study.optimize(
            lambda trial: _catboost_optuna_objective(
                trial, Xtr_smote, ytr_mapped, Xv_tune, yv_tune, sample_sw),
            n_trials=n_trials, show_progress_bar=False, n_jobs=1)
        best_params = study.best_params
        debug_info["optuna_best_val_f1"] = round(study.best_value, 5)
        debug_info["optuna_best_params"] = best_params
        print(f"    [Optuna] 最优val F1={study.best_value:.4f}")
    else:
        best_params = {
            "iterations": 500, "depth": 6, "learning_rate": 0.05,
            "l2_leaf_reg": 5.0, "subsample": 0.8,
            "colsample_bylevel": 0.8, "random_strength": 1.0,
            "min_data_in_leaf": 10, "bootstrap_type": "Bernoulli",
        }
        debug_info["optuna_best_val_f1"] = np.nan
        debug_info["optuna_best_params"] = best_params

    bootstrap_type = best_params.get("bootstrap_type", "Bernoulli")
    final_params = {
        "iterations":        best_params.get("iterations", 500),
        "depth":             best_params.get("depth", 6),
        "learning_rate":     best_params.get("learning_rate", 0.05),
        "l2_leaf_reg":       best_params.get("l2_leaf_reg", 5.0),
        "colsample_bylevel": best_params.get("colsample_bylevel", 0.8),
        "random_strength":   best_params.get("random_strength", 1.0),
        "min_data_in_leaf":  best_params.get("min_data_in_leaf", 10),
        "bootstrap_type":    bootstrap_type,
        "loss_function":     "MultiClass",
        "eval_metric":       "TotalF1:average=Macro",
        "verbose":           0,
        "random_seed":       RANDOM_STATE,
        "thread_count":      N_JOBS_CATBOOST,
    }
    if bootstrap_type == "Bernoulli":
        final_params["subsample"] = best_params.get("subsample", 0.8)
    else:
        final_params["bagging_temperature"] = best_params.get(
            "bagging_temperature", 0.5)

    final_model = CatBoostClassifier(**final_params)
    final_model.fit(Xtr_smote, ytr_mapped, sample_weight=sample_sw)
    print(f"    [最终模型] iters={final_params['iterations']}"
          f"  depth={final_params['depth']}"
          f"  lr={final_params['learning_rate']:.4f}")

    proba_te = _safe_normalize_proba(
        final_model.predict_proba(Xte_tree).astype(np.float32))
    pred_te  = inv_labels(final_model.predict(Xte_tree).flatten().astype(int))
    metrics  = compute_metrics(yte, pred_te, proba_te)

    pred_ve = inv_labels(final_model.predict(Xv_eval).flatten().astype(int))
    val_mf1 = f1_score(yv_eval, pred_ve, average="macro", zero_division=0)
    debug_info["val_eval_macro_f1"] = round(val_mf1, 5)
    print(f"    [{MAIN_MODEL_NAME}] val F1={val_mf1:.4f}"
          f"  test F1={metrics.get(_MK_F1, float('nan')):.4f}")

    return metrics, debug_info


# ════════════════════════════════════════════════════════════════════════
# 基线模型工厂
# ════════════════════════════════════════════════════════════════════════
def _auto_params(n: int) -> Dict:
    if n < 10_000:   return dict(ne=300, depth=6, lr=0.05, ml=20)
    elif n < 20_000: return dict(ne=500, depth=8, lr=0.05, ml=12)
    else:            return dict(ne=600, depth=9, lr=0.04, ml=10)


def build_baseline_model(
        model_name: str,
        class_weight_dict: dict = None,
):
    """返回 (model_instance, needs_mapped_labels)"""
    cw_sklearn = class_weight_dict if class_weight_dict else "balanced"

    if model_name == "LR":
        import sklearn
        lr_kwargs = dict(solver="saga", max_iter=2000,
                         class_weight=cw_sklearn, C=1.0,
                         random_state=RANDOM_STATE, n_jobs=N_JOBS)
        sk_ver = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
        if sk_ver < (1, 3):
            lr_kwargs["multi_class"] = "multinomial"
        return LogisticRegression(**lr_kwargs), False

    elif model_name == "RF":
        return RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=5,
            class_weight=cw_sklearn, n_jobs=N_JOBS,
            random_state=RANDOM_STATE), False

    elif model_name == "BalancedRF":
        if not HAS_IMBLEARN:
            return None, False
        return BalancedRandomForestClassifier(
            n_estimators=300, replacement=True,
            n_jobs=N_JOBS, random_state=RANDOM_STATE), False

    elif model_name == "EasyEnsemble":
        if not HAS_IMBLEARN:
            return None, False
        return EasyEnsembleClassifier(
            n_estimators=10, n_jobs=N_JOBS,
            random_state=RANDOM_STATE), False

    elif model_name == "XGBoost":
        if not HAS_XGB:
            return None, True
        a = _auto_params(500)
        return xgb.XGBClassifier(
            n_estimators=a["ne"], max_depth=a["depth"], learning_rate=a["lr"],
            subsample=0.8, colsample_bytree=0.8,
            objective="multi:softprob", num_class=3, eval_metric="mlogloss",
            tree_method="hist", random_state=RANDOM_STATE,
            n_jobs=N_JOBS, verbosity=0,
            use_label_encoder=False), True

    elif model_name == "LightGBM":
        if not HAS_LGB:
            return None, True
        return None, True   # 使用 lgb.train API，此处返回 None

    elif model_name == "CatBoost":
        if not HAS_CB:
            return None, True
        a = _auto_params(500)
        return CatBoostClassifier(
            iterations=a["ne"], depth=min(a["depth"], 8),
            learning_rate=a["lr"], loss_function="MultiClass",
            auto_class_weights="Balanced", verbose=0,
            random_seed=RANDOM_STATE, thread_count=N_JOBS_CATBOOST), True

    elif model_name == "LinearSVM":
        base = LinearSVC(multi_class="ovr", class_weight=cw_sklearn,
                         max_iter=2000, random_state=RANDOM_STATE, C=1.0)
        return CalibratedClassifierCV(
            estimator=base, method="sigmoid", cv=3), False

    elif model_name == "HistGradBoost":
        try:
            mdl = HistGradientBoostingClassifier(
                max_iter=300, max_depth=6, learning_rate=0.05,
                min_samples_leaf=20, l2_regularization=0.1,
                class_weight="balanced", random_state=RANDOM_STATE)
        except TypeError:
            mdl = HistGradientBoostingClassifier(
                max_iter=300, max_depth=6, learning_rate=0.05,
                min_samples_leaf=20, l2_regularization=0.1,
                random_state=RANDOM_STATE)
        return mdl, False

    else:
        raise ValueError(f"未知基线模型: {model_name}")


def train_and_predict_baseline(
        model_name: str,
        Xtr: np.ndarray, ytr: np.ndarray,
        Xv:  np.ndarray, yv:  np.ndarray,
        Xte: np.ndarray, yte: np.ndarray,
        class_weight_dict: dict = None,
) -> Dict:
    """训练基线模型并返回 metrics 字典。"""
    if class_weight_dict is None:
        class_weight_dict = {-1: 1.0, 0: 1.0, 1: 1.0}

    mdl, needs_mapped = build_baseline_model(model_name, class_weight_dict)
    if mdl is None and model_name != "LightGBM":
        return {}

    sample_sw = build_sample_weights(ytr, class_weight_dict)
    ytr_fit   = map_labels(ytr) if needs_mapped else ytr

    try:
        if model_name == "LightGBM":
            if not HAS_LGB:
                return {}
            a = _auto_params(len(ytr))
            params = {
                "objective": "multiclass", "num_class": 3,
                "metric": "multi_logloss", "seed": RANDOM_STATE,
                "learning_rate": a["lr"], "num_leaves": 63,
                "feature_fraction": 0.8, "bagging_fraction": 0.8,
                "bagging_freq": 5, "min_child_samples": a["ml"],
                "verbosity": -1, "min_gain_to_split": 0.01,
            }
            lgb_model = lgb.train(
                params,
                lgb.Dataset(Xtr, label=map_labels(ytr),
                            weight=sample_sw.astype(np.float32)),
                num_boost_round=a["ne"])
            proba_raw = lgb_model.predict(Xte)
            y_pred    = inv_labels(np.argmax(proba_raw, axis=1))

        elif model_name == "XGBoost":
            a = _auto_params(len(ytr))
            mdl.n_estimators  = a["ne"]
            mdl.max_depth     = a["depth"]
            mdl.learning_rate = a["lr"]
            mdl.fit(Xtr, ytr_fit, sample_weight=sample_sw.astype(np.float32))
            proba_raw = mdl.predict_proba(Xte)
            y_pred    = inv_labels(mdl.predict(Xte))

        elif model_name == "CatBoost":
            mdl.fit(Xtr, ytr_fit)
            proba_raw = mdl.predict_proba(Xte)
            y_pred    = inv_labels(mdl.predict(Xte).flatten().astype(int))

        elif model_name == "HistGradBoost":
            try:
                mdl.fit(Xtr, ytr_fit, sample_weight=sample_sw.astype(np.float32))
            except TypeError:
                mdl.fit(Xtr, ytr_fit)
            proba_raw = _align_proba_to_label_list(mdl, Xte)
            y_pred    = mdl.predict(Xte)

        else:
            mdl.fit(Xtr, ytr_fit)
            proba_raw = _align_proba_to_label_list(mdl, Xte)
            y_pred    = mdl.predict(Xte)

        proba_raw = _safe_normalize_proba(
            np.array(proba_raw, dtype=np.float32))
        return compute_metrics(yte, y_pred, proba_raw)

    except Exception as e:
        print(f"    [{model_name}] 训练/预测失败: {e}")
        return {}


# ════════════════════════════════════════════════════════════════════════
# 构建单条记录
# ════════════════════════════════════════════════════════════════════════
def _make_record(
        model_name: str,
        window_id: str,
        train_years: List[int],
        val_year: int,
        test_year: int,
        train_n: int,
        test_n: int,
        n_after_base: int,
        metrics: Dict,
        elapsed: float,
        preprocessing_route: str = "",
        base_drop_roots: Set[str] = None,
        skip_reason: str = "",
) -> Dict:
    rec = {
        "window_id（窗口标识）":                      window_id,
        "model（模型名称）":                           model_name,
        "preprocessing_route（预处理路径）":           preprocessing_route,
        "train_years（训练年份）":                     str(train_years),
        "val_year（验证年份）":                        val_year,
        "test_year（测试年份）":                       test_year,
        "train_n（训练集样本量）":                     train_n,
        "test_n（测试集样本量）":                      test_n,
        "n_feats_after_base_drop（基底剔除后特征数）": n_after_base,
        "base_dropped_roots（永久剔除根词）":          str(sorted(base_drop_roots))
                                                        if base_drop_roots else "none",
        "elapsed_sec（耗时秒）":                       elapsed,
        "skip_reason（跳过原因）":                     skip_reason,
    }
    for mk in ALL_METRIC_COLS:
        rec[mk] = metrics.get(mk, np.nan)
    return rec


# ════════════════════════════════════════════════════════════════════════
# 单窗口实验：所有模型
# ════════════════════════════════════════════════════════════════════════
def run_fixed_window(
        dtr_raw: pd.DataFrame,
        dv_raw:  pd.DataFrame,
        dte_raw: pd.DataFrame,
        feat_names: List[str],
        window_id: str,
        train_years: List[int],
        val_year: int,
        test_year: int,
        base_drop_roots: Set[str] = None,
) -> List[Dict]:
    """
    返回
    ----
    records : List[Dict]  本窗口所有模型的指标记录
    """
    if base_drop_roots is None:
        base_drop_roots = set()

    ok, reason = _check_label_integrity(dtr_raw, dte_raw, window_id)
    if not ok:
        skip_rec = _make_record(
            model_name="SKIPPED", window_id=window_id,
            train_years=train_years, val_year=val_year, test_year=test_year,
            train_n=len(dtr_raw), test_n=len(dte_raw),
            n_after_base=0, metrics={}, elapsed=0.0,
            preprocessing_route="N/A",
            base_drop_roots=base_drop_roots,
            skip_reason=reason)
        return [skip_rec]

    records: List[Dict] = []

    # ── 行业统计（防泄漏） ───────────────────────────────────────────
    dtr, dv, dte = compute_industry_stats_from_train(dtr_raw, dv_raw, dte_raw)

    # ── 基底剔除 ─────────────────────────────────────────────────────
    avail_all = [f for f in feat_names
                 if f in dtr.columns and f in dv.columns and f in dte.columns]
    avail     = get_avail_after_base_drop(avail_all, base_drop_roots)
    if len(avail) == 0:
        print(f"    [SKIP] 剔除后无可用特征列")
        return []

    # ── 原始特征矩阵构建 ─────────────────────────────────────────────
    def _bX(df_: pd.DataFrame) -> np.ndarray:
        arr = df_[avail].values.astype(np.float32)
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return arr

    Xtr_r = _bX(dtr)
    Xv_r  = _bX(dv)
    Xte_r = _bX(dte)

    bounds = cap_fit(Xtr_r)
    Xtr_c  = cap_transform(Xtr_r, bounds)
    Xv_c   = cap_transform(Xv_r,  bounds)
    Xte_c  = cap_transform(Xte_r, bounds)

    def _l3(df_: pd.DataFrame) -> np.ndarray:
        return np.where(df_[TARGET_COL] > 0,  1,
               np.where(df_[TARGET_COL] < 0, -1, 0)).astype(int)

    ytr3 = _l3(dtr)
    yv3  = _l3(dv)
    yte3 = _l3(dte)

    if len(np.unique(ytr3)) < 2:
        print("    [SKIP] 训练集类别数不足")
        return []

    cw_dict = compute_class_weights_3c(ytr3)
    print(f"    [动态类权重] {cw_dict}")

    # ── 特征选择 → 共线性过滤 ────────────────────────────────────────
    ytrb = dtr["y_bin"].values
    feat_mask, sel_feats = select_features_catboost(Xtr_c, ytrb, avail)
    Xtr_s = Xtr_c[:, feat_mask]
    Xv_s  = Xv_c[:,  feat_mask]
    Xte_s = Xte_c[:, feat_mask]
    print(f"    [特征选择] {feat_mask.sum()} / {len(avail)}")

    Xtr_f, kept_feats, corr_mask = filter_high_correlation(
        Xtr_s, sel_feats, threshold=CORR_THRESHOLD)
    Xv_f  = Xv_s[:,  corr_mask]
    Xte_f = Xte_s[:, corr_mask]

    # ── 三条预处理管道 ────────────────────────────────────────────────
    # 管道1：树模型 + SMOTE
    Xtr_tree_sm, ytr_tree_sm, smote_tree = try_smote_3class(Xtr_f, ytr3)
    cw_tree = compute_class_weights_3c(ytr_tree_sm)
    print(f"    [树模型SMOTE] {smote_tree}")

    # 管道2：线性模型 + PCA + SMOTE
    Xtr_lin, Xv_lin, Xte_lin = apply_pca_for_linear(Xtr_f, Xv_f, Xte_f)
    Xtr_lin_sm, ytr_lin_sm, smote_lin = try_smote_3class(Xtr_lin, ytr3)
    cw_lin = compute_class_weights_3c(ytr_lin_sm)
    print(f"    [线性模型SMOTE] {smote_lin}")

    # 管道3：内置平衡（无 SMOTE）
    cw_nosmote = compute_class_weights_3c(ytr3)

    # ════════════════════════════════════════════════════════════════
    # 基线模型循环
    # ════════════════════════════════════════════════════════════════
    for model_name in BASELINE_MODEL_NAMES:
        t0 = time.time()
        print(f"\n  → [{window_id}] 基线: {model_name}")

        if model_name in LINEAR_MODELS:
            Xtr_u, ytr_u = Xtr_lin_sm, ytr_lin_sm
            Xv_u,  Xte_u = Xv_lin,    Xte_lin
            cw_u         = cw_lin
            route_tag    = "FIX-A: Linear+StandardScaler+PCA+SMOTE"
        elif model_name in NO_SMOTE_MODELS:
            Xtr_u, ytr_u = Xtr_f,  ytr3
            Xv_u,  Xte_u = Xv_f,   Xte_f
            cw_u         = cw_nosmote
            route_tag    = "FIX-B: NoSMOTE（内置平衡）"
        else:
            Xtr_u, ytr_u = Xtr_tree_sm, ytr_tree_sm
            Xv_u,  Xte_u = Xv_f,        Xte_f
            cw_u         = cw_tree
            route_tag    = "FIX-A: Tree+SMOTE"

        print(f"    [预处理路径] {route_tag}"
              f"  训练集={len(ytr_u)}行  特征={Xtr_u.shape[1]}维")

        metrics = train_and_predict_baseline(
            model_name=model_name,
            Xtr=Xtr_u, ytr=ytr_u,
            Xv=Xv_u,   yv=yv3,
            Xte=Xte_u, yte=yte3,
            class_weight_dict=cw_u)
        elapsed = round(time.time() - t0, 1)

        rec = _make_record(
            model_name=model_name, window_id=window_id,
            train_years=train_years, val_year=val_year, test_year=test_year,
            train_n=len(dtr), test_n=len(dte),
            n_after_base=len(avail),
            metrics=metrics, elapsed=elapsed,
            preprocessing_route=route_tag,
            base_drop_roots=base_drop_roots)
        records.append(rec)

        if metrics:
            print(f"    [{model_name}] F1={metrics.get(_MK_F1, np.nan):.4f}"
                  f"  AUC={metrics.get(_MK_AUC_ROC, np.nan):.4f}"
                  f"  ({elapsed}s)")
        else:
            print(f"    [{model_name}] ⚠ 无有效指标  ({elapsed}s)")

    # ════════════════════════════════════════════════════════════════
    # 主模型：CatBoost + Optuna
    # ════════════════════════════════════════════════════════════════
    print(f"\n  → 主模型: {MAIN_MODEL_NAME}  窗口: [{window_id}]")
    t_main = time.time()

    metrics_main, dbg = run_catboost_main_window(
        Xtr_tree=Xtr_f, ytr=ytr3,
        Xtr_smote=Xtr_tree_sm, ytr_smote=ytr_tree_sm,
        Xv_tree=Xv_f, yv=yv3,
        Xte_tree=Xte_f, yte=yte3,
        dv=dv_raw)
    elapsed_main = round(time.time() - t_main, 1)

    rec_main = _make_record(
        model_name=MAIN_MODEL_NAME, window_id=window_id,
        train_years=train_years, val_year=val_year, test_year=test_year,
        train_n=len(dtr), test_n=len(dte),
        n_after_base=len(avail),
        metrics=metrics_main, elapsed=elapsed_main,
        preprocessing_route="FIX-A: Tree+SMOTE+Optuna",
        base_drop_roots=base_drop_roots)

    # 附加主模型调优信息
    rec_main["optuna_best_val_f1"] = dbg.get("optuna_best_val_f1", np.nan)
    rec_main["optuna_n_trials"]    = dbg.get("n_trials_actual",    np.nan)
    rec_main["val_eval_macro_f1"]  = dbg.get("val_eval_macro_f1",  np.nan)
    rec_main["best_iterations"]    = dbg.get("optuna_best_params", {}).get("iterations",   np.nan)
    rec_main["best_depth"]         = dbg.get("optuna_best_params", {}).get("depth",         np.nan)
    rec_main["best_lr"]            = dbg.get("optuna_best_params", {}).get("learning_rate", np.nan)
    rec_main["best_l2_leaf_reg"]   = dbg.get("optuna_best_params", {}).get("l2_leaf_reg",   np.nan)
    records.append(rec_main)

    if metrics_main:
        print(f"\n    ★ [{window_id}]"
              f"  Macro-F1={metrics_main.get(_MK_F1, float('nan')):.4f}"
              f"  AUC-ROC={metrics_main.get(_MK_AUC_ROC, float('nan')):.4f}"
              f"  ({elapsed_main}s)")

    return records


# ════════════════════════════════════════════════════════════════════════
# 主实验函数
# ════════════════════════════════════════════════════════════════════════
def run_experiment(
        df: pd.DataFrame,
        feat_names: List[str],
        fixed_windows: List[Dict] = None,
        base_drop_roots: Set[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    返回
    ----
    df_detail  : 逐窗口逐模型明细 DataFrame
    df_summary : 所有模型跨窗口均值汇总 DataFrame
    """
    if fixed_windows   is None: fixed_windows   = FIXED_WINDOWS
    if base_drop_roots is None: base_drop_roots = set()

    all_records: List[Dict] = []
    t_total = time.time()

    print(f"\n{'█'*72}")
    print(f"  整合实验 v90（9 基线 + 主模型，无 SHAP，无 CM）")
    print(f"  BASE_DROP_ROOTS={sorted(base_drop_roots)}")
    print(f"  窗口数={len(fixed_windows)}  模型总数={len(ALL_MODEL_NAMES)}")
    print(f"{'█'*72}")

    for win in fixed_windows:
        window_id   = win["window_id"]
        train_years = win["train_years"]
        val_year    = win["val_year"]
        test_year   = win["test_year"]

        print(f"\n  {'═'*60}")
        print(f"  窗口: {window_id}")
        print(f"    训练={train_years}  验证={val_year}  测试={test_year}")
        print(f"  {'═'*60}")

        dtr_raw = df[df[YEAR_COL].isin(train_years)].copy()
        dv_raw  = df[df[YEAR_COL] == val_year].copy()
        dte_raw = df[df[YEAR_COL] == test_year].copy()

        if len(dtr_raw) == 0 or len(dv_raw) == 0 or len(dte_raw) == 0:
            print(f"  [SKIP] 数据集为空")
            continue

        print(f"    数据规模  训练={len(dtr_raw)}行"
              f"  验证={len(dv_raw)}行  测试={len(dte_raw)}行")

        records = run_fixed_window(
            dtr_raw=dtr_raw, dv_raw=dv_raw, dte_raw=dte_raw,
            feat_names=feat_names, window_id=window_id,
            train_years=train_years, val_year=val_year, test_year=test_year,
            base_drop_roots=base_drop_roots)

        all_records.extend(records)

    total_elapsed = round(time.time() - t_total, 1)
    print(f"\n  实验完成  总记录={len(all_records)}"
          f"  总耗时={total_elapsed}s（{total_elapsed/60:.1f}分钟）")

    if not all_records:
        return pd.DataFrame(), pd.DataFrame()

    df_detail = pd.DataFrame(all_records)

    # ── 跨窗口均值汇总 ────────────────────────────────────────────────
    df_valid  = df_detail[
        df_detail["skip_reason（跳过原因）"].fillna("") == ""].copy()
    exist_sum = [c for c in SUMMARY_METRIC_COLS if c in df_valid.columns]
    df_summary = (df_valid
                  .groupby("model（模型名称）")[exist_sum]
                  .mean().round(4).reset_index())

    # 按 ALL_MODEL_NAMES 顺序排列
    model_order_map = {m: i for i, m in enumerate(ALL_MODEL_NAMES)}
    df_summary["_sort"] = df_summary["model（模型名称）"].map(
        model_order_map).fillna(999)
    df_summary.sort_values("_sort", inplace=True)
    df_summary.drop(columns=["_sort"], inplace=True)
    df_summary.reset_index(drop=True, inplace=True)

    return df_detail, df_summary


# ════════════════════════════════════════════════════════════════════════
# 控制台打印汇总
# ════════════════════════════════════════════════════════════════════════
def print_summary(
        df_detail:  pd.DataFrame,
        df_summary: pd.DataFrame,
) -> None:
    SEP = "─" * 140

    # ── 预处理路径映射 ────────────────────────────────────────────────
    route_map: Dict[str, str] = {}
    for m in BASELINE_MODEL_NAMES:
        if m in LINEAR_MODELS:
            route_map[m] = "FIX-A: Linear+PCA+SMOTE"
        elif m in NO_SMOTE_MODELS:
            route_map[m] = "FIX-B: NoSMOTE"
        else:
            route_map[m] = "FIX-A: Tree+SMOTE"
    route_map[MAIN_MODEL_NAME] = "FIX-A: Tree+SMOTE+Optuna"

    # ── 跨窗口均值汇总表 ──────────────────────────────────────────────
    print(f"\n{'='*140}")
    print(f"  【所有模型跨窗口均值汇总】（9 基线 + 主模型）")
    print(f"{'='*140}")

    if not df_summary.empty:
        print(f"  {'模型名称':<24}  {'预处理路径':<28}  "
              f"{'Macro-F1':>10}  {'AUC-ROC':>10}  {'AUC-PR':>10}"
              f"  {'Rec(↓)':>8}  {'Rec(平)':>9}  {'Rec(↑)':>8}"
              f"  {'MacRec':>8}  {'Acc':>8}")
        print(f"  {SEP}")

        for _, row in df_summary.iterrows():
            mname = str(row.get("model（模型名称）", ""))
            route = route_map.get(mname, "")

            def _g(k, w=10):
                v = row.get(k, np.nan)
                return f"{v:{w}.4f}" if pd.notna(v) else f"{'NaN':>{w}}"

            tag = "  ★主" if mname == MAIN_MODEL_NAME else "    "
            print(f"  {mname:<24}  {route:<28}  "
                  f"{_g(_MK_F1)}  {_g(_MK_AUC_ROC)}  {_g(_MK_AUC_PR)}"
                  f"  {_g(_MK_REC_DN,8)}  {_g(_MK_REC_ST,9)}"
                  f"  {_g(_MK_REC_UP,8)}  {_g(_MK_REC_MAC,8)}"
                  f"  {_g(_MK_ACC,8)}{tag}")
        print(f"  {SEP}")

    # ── 逐窗口逐模型明细 ──────────────────────────────────────────────
    print(f"\n{'='*140}")
    print(f"  【逐窗口逐模型明细】")
    print(f"{'='*140}")

    if not df_detail.empty:
        df_v = df_detail[
            df_detail["skip_reason（跳过原因）"].fillna("") == ""].copy()

        for win in FIXED_WINDOWS:
            wid  = win["window_id"]
            df_w = df_v[df_v["window_id（窗口标识）"] == wid]
            if df_w.empty:
                continue
            print(f"\n  窗口: {wid}  测试年={win['test_year']}")
            print(f"  {'模型':<24}  {'Macro-F1':>10}  {'AUC-ROC':>10}"
                  f"  {'AUC-PR':>10}  {'Rec(↓)':>8}  {'Rec(平)':>9}"
                  f"  {'Rec(↑)':>8}  {'MacRec':>8}  {'Acc':>8}")
            print(f"  {SEP}")

            # 按 ALL_MODEL_NAMES 顺序输出
            model_order = {m: i for i, m in enumerate(ALL_MODEL_NAMES)}
            df_w = df_w.copy()
            df_w["_sort"] = df_w["model（模型名称）"].map(model_order).fillna(999)
            df_w.sort_values("_sort", inplace=True)

            for _, row in df_w.iterrows():
                def _g(k, w=10):
                    v = row.get(k, np.nan)
                    return f"{v:{w}.4f}" if pd.notna(v) else f"{'NaN':>{w}}"
                mname = str(row["model（模型名称）"])
                tag   = "  ★" if mname == MAIN_MODEL_NAME else "   "
                print(f"  {mname:<24}  {_g(_MK_F1)}  {_g(_MK_AUC_ROC)}"
                      f"  {_g(_MK_AUC_PR)}  {_g(_MK_REC_DN,8)}"
                      f"  {_g(_MK_REC_ST,9)}  {_g(_MK_REC_UP,8)}"
                      f"  {_g(_MK_REC_MAC,8)}  {_g(_MK_ACC,8)}{tag}")
            print(f"  {SEP}")

    print(f"{'='*140}")


# ════════════════════════════════════════════════════════════════════════
# 结果保存到 Excel
# ════════════════════════════════════════════════════════════════════════
def save_results(
        df_detail:  pd.DataFrame,
        df_summary: pd.DataFrame,
        output_path: str = OUTPUT_PATH,
) -> None:
    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True)

    # ── 预处理路径映射 ────────────────────────────────────────────────
    route_map: Dict[str, str] = {}
    for m in BASELINE_MODEL_NAMES:
        if m in LINEAR_MODELS:
            route_map[m] = "FIX-A: Linear+StandardScaler+PCA(0.95)+SMOTE"
        elif m in NO_SMOTE_MODELS:
            route_map[m] = "FIX-B: Cap→特征选择→共线性（跳过SMOTE）"
        else:
            route_map[m] = "FIX-A: Cap→特征选择→共线性→SMOTE（树模型）"
    route_map[MAIN_MODEL_NAME] = (
        "FIX-A: Cap→特征选择→共线性→SMOTE→CatBoost+Optuna")

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:

        # ── Sheet1：逐窗口逐模型明细 ─────────────────────────────────
        if not df_detail.empty:
            df_detail.to_excel(
                writer, sheet_name="逐窗口逐模型明细", index=False)

        # ── Sheet2：跨窗口均值汇总 ───────────────────────────────────
        if not df_summary.empty:
            df_out = df_summary.copy()
            df_out.insert(
                1, "预处理路径",
                df_out["model（模型名称）"].map(route_map).fillna(""))
            df_out.to_excel(
                writer, sheet_name="所有模型跨窗口均值汇总", index=False)

        # ── Sheet3：各窗口各类别详细指标（逐模型分拆） ──────────────
        if not df_detail.empty:
            df_v = df_detail[
                df_detail["skip_reason（跳过原因）"].fillna("") == ""].copy()
            if not df_v.empty:
                # 选出全量指标列
                detail_cols = (
                    ["window_id（窗口标识）", "model（模型名称）",
                     "test_year（测试年份）", "train_n（训练集样本量）",
                     "test_n（测试集样本量）", "preprocessing_route（预处理路径）"]
                    + ALL_METRIC_COLS
                )
                avail_cols = [c for c in detail_cols if c in df_v.columns]
                df_v[avail_cols].to_excel(
                    writer, sheet_name="逐窗口全量指标明细", index=False)

        # ── Sheet4：配置说明 ─────────────────────────────────────────
        cfg_rows = [
            {"配置项": "脚本版本",          "内容": "v90_pollution_integrated_comparison.py"},
            {"配置项": "整合说明",          "内容": "v89主模型 + v87_v3九基线模型，移除SHAP和CM"},
            {"配置项": "数据文件",          "内容": DATA_PATH},
            {"配置项": "数据Sheet",         "内容": SHEET_NAME},
            {"配置项": "目标列",            "内容": TARGET_COL},
            {"配置项": "BASE_DROP_ROOTS",   "内容": str(sorted(BASE_DROP_ROOTS))},
            {"配置项": "Growth/ROA",        "内容": "已恢复，参与建模"},
            {"配置项": "SHAP可解释性",      "内容": "已移除（整合版本不输出SHAP图）"},
            {"配置项": "混淆矩阵",          "内容": "已移除（整合版本不输出CM图）"},
            {"配置项": "模型总数",          "内容": f"{len(ALL_MODEL_NAMES)}（9基线+1主模型）"},
            {"配置项": "滚动窗口数",        "内容": str(len(FIXED_WINDOWS))},
            {"配置项": "特征选择",          "内容": f"CatBoost初筛Top{FEAT_TOPK} → 共线性过滤|r|>{CORR_THRESHOLD}"},
            {"配置项": "SMOTE",             "内容": "树模型/线性模型均启用；BalancedRF/EasyEnsemble跳过"},
            {"配置项": "Optuna trials",     "内容": f"{OPTUNA_N_TRIALS_MIN}~{OPTUNA_N_TRIALS_MAX}"},
            {"配置项": "N_JOBS",            "内容": str(N_JOBS)},
            {"配置项": "N_JOBS_CATBOOST",   "内容": str(N_JOBS_CATBOOST)},
            {"配置项": "RANDOM_STATE",      "内容": str(RANDOM_STATE)},
        ]
        for win in FIXED_WINDOWS:
            cfg_rows.append({
                "配置项": f"窗口: {win['window_id']}",
                "内容": f"训练={win['train_years']}  验证={win['val_year']}  测试={win['test_year']}"
            })
        for i, (m, rte) in enumerate([
            ("LR",            "FIX-A: Linear+PCA+SMOTE"),
            ("RF",            "FIX-A: Tree+SMOTE"),
            ("BalancedRF",    "FIX-B: NoSMOTE（内置平衡）"),
            ("EasyEnsemble",  "FIX-B: NoSMOTE（内置平衡）"),
            ("XGBoost",       "FIX-A: Tree+SMOTE"),
            ("LightGBM",      "FIX-A: Tree+SMOTE"),
            ("CatBoost",      "FIX-A: Tree+SMOTE"),
            ("LinearSVM",     "FIX-A: Linear+PCA+SMOTE"),
            ("HistGradBoost", "FIX-A: Tree+SMOTE"),
            (MAIN_MODEL_NAME, "FIX-A: Tree+SMOTE+Optuna（主模型）"),
        ]):
            cfg_rows.append({"配置项": f"模型{i+1:02d}: {m}", "内容": rte})

        pd.DataFrame(cfg_rows).to_excel(
            writer, sheet_name="配置说明", index=False)

    print(f"\n  [保存] Excel → {output_path}")
    if not df_detail.empty:
        csv_path = output_path.replace(".xlsx", "_detail.csv")
        df_detail.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"  [保存] CSV   → {csv_path}")


# ════════════════════════════════════════════════════════════════════════
# 主函数
# ════════════════════════════════════════════════════════════════════════
def main():
    t_total = time.time()

    print(f"{'='*72}")
    print(f"  v90_pollution_integrated_comparison.py")
    print(f"  整合版：9 基线模型 + 主模型（CatBoost+Optuna）")
    print(f"  已移除：SHAP 可解释性层 + 混淆矩阵输出")
    print(f"  输出：逐窗口指标明细 + 跨窗口均值汇总")
    print(f"  {datetime.now():%Y-%m-%d %H:%M:%S}  OS={platform.system()}")
    print(f"{'='*72}")

    print(f"\n  BASE_DROP_ROOTS（仅 mean_*，Growth/ROA 已恢复）：")
    print(f"    {sorted(BASE_DROP_ROOTS)}")

    print(f"\n  模型列表（共 {len(ALL_MODEL_NAMES)} 个）：")
    for m in ALL_MODEL_NAMES:
        if m in LINEAR_MODELS:        tag = "[FIX-A 线性]"
        elif m in NO_SMOTE_MODELS:    tag = "[FIX-B 内置平衡]"
        elif m == MAIN_MODEL_NAME:    tag = "[主模型 + Optuna]"
        else:                         tag = "[FIX-A 树模型]"
        print(f"    {m:<26} {tag}")

    print(f"\n  固定滚动窗口（共 {len(FIXED_WINDOWS)} 个）：")
    for w in FIXED_WINDOWS:
        print(f"    {w['window_id']}"
              f"  训练={w['train_years']}"
              f"  验证={w['val_year']}  测试={w['test_year']}")

    print(f"\n  CPU 线程：N_JOBS={N_JOBS}  N_JOBS_CATBOOST={N_JOBS_CATBOOST}")
    print(f"  输出文件：{OUTPUT_PATH}")
    print(f"{'='*72}\n")

    # ── 1. 数据加载 ──────────────────────────────────────────────────
    if not os.path.exists(DATA_PATH):
        print(f"  ERROR: 数据文件未找到: {DATA_PATH}")
        return

    print("  [1/3] 数据加载中...")
    df_raw = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
    for c in CORE_NUM_FEATURES:
        if c in df_raw.columns:
            df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")
    for gas in POLL_ANOMALY_GASES:
        for sfx in ["_zscore", "_exceed"]:
            col = f"{gas}{sfx}"
            if col in df_raw.columns:
                df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")
    print(f"  原始数据: {df_raw.shape}"
          f"  年份: {sorted(df_raw[YEAR_COL].unique())}")

    avail_years = set(df_raw[YEAR_COL].unique())
    all_needed  = set()
    for w in FIXED_WINDOWS:
        all_needed.update(w["train_years"])
        all_needed.add(w["val_year"])
        all_needed.add(w["test_year"])
    missing = all_needed - avail_years
    if missing:
        print(f"  [WARN] 数据中缺失以下年份: {sorted(missing)}")
    else:
        print(f"  [OK] 年份完整覆盖: {sorted(all_needed)}")

    # ── 2. 特征工程 ──────────────────────────────────────────────────
    print(f"\n  [2/3] 特征工程中...")
    df    = feature_engineering(df_raw)
    df    = onehot_industry(df)
    feats = get_feature_columns_fixed(df)
    print(f"  建模特征总数（BASE_DROP 前）: {len(feats)}")

    # ── 3. 实验 ──────────────────────────────────────────────────────
    print(f"\n  [3/3] 实验开始...\n")
    df_detail, df_summary = run_experiment(
        df=df,
        feat_names=feats,
        fixed_windows=FIXED_WINDOWS,
        base_drop_roots=BASE_DROP_ROOTS)

    if df_detail.empty:
        print("  [WARNING] 未产生任何有效结果，请检查数据")
        return

    print_summary(df_detail, df_summary)
    save_results(df_detail, df_summary, OUTPUT_PATH)

    total_min = (time.time() - t_total) / 60
    print(f"\n  总耗时: {total_min:.1f} 分钟")
    print(f"{'='*72}")
    print(f"  指标结果 → {OUTPUT_PATH}")
    print(f"    Sheet1: 逐窗口逐模型明细")
    print(f"    Sheet2: 所有模型跨窗口均值汇总")
    print(f"    Sheet3: 逐窗口全量指标明细（含 Support 等完整指标）")
    print(f"    Sheet4: 配置说明")
    print(f"  CSV备份 → {OUTPUT_PATH.replace('.xlsx', '_detail.csv')}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
