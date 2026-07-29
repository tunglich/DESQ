from __future__ import annotations

import argparse
import gc
import json
import os
import re
import inspect
from glob import glob
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import keras

from combined_features import (
    _expand_fundamental_features,
    _expand_macro_features,
    _expand_moment_features,
    _expand_tech_trend_features,
)


def _load_module(path: Path, name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).resolve().parent
FEATURE_ROOT = ROOT / "feature"
EXPERIMENT_ROOT = ROOT / "experiment"
SELECTION_ROOT = ROOT / "selection"
SCALER_ROOT = ROOT / "scalar"
DES_MODEL_ROOT = ROOT / "DES_model_US"
DES_PRED_ROOT = ROOT / "model_pred_DES_US"

FAMILIES = ["fundamental", "tech_trend", "moment", "macro"]
TICKER_ALIAS = {"GOOG": "GOOGL"}


def _sanitize_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize feature matrix to finite float64 for sklearn transformers."""
    out = df.copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.apply(pd.to_numeric, errors="coerce")

    # Fill by column median first; if all-NaN column exists, fallback to 0.
    med = out.median(axis=0, numeric_only=True)
    out = out.fillna(med)
    out = out.fillna(0.0)

    arr = np.nan_to_num(out.to_numpy(dtype=np.float64, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    return pd.DataFrame(arr, index=out.index, columns=out.columns)

def _apply_sanitize_statistics(df: pd.DataFrame, stats: dict[str, Any]) -> pd.DataFrame:
    max_abs = float(stats.get("max_abs", 1e6))
    sanitized = df.replace([np.inf, -np.inf], np.nan)
    sanitized = sanitized.apply(pd.to_numeric, errors="coerce").astype(np.float64)

    fill_values = pd.Series(stats.get("fill_values", {}), dtype=np.float64).reindex(sanitized.columns).fillna(0.0)
    lower = pd.Series(stats.get("lower", {}), dtype=np.float64).reindex(sanitized.columns).fillna(-max_abs)
    upper = pd.Series(stats.get("upper", {}), dtype=np.float64).reindex(sanitized.columns).fillna(max_abs)

    sanitized = sanitized.fillna(fill_values)
    sanitized = sanitized.clip(lower=lower, upper=upper, axis=1)
    sanitized = sanitized.clip(lower=-max_abs, upper=max_abs)
    sanitized = np.sign(sanitized) * np.log1p(np.abs(sanitized))

    values = np.nan_to_num(
        sanitized.to_numpy(dtype=np.float64, copy=False),
        nan=0.0,
        posinf=np.log1p(max_abs),
        neginf=-np.log1p(max_abs),
    )
    return pd.DataFrame(values, index=sanitized.index, columns=sanitized.columns)


def _transform_with_pipeline(feature_df: pd.DataFrame, pipeline_obj: Any) -> pd.DataFrame:
    feature_df = _sanitize_numeric_features(feature_df)

    if isinstance(pipeline_obj, dict) and "scaler" in pipeline_obj:
        scaler = pipeline_obj["scaler"]
        sanitize_stats = pipeline_obj.get("sanitize_stats")
        transformed_input = feature_df
        if sanitize_stats is not None:
            transformed_input = _apply_sanitize_statistics(feature_df, sanitize_stats)
        if scaler is None:
            transformed = transformed_input.to_numpy(dtype=np.float64, copy=True)
            transformed = np.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            transformed = scaler.transform(transformed_input)
    else:
        transformed = pipeline_obj.transform(feature_df)

    return pd.DataFrame(transformed, index=feature_df.index, columns=feature_df.columns)


def _sequence_to_windows(seq: pd.DataFrame, y: pd.Series, n_steps: int):
    xs, ys = [], []
    for i in range(seq.shape[0] - n_steps + 1):
        end_ix = i + n_steps
        xs.append(np.array(seq.iloc[i:end_ix]))
        ys.append(np.array(y.iloc[end_ix - 1]))
    return np.array(xs), np.array(ys)


def _get_windows(X: np.ndarray, y: np.ndarray, slc: slice, steps: int):
    min_idx = 250 - 1
    start_idx, end_idx, _ = slc.start, slc.stop, slc.step
    start_idx = max(min_idx, start_idx - steps + 1)
    end_idx = end_idx - steps + 1
    return X[start_idx:end_idx], y[start_idx:end_idx]


@tf.keras.utils.register_keras_serializable()
class SinusoidalPositionalEncoding(tf.keras.layers.Layer):
    def call(self, x):
        seq_len = tf.shape(x)[1]
        d_model = tf.shape(x)[2]
        d_model_float = tf.cast(d_model, tf.float32)
        positions = tf.cast(tf.range(seq_len), tf.float32)[:, tf.newaxis]
        dims = tf.cast(tf.range(d_model), tf.float32)[tf.newaxis, :]
        angles = positions / tf.pow(10000.0, 2.0 * tf.math.floor(dims / 2.0) / d_model_float)
        sin_mask = tf.math.mod(tf.range(d_model), 2)
        cos_mask = 1 - sin_mask
        encoding = tf.sin(angles) * tf.cast(cos_mask, tf.float32) + tf.cos(angles) * tf.cast(sin_mask, tf.float32)
        encoding = encoding[tf.newaxis, :, :]
        encoding = tf.cast(encoding, x.dtype)
        return x + encoding


@tf.keras.utils.register_keras_serializable()
class TemperatureScaling(tf.keras.layers.Layer):
    def __init__(self, temp=1.0, **kwargs):
        super().__init__(**kwargs)
        self.temp = float(temp)

    def call(self, x):
        return x / tf.cast(self.temp, x.dtype)

    def get_config(self):
        cfg = super().get_config()
        cfg["temp"] = self.temp
        return cfg


@tf.keras.utils.register_keras_serializable()
class FloodingModel(tf.keras.Model):
    """Fallback class for loading serialized ATT models at inference time."""
    flooding_b = 0.10


@tf.keras.utils.register_keras_serializable()
class DynamicCausalMask(tf.keras.layers.Layer):
    """Fallback causal mask layer used by ATT models."""

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return tf.linalg.band_part(tf.ones((1, seq_len, seq_len), dtype=tf.float32), -1, 0)

    def get_config(self):
        return super().get_config()


@tf.keras.utils.register_keras_serializable()
class CompatMultiHeadAttention(tf.keras.layers.MultiHeadAttention):
    """Compatibility wrapper that drops unknown config keys across Keras versions."""

    @classmethod
    def from_config(cls, config):
        cfg = dict(config)
        allowed = set(inspect.signature(tf.keras.layers.MultiHeadAttention.__init__).parameters.keys())
        allowed.discard("self")
        filtered = {k: v for k, v in cfg.items() if k in allowed}
        return cls(**filtered)


def _load_model_with_fallback(path: str):
    _enable_generic_layer_config_compat()
    _enable_mha_config_compat()
    _enable_lambda_marshal_compat()
    _enable_lambda_shape_runtime_compat()
    custom_objects = {
        "FloodingModel": FloodingModel,
        "DynamicCausalMask": DynamicCausalMask,
        "MultiHeadAttention": CompatMultiHeadAttention,
        "SinusoidalPositionalEncoding": SinusoidalPositionalEncoding,
        "TemperatureScaling": TemperatureScaling,
    }
    custom_objects = {k: v for k, v in custom_objects.items() if v is not None}
    return tf.keras.models.load_model(path, custom_objects=custom_objects, compile=False, safe_mode=False)


_LAMBDA_COMPAT_ENABLED = False
_MHA_COMPAT_ENABLED = False
_LAYER_COMPAT_ENABLED = False
_LAMBDA_SHAPE_RUNTIME_COMPAT_ENABLED = False


def _enable_generic_layer_config_compat() -> None:
    """Patch base Layer.from_config to ignore unknown kwargs from newer Keras exports."""
    global _LAYER_COMPAT_ENABLED
    if _LAYER_COMPAT_ENABLED:
        return

    base_cls = tf.keras.layers.Layer
    orig_from_config = base_cls.from_config

    @classmethod
    def _from_config_compat(cls, config):
        cfg = dict(config)
        allowed = set(inspect.signature(cls.__init__).parameters.keys())
        allowed.discard("self")
        filtered = {k: v for k, v in cfg.items() if k in allowed}
        try:
            return cls(**filtered)
        except Exception:
            return orig_from_config.__func__(cls, config)  # type: ignore[attr-defined]

    base_cls.from_config = _from_config_compat  # type: ignore[assignment]
    _LAYER_COMPAT_ENABLED = True


def _enable_mha_config_compat() -> None:
    """Patch built-in MHA deserializer to ignore cross-version kwargs (e.g. use_gate)."""
    global _MHA_COMPAT_ENABLED
    if _MHA_COMPAT_ENABLED:
        return

    layer_cls = tf.keras.layers.MultiHeadAttention
    orig_from_config = layer_cls.from_config
    allowed = set(inspect.signature(layer_cls.__init__).parameters.keys())
    allowed.discard("self")

    @classmethod
    def _from_config_compat(cls, config):
        cfg = dict(config)
        filtered = {k: v for k, v in cfg.items() if k in allowed}
        return orig_from_config.__func__(cls, filtered)  # type: ignore[attr-defined]

    layer_cls.from_config = _from_config_compat  # type: ignore[assignment]
    _MHA_COMPAT_ENABLED = True


def _enable_lambda_marshal_compat() -> None:
    """Monkey-patch keras lambda loader for cross-Python marshal incompatibility.

    Some legacy `.keras` files contain Lambda bytecode serialized under a different
    Python minor version, triggering `ValueError: bad marshal data` during load.
    We intercept that path and provide deterministic fallbacks for known lambda
    patterns used by ATT models:
      1) positional encoding add: `x + pe`
      2) temperature scaling: `x / temp`
    """
    global _LAMBDA_COMPAT_ENABLED
    if _LAMBDA_COMPAT_ENABLED:
        return

    try:
        from keras.src.utils import python_utils as _py_utils  # type: ignore
    except Exception:
        try:
            from keras.utils import python_utils as _py_utils  # type: ignore
        except Exception:
            return

    orig_func_load = _py_utils.func_load

    def _normalize_lambda_closure_obj(obj):
        if isinstance(obj, dict):
            if obj.get("class_name") == "__tensor__":
                cfg = obj.get("config", {}) if isinstance(obj.get("config", {}), dict) else {}
                vals = cfg.get("value", None)
                dtype_name = str(cfg.get("dtype", "float32"))
                dtype = getattr(tf, dtype_name, tf.float32)
                return tf.convert_to_tensor(vals, dtype=dtype)
            return {k: _normalize_lambda_closure_obj(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_normalize_lambda_closure_obj(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(_normalize_lambda_closure_obj(v) for v in obj)
        return obj

    def _compat_func_load(code, defaults=None, closure=None, globs=None):
        merged_globs = dict(globs or {})
        merged_globs.setdefault("tf", tf)
        merged_globs.setdefault("np", np)
        merged_globs.setdefault("keras", keras)
        normalized_closure = _normalize_lambda_closure_obj(closure)

        def _normalize_func_defaults(fn):
            try:
                if getattr(fn, "__defaults__", None):
                    fn.__defaults__ = tuple(_normalize_lambda_closure_obj(v) for v in fn.__defaults__)
                if getattr(fn, "__kwdefaults__", None):
                    fn.__kwdefaults__ = {
                        k: _normalize_lambda_closure_obj(v)
                        for k, v in fn.__kwdefaults__.items()
                    }
            except Exception:
                pass
            return fn

        try:
            fn = orig_func_load(code, defaults=defaults, closure=normalized_closure, globs=merged_globs)
            return _normalize_func_defaults(fn)
        except Exception as exc:
            if "bad marshal data" not in str(exc):
                raise

            payload = None
            if isinstance(normalized_closure, (list, tuple)) and len(normalized_closure) > 0:
                payload = normalized_closure[0]

            # Lambda #1: x + positional encoding tensor
            if payload is not None and isinstance(payload, (tf.Tensor, np.ndarray, list, tuple)):
                pe = tf.convert_to_tensor(payload, dtype=tf.float32)

                def _fn_add(x, pe=pe):
                    return tf.cast(x, pe.dtype) + pe

                return _normalize_func_defaults(_fn_add)

            # Lambda #2: x / temp (closure is hp dict with "temp")
            if isinstance(payload, dict) and "temp" in payload:
                temp = float(payload.get("temp", 1.0))
                if temp == 0:
                    temp = 1.0

                def _fn_temp(x, temp=temp):
                    return x / tf.cast(temp, x.dtype)

                return _normalize_func_defaults(_fn_temp)

            # Conservative fallback for unknown lambdas.
            def _fn_identity(x):
                return x

            return _normalize_func_defaults(_fn_identity)

    _py_utils.func_load = _compat_func_load
    _LAMBDA_COMPAT_ENABLED = True


def _enable_lambda_shape_runtime_compat() -> None:
    """Patch Lambda shape inference to fallback to identity shape on failure.

    Some legacy lambda payloads in ATT models fail automatic output-shape
    inference in newer Keras. For our known lambdas (positional add/temp scale),
    output shape equals input shape.
    """
    global _LAMBDA_SHAPE_RUNTIME_COMPAT_ENABLED
    if _LAMBDA_SHAPE_RUNTIME_COMPAT_ENABLED:
        return

    layer_cls = tf.keras.layers.Lambda

    if hasattr(layer_cls, "compute_output_shape"):
        orig_compute_output_shape = layer_cls.compute_output_shape

        def _compute_output_shape_compat(self, input_shape):
            try:
                return orig_compute_output_shape(self, input_shape)
            except Exception as exc:
                msg = str(exc).lower()
                if "could not automatically infer" in msg or "output_shape" in msg:
                    return input_shape
                raise

        layer_cls.compute_output_shape = _compute_output_shape_compat  # type: ignore[assignment]

    if hasattr(layer_cls, "compute_output_spec"):
        orig_compute_output_spec = layer_cls.compute_output_spec

        def _compute_output_spec_compat(self, *args, **kwargs):
            try:
                return orig_compute_output_spec(self, *args, **kwargs)
            except Exception as exc:
                msg = str(exc).lower()
                if "could not automatically infer" in msg or "output_shape" in msg:
                    if args:
                        return args[0]
                raise

        layer_cls.compute_output_spec = _compute_output_spec_compat  # type: ignore[assignment]

    _LAMBDA_SHAPE_RUNTIME_COMPAT_ENABLED = True


def _extract_positive_class_prob(pred: np.ndarray, classes: np.ndarray | None = None) -> np.ndarray:
    """Return positive-class probability from model outputs with robust shape handling."""
    arr = np.asarray(pred)
    if arr.ndim == 1:
        return arr.astype(np.float64)

    if arr.ndim >= 2:
        if arr.shape[1] >= 2:
            return arr[:, 1].astype(np.float64)
        if arr.shape[1] == 1:
            col = arr[:, 0].astype(np.float64)
            if classes is None or len(classes) == 0:
                return col
            # Single-column predict_proba implies one observed class.
            cls0 = classes[0]
            try:
                cls0_int = int(cls0)
            except Exception:
                cls0_int = 1 if str(cls0) == "1" else 0
            return col if cls0_int == 1 else np.zeros_like(col, dtype=np.float64)

    raise ValueError(f"Unexpected prediction shape: {arr.shape}")


def _read_selection(ticker: str, family: str) -> list[str]:
    p = SELECTION_ROOT / f"{ticker}.json"
    if not p.exists() and ticker in TICKER_ALIAS:
        p = SELECTION_ROOT / f"{TICKER_ALIAS[ticker]}.json"
    if not p.exists():
        raise FileNotFoundError(f"selection not found for {ticker}")
    obj = json.loads(p.read_text(encoding="utf-8"))
    if family not in obj:
        raise KeyError(f"{ticker}: family {family} not found in selection json")
    return obj[family]


def _load_feature_df(ticker: str, family: str) -> pd.DataFrame:
    data_ticker = ticker
    p = FEATURE_ROOT / f"{family}_{ticker}.csv"
    if not p.exists() and ticker in TICKER_ALIAS:
        data_ticker = TICKER_ALIAS[ticker]
        p = FEATURE_ROOT / f"{family}_{data_ticker}.csv"
    if not p.exists():
        raise FileNotFoundError(f"feature csv not found: {p}")

    df = pd.read_csv(p, index_col=0, parse_dates=True)
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")]

    if family == "macro":
        df = _expand_macro_features(df)
    elif family == "tech_trend":
        df = _expand_tech_trend_features(df)
    elif family == "fundamental":
        df = _expand_fundamental_features(df)
    elif family == "moment":
        df = _expand_moment_features(df)

    # last 4 are y_10,y_20,y_40,y_60 by convention
    labels = df.columns[-4:].tolist()
    fs = _read_selection(ticker, family)
    for col in fs:
        if col not in df.columns:
            df[col] = 0.0
    return df[fs + labels], data_ticker


def _infer_n_steps_from_model(path: str) -> int:
    m = _load_model_with_fallback(path)
    try:
        n_steps = int(m.input_shape[1])
    finally:
        del m
        gc.collect()
        tf.keras.backend.clear_session()
    return n_steps


def _update_att_family_predictions(
    ticker: str,
    family: str,
    start: str | None,
    end: str | None,
) -> pd.Series:
    exp_dir = EXPERIMENT_ROOT / f"ATT_{family}_{ticker}"
    if not exp_dir.exists() and ticker in TICKER_ALIAS:
        exp_dir = EXPERIMENT_ROOT / f"ATT_{family}_{TICKER_ALIAS[ticker]}"
    model_files = sorted(glob(str(exp_dir / "experiment_*.keras")))
    if not model_files:
        raise FileNotFoundError(f"{ticker}/{family}: no experiment_*.keras")

    data, _ = _load_feature_df(ticker, family)
    pipeline_path = SCALER_ROOT / f"scaler_{family}_{ticker}.pkl"
    if not pipeline_path.exists() and ticker in TICKER_ALIAS:
        pipeline_path = SCALER_ROOT / f"scaler_{family}_{TICKER_ALIAS[ticker]}.pkl"
    if not pipeline_path.exists():
        raise FileNotFoundError(f"{ticker}/{family}: scaler missing")
    pipeline = joblib.load(pipeline_path)

    transformed_features = _transform_with_pipeline(data.iloc[:, :-4], pipeline)
    data = pd.concat([transformed_features, data.iloc[:, -4:]], axis=1)

    family_preds = []
    for model_path in model_files:
        csv_path = model_path.replace("experiment_", "experiment_result_").replace(".keras", ".csv")
        old = pd.Series(dtype=np.float64)
        if os.path.exists(csv_path):
            old = pd.read_csv(csv_path, header=None, index_col=0, parse_dates=True).squeeze("columns")
            old.index = pd.to_datetime(old.index)

        n_steps = _infer_n_steps_from_model(model_path)
        X, y = _sequence_to_windows(data.iloc[:, :-4], data["y_20"], n_steps)
        if len(X) == 0:
            continue

        if end is None:
            end_ts = data.index.max()
        else:
            end_ts = min(pd.Timestamp(end), data.index.max())

        if start is not None:
            start_ts = pd.Timestamp(start)
        elif not old.empty:
            # Recompute a small overlap window to ensure enough lookback for new dates.
            start_ts = old.index.max() - pd.Timedelta(days=120)
        else:
            start_ts = data.index.min()

        # Guard: when caller passes a start date later than the last available
        # feature date (e.g. daily orchestrator runs before US market close on
        # today's session, so features only go up to yesterday), the requested
        # window becomes empty and we would silently keep stale ATT logits.
        # Fall back to a rolling tail-refresh so new dates in `data` still get
        # scored.
        if start_ts > end_ts:
            if not old.empty:
                start_ts = min(old.index.max(), end_ts) - pd.Timedelta(days=120)
            else:
                start_ts = data.index.min()

        slc = data.index.slice_indexer(start=start_ts, end=end_ts)
        X_test, _ = _get_windows(X, y, slc, n_steps)
        if len(X_test) == 0:
            family_preds.append(old)
            continue

        # Prediction dates align with the *end* of each rolling window inside
        # `slc`. Window k (in X_test) ends at data.index position
        # `adj_end + n_steps - 1` where `adj_end = slc.stop - n_steps + 1`, so
        # the last window ends exactly at `data.index[slc.stop - 1]`. Using
        # `data.index[-len(X_test):]` is WRONG when data.index extends past
        # end_ts (e.g. features already have today but caller asked for
        # yesterday) — it silently misaligns predictions by shifting them
        # onto later dates and leaves the intended dates empty.
        pred_dates = data.index[slc.stop - len(X_test):slc.stop]

        model = _load_model_with_fallback(model_path)
        try:
            raw_pred = model.predict(X_test, verbose=0)
            pos_prob = _extract_positive_class_prob(raw_pred)
            new_vals = pd.Series(pos_prob, index=pred_dates)
        finally:
            del model
            gc.collect()
            tf.keras.backend.clear_session()

        merged = pd.concat([old, new_vals])
        # keep="last" so a fresh prediction (e.g. after retrain / bug fix)
        # overwrites any cached value on the same date instead of being ignored.
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        merged.to_csv(csv_path, header=False, encoding="utf-8")
        family_preds.append(merged)

    if not family_preds:
        raise RuntimeError(f"{ticker}/{family}: no family predictions")

    out = pd.concat(family_preds, axis=1).mean(axis=1)
    out.index = pd.to_datetime(out.index)
    return out.sort_index()


def _update_des_prediction(ticker: str, logits_df: pd.DataFrame) -> dict[str, Any]:
    model_candidates = sorted(glob(str(DES_MODEL_ROOT / f"DES_{ticker}_*.pkl")))
    if not model_candidates:
        raise FileNotFoundError(f"{ticker}: no DES model")
    model_path = model_candidates[-1]
    des_model = joblib.load(model_path)

    pred_candidates = sorted(glob(str(DES_PRED_ROOT / f"DES_pred_{ticker}_*.csv")))
    if pred_candidates:
        pred_path = pred_candidates[-1]
        old = pd.read_csv(pred_path, index_col=0, parse_dates=True).squeeze("columns")
    else:
        pred_path = str(DES_PRED_ROOT / f"DES_pred_{ticker}_2019-12-31.csv")
        old = pd.Series(dtype=np.float64)

    raw_proba = des_model.predict_proba(logits_df)
    classes = getattr(des_model, "classes_", None)
    new = pd.Series(_extract_positive_class_prob(raw_proba, classes=classes), index=logits_df.index)
    merged = pd.concat([old, new])
    merged.index = pd.to_datetime(merged.index)
    # keep="last": fresh DES prediction wins on duplicate dates so re-runs
    # after model updates or bug fixes actually take effect.
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    merged.to_csv(pred_path, header=True, encoding="utf-8")

    old_last = old.index.max() if not old.empty else None
    added = int((merged.index > old_last).sum()) if old_last is not None else int(len(merged))
    return {
        "pred_path": pred_path,
        "rows": int(len(merged)),
        "new_rows": added,
        "last_date": str(merged.index.max().date()) if len(merged) else None,
    }


def update_ticker(ticker: str, start: str | None, end: str | None) -> dict[str, Any]:
    family_series: dict[str, pd.Series] = {}
    result: dict[str, Any] = {"ticker": ticker, "status": "ok", "families": {}, "des": {}}

    for family in FAMILIES:
        s = _update_att_family_predictions(ticker, family, start=start, end=end)
        family_series[family] = s
        result["families"][family] = {
            "rows": int(len(s)),
            "last_date": str(s.index.max().date()) if len(s) else None,
        }

    X_all = pd.concat([family_series[f] for f in FAMILIES if f in family_series], axis=1)
    X_all.columns = [f for f in FAMILIES if f in family_series]
    X_all = X_all.sort_index().ffill().bfill().fillna(0.5).astype("float64")

    result["des"] = _update_des_prediction(ticker, X_all)
    return result


def discover_default_tickers() -> list[str]:
    tickers = set()
    for p in DES_MODEL_ROOT.glob("DES_*_*.pkl"):
        m = re.match(r"^DES_(.+)_\d{4}-\d{2}-\d{2}\.pkl$", p.name)
        if m:
            tickers.add(m.group(1))
    return sorted(tickers)


def update_predictions(tickers: list[str], start: str | None, end: str | None, strict: bool = False):
    out = {"ok": [], "skipped": [], "failed": []}
    for tk in tickers:
        try:
            rec = update_ticker(tk, start=start, end=end)
            out["ok"].append(rec)
            print(f"[OK] {tk}: +{rec['des']['new_rows']} rows -> {rec['des']['pred_path']}")
        except FileNotFoundError as exc:
            rec = {"ticker": tk, "status": "skipped", "reason": str(exc)}
            out["skipped"].append(rec)
            print(f"[SKIP] {tk}: {exc}")
            if strict:
                raise
        except Exception as exc:
            rec = {"ticker": tk, "status": "failed", "reason": str(exc)}
            out["failed"].append(rec)
            print(f"[FAIL] {tk}: {exc}")
            if strict:
                raise
    return out


def _parse_tickers(raw: str | None) -> list[str]:
    if not raw:
        return discover_default_tickers()
    parts = [x.strip().upper() for x in raw.split(",") if x.strip()]
    return sorted(set(parts))


def main():
    parser = argparse.ArgumentParser(description="Update US ATT logits and DES predictions")
    parser.add_argument("--tickers", type=str, default=None, help="comma-separated tickers; default=discover from DES_model_US")
    parser.add_argument("--start", type=str, default=None, help="start date (YYYY-MM-DD) for prediction refresh")
    parser.add_argument("--end", type=str, default=None, help="end date (YYYY-MM-DD) for prediction refresh")
    parser.add_argument("--strict", action="store_true", help="fail-fast on first error")
    parser.add_argument("--summary", type=str, default=None, help="optional summary json path")
    args = parser.parse_args()

    tickers = _parse_tickers(args.tickers)
    if not tickers:
        raise SystemExit("No tickers to process")

    res = update_predictions(tickers=tickers, start=args.start, end=args.end, strict=args.strict)
    print(f"[DONE] ok={len(res['ok'])} skipped={len(res['skipped'])} failed={len(res['failed'])}")

    if args.summary:
        out_path = Path(args.summary)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
