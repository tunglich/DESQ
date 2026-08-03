"""tw50_dflood.py

Stage 2 of the TW-50 pipeline.

Load the best hyperparameters from Stage 1 (tw50_flood.py), retrain the ATT
model with Dynamic Flooding, then emit per-(stock, aspect) prediction CSVs
that Stage 3 (tw50_des.py) consumes.

Dynamic Flooding
----------------
Instead of a fixed flooding_b, `DynamicFloodingCallback` starts from the
best-b found in Stage 1 and adjusts on val_recall plateau:

    - if val_recall stops improving -> lower b (encourage more fit)
    - if val_loss diverges          -> raise b (more regularization)

b is clamped to [0.0, 0.4] per user requirement.

Usage
-----
    python tw50_dflood.py --stock-ids 2330 --aspect fundamental
    python tw50_dflood.py --top50 --aspect all

Outputs
-------
    ./artifacts/dflood/pred/<stock>_<aspect>.csv
        columns: Date, y_true_20, prob_down, prob_up, source
        - source='insample': DES-train slice, predicted by ATT that was also
          trained on this slice (legacy behavior, default).
        - source='oof'     : DES-train slice, predicted by an INNER ATT trained
          only on TRAIN_START..(DES_TRAIN_START - WF_GAP) so predictions are
          strictly out-of-fold. Enabled with `--des-oof`.
        - source='test'    : TEST_START..TEST_END, predicted by the final ATT
          trained on the full TRAIN_START..TRAIN_END window.
    ./artifacts/dflood/models/<stock>_<aspect>.keras
        Retrained model weights (loadable via keras.models.load_model).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers, mixed_precision
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import joblib

# Re-use everything already implemented in tw50_flood.
from tw50_flood import (
    ARTIFACT_ROOT as FLOOD_ARTIFACT_ROOT,
    ASPECTS,
    CausalMask,
    DEFAULT_SEED,
    FLOODING_GRID,
    FloodingModel,
    HYPERBAYES_DIR,
    LOOKBACK_CHOICES,
    REPO_ROOT,
    SinusoidalPositionalEncoding,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    WF_GAP,
    WF_N_SPLITS,
    WF_VAL_RATIO,
    _mha_block,
    _set_global_seed,
    build_windows,
    configure_gpu,
    load_top50_ids,
    parse_aspects,
    parse_stock_ids,
    preprocess_features,
    walk_forward_folds,
)

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')


DFLOOD_ROOT = Path(os.environ.get('DFLOOD_ROOT', REPO_ROOT / 'artifacts' / 'dflood'))
PRED_DIR = DFLOOD_ROOT / 'pred'
MODEL_DIR = DFLOOD_ROOT / 'models'
for _p in (PRED_DIR, MODEL_DIR):
    _p.mkdir(parents=True, exist_ok=True)


# Dynamic-flooding bounds (per requirement).
DFLOOD_MIN_B = 0.0
DFLOOD_MAX_B = 0.4
DFLOOD_STEP = 0.05

# Stage 3 (tw50_des.py) fits KNORA-E on DES_TRAIN_START..TRAIN_END, so we
# emit in-sample train-window predictions from this date in addition to the
# out-of-sample test window (TEST_START..TEST_END).
DES_TRAIN_START = '2020-01-01'


# =============================================================================
# Dynamic Flooding callback
# =============================================================================


class DynamicFloodingCallback(Callback):
    """Adjust `model.flooding_b` on val plateau.

    - No improvement in val_recall for `patience` epochs -> b -= step (relax).
    - val_loss diverges (rises by > `loss_up_ratio`)     -> b += step (regularize).
    - Both actions are clamped to [DFLOOD_MIN_B, DFLOOD_MAX_B].
    """

    def __init__(self, initial_b: float, patience: int = 8,
                 loss_up_ratio: float = 1.15, verbose: int = 0):
        super().__init__()
        self.initial_b = float(initial_b)
        self.patience = int(patience)
        self.loss_up_ratio = float(loss_up_ratio)
        self.verbose = verbose
        self.best_recall = -np.inf
        self.best_val_loss = np.inf
        self.wait = 0
        self.history: list[dict] = []

    def on_train_begin(self, logs=None):
        b = float(np.clip(self.initial_b, DFLOOD_MIN_B, DFLOOD_MAX_B))
        self.model.flooding_b = b
        self.best_recall = -np.inf
        self.best_val_loss = np.inf
        self.wait = 0
        self.history.clear()
        if self.verbose:
            print(f'[DFlood] init b={b:.2f}')

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        recall = float(logs.get('val_recall', 0.0))
        val_loss = float(logs.get('val_loss', np.nan))
        current_b = float(getattr(self.model, 'flooding_b', 0.0))
        new_b = current_b

        # Track best.
        if recall > self.best_recall + 1e-4:
            self.best_recall = recall
            self.wait = 0
        else:
            self.wait += 1

        # Divergence guard.
        if np.isfinite(val_loss) and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
        elif np.isfinite(val_loss) and val_loss > self.best_val_loss * self.loss_up_ratio:
            new_b = min(DFLOOD_MAX_B, current_b + DFLOOD_STEP)

        # Plateau relaxation.
        if self.wait >= self.patience:
            new_b = max(DFLOOD_MIN_B, current_b - DFLOOD_STEP)
            self.wait = 0

        if new_b != current_b:
            self.model.flooding_b = float(new_b)
            if self.verbose:
                print(f'[DFlood] epoch={epoch+1} recall={recall:.4f} '
                      f'val_loss={val_loss:.4f} b: {current_b:.2f} -> {new_b:.2f}')

        self.history.append({'epoch': int(epoch + 1),
                             'val_recall': recall,
                             'val_loss': val_loss,
                             'flooding_b': float(new_b)})


# =============================================================================
# Model construction from best HP
# =============================================================================


def build_att_from_hp(input_shape: tuple[int, int], hp: dict) -> FloodingModel:
    """Instantiate the ATT model using a plain dict of hyperparameters."""
    inputs = keras.Input(shape=input_shape, name='inputs')
    x = SinusoidalPositionalEncoding(name='pos')(inputs)

    n_layers = int(hp.get('attn_layers', 2))
    for i in range(n_layers):
        x = _mha_block(
            x,
            num_heads=int(hp.get(f'attn_heads_{i+1}', 2)),
            key_dim=int(hp.get(f'attn_key_dim_{i+1}', 16)),
            dropout_rate=float(hp.get(f'attn_dropout_{i+1}', 0.1)),
            prefix=f'a{i+1}',
        )

    x = layers.GlobalAveragePooling1D(name='pool')(x)
    x = layers.LayerNormalization(name='pool_ln')(x)
    x = layers.Dense(
        int(hp.get('dense_units', 96)),
        kernel_initializer=hp.get('dense_kernel_1', 'glorot_uniform'),
        activation=hp.get('activation', 'relu'),
        name='dense',
    )(x)
    x = layers.LayerNormalization(name='dense_ln')(x)
    logits = layers.Dense(
        2,
        kernel_initializer=hp.get('dense_kernel_2', 'glorot_uniform'),
        name='logits',
    )(x)
    temp = float(hp.get('temperature', 1.0))
    logits = layers.Lambda(lambda t: t / temp, name='temp')(logits)
    outputs = layers.Activation('softmax', dtype='float32', name='softmax')(logits)

    model = FloodingModel(inputs, outputs)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        metrics=[
            keras.metrics.Recall(class_id=1, name='recall'),
            keras.metrics.AUC(curve='ROC', name='auc'),
            keras.metrics.AUC(curve='PR', name='pr_auc'),
        ],
    )
    return model


# =============================================================================
# Retrain + predict for one (stock, aspect)
# =============================================================================


def load_best_summary(aspect: str, stock_id: str) -> dict:
    fp = HYPERBAYES_DIR / f'ATT_{aspect}_{stock_id}' / 'best_trial_summary.json'
    if not fp.exists():
        raise FileNotFoundError(
            f'Stage-1 summary missing: {fp}. Run tw50_flood.py first.'
        )
    with open(fp, 'r', encoding='utf-8') as fh:
        return json.load(fh)


def _fit_att(
    X_tr: np.ndarray, y_tr_oh: np.ndarray, hp: dict, init_b: float,
    epochs: int, batch_size: int, val_frac: float = 0.2, log_tag: str = '',
) -> tuple[FloodingModel, DynamicFloodingCallback]:
    """Fit one FloodingModel on the given windows using WF last-fold val split.

    val_frac is only used if walk_forward_folds cannot be invoked (e.g. inner
    OOF training with a short window). Otherwise the last WF fold is used
    for validation to stay consistent with Stage 1's evaluation regime.
    """
    n = len(X_tr)
    val_slice = None
    try:
        for tr_idx, va_idx in walk_forward_folds(n):
            val_slice = (tr_idx, va_idx)
    except Exception:
        val_slice = None

    if val_slice is None:
        n_val = max(1, int(n * val_frac))
        tr_idx = np.arange(0, n - n_val)
        va_idx = np.arange(n - n_val, n)
    else:
        tr_idx, va_idx = val_slice

    x_train, y_train = X_tr[tr_idx], y_tr_oh[tr_idx]
    x_val, y_val = X_tr[va_idx], y_tr_oh[va_idx]

    labels_int = np.argmax(y_train, axis=1)
    uniq, counts = np.unique(labels_int, return_counts=True)
    cw = {int(u): float((1.0 / c) * len(labels_int)) for u, c in zip(uniq, counts)}

    model = build_att_from_hp(input_shape=X_tr.shape[1:], hp=hp)
    dflood_cb = DynamicFloodingCallback(initial_b=init_b, patience=8, verbose=1)
    started = time.time()
    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=batch_size, epochs=epochs, verbose=0, shuffle=False,
        class_weight=cw,
        callbacks=[
            dflood_cb,
            ReduceLROnPlateau(monitor='val_recall', mode='max',
                              factor=0.5, patience=10, min_delta=1e-6, verbose=0),
            EarlyStopping(monitor='val_recall', mode='max',
                          patience=25, restore_best_weights=True, verbose=0),
        ],
    )
    tag = f' ({log_tag})' if log_tag else ''
    print(f'[FIT{tag}] final_b={model.flooding_b:.2f}, wall={time.time()-started:.1f}s, '
          f'n_train={len(x_train)}, n_val={len(x_val)}')
    return model, dflood_cb


def _predict_slice(
    model: FloodingModel, processed: pd.DataFrame,
    lookback: int, batch_size: int,
    slice_start: str, slice_end: str, source: str,
) -> pd.DataFrame:
    """Predict on a date slice, using a lookback stitch so the first
    window has enough context. `source` is written into the output CSV."""
    target = processed.loc[slice_start:slice_end]
    if len(target) == 0:
        return pd.DataFrame(columns=['Date', 'y_true_20', 'prob_down', 'prob_up', 'source'])

    if lookback > 1:
        stitch_from = processed.index[processed.index < pd.Timestamp(slice_start)]
        if len(stitch_from) >= lookback - 1:
            stitch_start = stitch_from[-(lookback - 1)]
            combined = processed.loc[stitch_start:slice_end]
        else:
            combined = target
    else:
        combined = target

    X, y, dates = build_windows(combined, lookback=lookback)
    mask = np.asarray((dates >= pd.Timestamp(slice_start)) &
                      (dates <= pd.Timestamp(slice_end)))
    X = X[mask].astype(np.float32)
    y = y[mask]
    dates = dates[mask]
    if len(X) == 0:
        return pd.DataFrame(columns=['Date', 'y_true_20', 'prob_down', 'prob_up', 'source'])

    probs = model.predict(X, batch_size=batch_size, verbose=0)
    return pd.DataFrame({
        'Date': dates,
        'y_true_20': y.astype(int),
        'prob_down': probs[:, 0],
        'prob_up': probs[:, 1],
        'source': source,
    })


def retrain_and_predict(
    aspect: str, stock_id: str, epochs: int, batch_size: int,
    des_oof: bool = False,
) -> pd.DataFrame:
    print(f'\n=== DFLOOD: stock={stock_id}, aspect={aspect}, des_oof={des_oof} ===')
    summary = load_best_summary(aspect, stock_id)
    hp = summary.get('hyperparameters', {})
    lookback = int(summary.get('lookback_window', 20))
    init_b = float(summary.get('flooding_b', 0.10))
    print(f'[HP] lookback={lookback}, init_flooding_b={init_b}, layers={hp.get("attn_layers")}')

    processed = preprocess_features(aspect, stock_id)
    train_df = processed.loc[TRAIN_START:TRAIN_END]
    test_df = processed.loc[TEST_START:TEST_END]

    if len(train_df) < lookback + 100:
        raise RuntimeError(f'Train too short: {len(train_df)} rows.')
    if len(test_df) < lookback:
        raise RuntimeError(f'Test too short: {len(test_df)} rows for lookback={lookback}.')

    frames: list[pd.DataFrame] = []

    # -- OOF DES-train predictions from an INNER ATT model.
    # Inner-train ends before DES_TRAIN_START with an extra WF_GAP-day purge
    # so no 20-day forward label can leak into the DES-train predictions.
    if des_oof:
        inner_cutoff = pd.Timestamp(DES_TRAIN_START) - pd.Timedelta(days=WF_GAP + lookback)
        inner_train_df = train_df.loc[:inner_cutoff.strftime('%Y-%m-%d')]
        if len(inner_train_df) < lookback + 100:
            raise RuntimeError(
                f'Inner-train too short for OOF ({len(inner_train_df)} rows). '
                f'Reduce WF_GAP or DES_TRAIN_START.'
            )
        X_in, y_in, _ = build_windows(inner_train_df, lookback=lookback)
        X_in = X_in.astype(np.float32)
        y_in_oh = to_categorical(y_in, num_classes=2).astype(np.float32)
        inner_model, _ = _fit_att(
            X_in, y_in_oh, hp, init_b, epochs, batch_size, log_tag='inner',
        )
        df_pred_des = _predict_slice(
            inner_model, processed, lookback, batch_size,
            slice_start=DES_TRAIN_START, slice_end=TRAIN_END, source='oof',
        )
        frames.append(df_pred_des)
        del inner_model
        K.clear_session()
        gc.collect()

    # -- Final ATT trained on the full TRAIN_START..TRAIN_END window.
    X_tr, y_tr, dates_tr = build_windows(train_df, lookback=lookback)
    X_tr = X_tr.astype(np.float32)
    y_tr_oh = to_categorical(y_tr, num_classes=2).astype(np.float32)
    final_model, dflood_cb = _fit_att(
        X_tr, y_tr_oh, hp, init_b, epochs, batch_size, log_tag='final',
    )

    # If OOF is off, keep the legacy in-sample DES-train predictions from the
    # final model so tw50_des.py has data to fit KNORA-E on.
    if not des_oof:
        df_pred_des = _predict_slice(
            final_model, processed, lookback, batch_size,
            slice_start=DES_TRAIN_START, slice_end=TRAIN_END, source='insample',
        )
        frames.append(df_pred_des)

    # Test predictions always come from the final model.
    df_pred_te = _predict_slice(
        final_model, processed, lookback, batch_size,
        slice_start=TEST_START, slice_end=TEST_END, source='test',
    )
    frames.append(df_pred_te)

    df_pred = (pd.concat(frames, axis=0, ignore_index=True)
                 .drop_duplicates(subset=['Date'], keep='last')
                 .sort_values('Date')
                 .reset_index(drop=True))
    out_pred = PRED_DIR / f'{stock_id}_{aspect}.csv'
    df_pred.to_csv(out_pred, index=False)
    counts = df_pred['source'].value_counts().to_dict()
    print(f'[SAVE] pred -> {out_pred}  rows_by_source={counts}')

    out_model = MODEL_DIR / f'{stock_id}_{aspect}.keras'
    try:
        final_model.save(out_model)
    except Exception as err:
        print(f'[WARN] could not save model .keras: {err}')

    hist_path = MODEL_DIR / f'{stock_id}_{aspect}_dflood_history.json'
    with open(hist_path, 'w', encoding='utf-8') as fh:
        json.dump({
            'initial_b': init_b,
            'final_b': float(final_model.flooding_b),
            'history': dflood_cb.history,
            'des_oof': bool(des_oof),
        }, fh, indent=2)

    K.clear_session()
    gc.collect()
    return df_pred


# =============================================================================
# CLI
# =============================================================================


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--stock-ids', help='comma-separated list, e.g. 2330,2454')
    p.add_argument('--top50', action='store_true', help='use all stocks from tw50_top50.csv')
    p.add_argument('--aspect', default='all',
                   help=f'comma-separated aspects or "all" (default: all). Valid: {ASPECTS}')
    p.add_argument('--epochs', type=int, default=120)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--des-oof', action='store_true',
                   help='Emit out-of-fold DES-train predictions from an inner ATT '
                        'trained only on TRAIN_START..(DES_TRAIN_START - WF_GAP). '
                        'Recommended for leakage-free Stage-3 KNORA-E fitting.')
    p.add_argument('--seed', type=int, default=DEFAULT_SEED,
                   help='Global RNG seed for PYTHONHASHSEED/random/numpy/tf.')
    args = p.parse_args(argv)

    _set_global_seed(args.seed)
    configure_gpu()
    stock_ids = parse_stock_ids(args.stock_ids, args.top50)
    aspects = parse_aspects(args.aspect)
    print(f'[PLAN] stocks={stock_ids}, aspects={aspects}, epochs={args.epochs}, '
          f'des_oof={args.des_oof}')

    for sid in stock_ids:
        for aspect in aspects:
            try:
                retrain_and_predict(aspect, sid, args.epochs, args.batch_size,
                                    des_oof=args.des_oof)
            except Exception as exc:  # noqa: BLE001
                print(f'[FAIL] {sid}/{aspect}: {exc}')
                import traceback
                traceback.print_exc()
    return 0


if __name__ == '__main__':
    sys.exit(main())
