import sys, json, joblib, gc
from glob import glob
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import multiprocessing as mp
# 從 prediction_update_tony_2026 導入更新函數
import importlib

# 預設關閉 GPU，避免 WSL + multiprocessing 導致 CUDA 初始化異常
os.environ.setdefault('ATT_PREDICT_USE_GPU', '0')
pu = importlib.import_module('prediction_update_tony_2026')


def platform_path(path_str):
    """Normalize Windows paths for current runtime (Windows/WSL)."""
    if os.name == 'nt' or len(path_str) < 2 or path_str[1] != ':':
        return path_str

    drive = path_str[0].lower()
    rest = path_str[2:].replace('\\', '/')
    candidates = [
        f"/mnt/{drive}{rest}",
        f"/mnt/host/{drive}{rest}",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]

# 從 DES 模型資料夾取得股票代碼清單
files = glob(f"{platform_path('D:/DES_model_test')}/DES_*_*.pkl")
tickers = [os.path.basename(x).split('_')[1] for x in files]
tickers = list(set(tickers))  # 去重複
tickers = ['2382','3017'] # '2317', '2454', '6505', '1301', '1303', '2308', '2882', '2881', '2002']  # 測試用


def safe_prediction_update(stock_id):
    """避免單一股票失敗中斷整批更新。"""
    try:
        result = pu._prediction_update(stock_id)
        return {
            'stock': stock_id,
            'ok': True,
            'result': result,
            'error': None
        }
    except Exception as e:
        return {
            'stock': stock_id,
            'ok': False,
            'result': None,
            'error': str(e)
        }

if __name__ == '__main__':

    stock_list = tickers
    print(f"總共 {len(stock_list)} 檔股票待更新")
    use_multiprocess = os.getenv('PREDICTION_USE_MULTIPROCESS', '0') == '1'
    n_done = 1

    if use_multiprocess:
        worker_count = min(len(stock_list), int(os.getenv('PREDICTION_WORKERS', '2')))
        # 使用 spawn 避免 fork 後 TensorFlow/CUDA 初始化異常
        ctx = mp.get_context('spawn')
        with ctx.Pool(worker_count) as pool:
            for result in pool.imap_unordered(safe_prediction_update, stock_list):
                if not result['ok']:
                    print(f"\n[ERROR] {result['stock']}: {result['error']}")
                sys.stdout.write(f'Prediction Progress: {n_done}/{len(stock_list)}\r')
                n_done += 1
    else:
        for stock_id in stock_list:
            result = safe_prediction_update(stock_id)
            if not result['ok']:
                print(f"\n[ERROR] {result['stock']}: {result['error']}")
            sys.stdout.write(f'Prediction Progress: {n_done}/{len(stock_list)}\r')
            n_done += 1

    print()