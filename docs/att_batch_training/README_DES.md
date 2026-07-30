# DES_update_ATT-sentiment.py 執行指南

## 概述

此程式使用 **Dynamic Ensemble Selection (DES)** 方法，結合 6 個面向的 ATT 模型預測結果與 CUSUM 統計量，產生股票買賣信號並進行回測與績效繪圖。

### 流程

```
ATT 模型預測 → 合併 6 面向特徵 → RF 調參 → KNORAE 集成 → CUSUM 過濾 → 回測交易 → 績效報告
```

---

## 環境需求

| 項目 | 版本 |
|------|------|
| Python | 3.11.x |
| Conda env | `finlab` |
| TensorFlow | 2.21.0 |
| scikit-learn | 1.7.x |
| DESlib | 0.3.7 |
| pandas | 3.0.x |
| matplotlib | (TkAgg 後端) |
| joblib | 1.5.x |

### 安裝依賴

```bash
conda activate finlab
pip install pandas numpy scikit-learn deslib matplotlib joblib
```

---

## 前置資料準備

執行前須確認以下資料已存在於 `D:/` 磁碟：

| 路徑 | 說明 | 來源 |
|------|------|------|
| `D:/experiments_df_test/ATT_{aspect}_{stock_id}/experiment_result_*.csv` | 各面向 ATT 模型預測結果 | `ATT+Dflooding.py` 訓練產出 |
| `D:/Feature_new/fundamental_{stock_id}.csv` | 基本面特徵（含 y_20 標籤） | `Feature_Cmoney_update.py` |
| `D:/CmoneyFactor/Open.csv`, `Close.csv`, `High.csv`, `Low.csv`, `Volume.csv` | CMoney 股價資料 | `Feature_Cmoney_update.py` |
| `D:/CmoneyFactor/Stock_name.csv` | 股票名稱對照表 | CMoney |
| `./cumSum/cusum_{stock_id}.csv` | CUSUM 統計量 | `CUMSUM_feature_finlab.py` |
| `./cumSum_prob_6/cumsum_prob_{stock_id}.csv` | CUSUM 機率序列 | `CUSUM_prob_multi_finlab.py` |

### 輸出資料

| 路徑 | 說明 |
|------|------|
| `D:/DES_model_test/DES_{stock_id}_{period}.pkl` | 訓練好的 DES 模型 |
| `D:/RF_model_test/RF_{stock_id}_{period}.pkl` | 訓練好的 RF 基礎分類器 |
| `D:/model_pred_DES_test/DES_pred_{stock_id}_{period}.csv` | DES 預測結果 |
| `D:/model_pred_RF_test/RF_pred_{stock_id}_{period}.csv` | RF 預測結果 |
| `D:/model_output/ensemble_{stock_id}.png` | 信號綜覽圖（10 子圖） |
| `./evaluation/backtest_{stock_id}_L1S1.png` | 回測績效圖 |

---

## 執行方式

> **單一標的預估時間**：DES 集成（RF 調參 + KNORAE + CUSUM 過濾 + 回測）約 **5 分鐘**（RTX 5090 / 5080 差不多）。前置的 ATT 兩階段訓練另計：Phase 1 超參數搜尋 ≈ 3 小時、Phase 2 Dflooding ≈ 2 小時（見 `README_Batch_training.md`）。

### 互動模式（預設）

程式啟動後會透過 `input()` 要求輸入股票代號：

```bash
conda activate finlab
cd c:\Users\tungl\finlab\workspace_vscoding
python DES_update_ATT-sentiment.py

cd /mnt/c/Users/tungl/finlab/workspace_vscoding
python DES_update_ATT-sentiment.py
```

```
Please input stock_id: 2330
```

### 注意事項

1. **需要在有圖形介面的環境執行**，程式使用 `matplotlib TkAgg` 後端彈出視窗顯示圖形
2. **WSL 使用者**需設定 X11 forwarding 或使用 `show_fig = False`（修改程式第 476 行）
3. 若模型和預測結果已存在（`.pkl` / `.csv` 檔），程式會直接載入不重新訓練

---

## 全域參數

程式中可調整的關鍵參數（位於主程式區塊）：

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `train_start` | `'2007-08-01'` | 訓練資料起始日期 |
| `train_end` | `'2024-06-30'` | 訓練資料結束日期 |
| `test_start` | `'2024-07-01'` | 測試資料起始日期 |
| `period` | `['2019-12-31']` | 模型訓練時間節點（可加入多個做滾動更新） |
| `long` | `1` | 買入信號需連續看多天數 |
| `short` | `1` | 賣出信號需連續看空天數 |
| `threshold` | `0.50` | 機率 > threshold 視為看多 |
| `span` | `1` | EWM 平滑跨度（1 = 不平滑） |
| `show_fig` | `True` | 是否彈出圖形視窗 |
| `save_fig` | `True` | 是否存檔圖形 |

---

## 完整 Pipeline 執行順序

本程式是最後一步（集成 + 回測），完整流程如下：

```
1. Feature_Cmoney_update.py        → 更新 CMoney 股價/特徵資料
2. CUMSUM_feature_finlab.py         → 計算 CUSUM 統計量
3. CUSUM_prob_multi_finlab.py       → 計算 CUSUM 機率序列
4. ATT+Dflooding.py                 → 訓練 ATT 模型（AutoML + Flooding）
5. prediction_ATT_update.py         → 批次更新 ATT 模型預測結果
6. DES_update_ATT-sentiment.py      → DES 集成 + CUSUM 過濾 + 回測 ← 本程式
```

---

## 輸出範例

執行完成後會輸出績效報告：

```
交易次數:  12
獲利次數:  8
勝率:           0.67
總獲利:  15234567.89
平均獲利: 1904320.99
總損失:  -3456789.12
平均損失: -864197.28
盈虧比:         2.20
```

同時產生：
- **信號綜覽圖**（10 子圖）：股價、DES 原始/平滑/混合信號、6 面向個別信號
- **回測績效圖**：模型累積報酬 vs 股票 Buy & Hold，標註買賣點與模型更新點
