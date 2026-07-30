# Batch_training Agent

自動化批次訓練代理，負責依序對多檔標的執行 ATT+Flood（AutoML 搜參）與 ATT+Dflooding（固定參數重複訓練），並具備 **即時 epoch 監控**與 **WSL 自動重啟** 能力。

---

## 架構

```
Windows                           WSL (Ubuntu / finlab env)
┌──────────────────┐              ┌──────────────────────────┐
│ Batch_training   │─── wsl ────▶│ Batch_training.sh        │
│ .bat             │              │  ├─ ATT+Flood.py   (P1)  │
│                  │◀── exit 42 ──│  ├─ ATT+Dflooding.py(P2) │
│ wsl --terminate  │              │  └─ monitor_log()        │
│ wait 10s         │              └──────────────────────────┘
│ re-launch ───────│─── wsl ────▶ (resume from state file)
└──────────────────┘
```

**為何分兩層？**  
`wsl --terminate` 會終止整個 WSL instance，包含監控腳本本身。因此外層 `.bat` 在 Windows 端負責偵測 exit code 42、重啟 WSL 並重新呼叫 `.sh`。

---

## 快速開始

### 1. 在 Windows CMD / PowerShell 執行

```bat
Batch_training.bat 3293
```

多檔標的：

```bat
Batch_training.bat 3293,2330,2317
```

指定 epoch 逾時秒數（預設 10 秒）：

```bat
Batch_training.bat 3293 --epoch-timeout 15
```

僅訓練特定因子：

```bat
Batch_training.bat 3293 --model-types fundamental,trade,moment
```

指定驗證模式（整批統一）：

```bat
Batch_training.bat 3293 --validation-mode traditional
Batch_training.bat 3293 --validation-mode walk_forward_expanding
Batch_training.bat 3293 --validation-mode walk_forward_rolling
```

> `traditional` 是 `blocking` 的同義字（舊版切分方式）。

若未指定 `--validation-mode` 且是新批次，`Batch_training.sh` 會互動詢問：
- `1) traditional`
- `2) walk-forward expanding`
- `3) walk-forward rolling`（預設）

**整批特徵前處理決定**（只問一次，由所有子任務繼承）：

```bat
Batch_training.bat 2317,2301,3231 --no-preprocess   # 跳過特徵前處理
Batch_training.bat 2317,2301,3231 --preprocess      # 強制啟用特徵前處理
Batch_training.bat 2317,2301,3231                    # 互動問一次（Y/n）
```

> `ATT+Flood.py` / `ATT+Dflooding.py` 原本會針對每個 `(stock, model_type)` 問一次「是否執行特徵前處理」，Agent 改為**整批問一次**，決定寫入 `.batch_training_state` 並透過環境變數 `FEATURE_PREPROCESS` 傳給所有子程序，WSL 重啟後也保留該決定。

### 2. 直接在 WSL 內執行（不需要自動 WSL 重啟）

```bash
bash Batch_training.sh 3293
bash Batch_training.sh 3293,2330 --epoch-timeout 12
```

### 3. 清除狀態檔重新開始

```bat
Batch_training.bat --reset
```

> ⚠️ **狀態檔優先規則 (重要)**  
> 若 `.batch_training_state` 已存在，`Batch_training.sh` 會 **以狀態檔為準**，命令列傳入的 `stock_ids`、`--no-preprocess/--preprocess` 與 `--validation-mode` 都會被忽略（只會印 `[WARN]`），也不會再詢問特徵前處理與 validation mode。  
> **常見徵兆**：你下了新的 stock_ids（例如 `8299,5347`）卻看到訓練的是舊的標的（例如 `2330`），且沒被問是否做特徵前處理。  
>
> **Agent 處理流程**：當使用者下批次訓練指令時，若偵測到 `.batch_training_state` 中的 `stock_ids` 與本次參數不同，**必須先停下並詢問使用者**：
> 1. 是否要中止先前未完成的批次（執行 `--reset` 後重跑新標的）？
> 2. 或繼續先前未完成的批次（忽略本次參數）？
>
> 確認後再執行對應動作，不可直接覆寫或靜默沿用舊狀態。

---

## 執行流程

> **單一標的預估時間（RTX 5090 / 5080 實測，兩者差不多）**  
> Phase 1 超參數搜尋 ≈ 3 小時，Phase 2 Dflooding 正式訓練 ≈ 2 小時，後續 DES 集成 ≈ 5 分鐘。  
> 一支股票 6 個面向全跑完約 5 小時（不含 DES）。

1. **Phase 1 — AutoML（ATT+Flood.py）**  
   對每個 `(stock_id, model_type)` 組合執行 Bayesian Optimization 超參數搜尋（stage1: 12 trials / 80 epochs → stage2: 24 trials / 120 epochs）。

2. **Phase 2 — Dynamic Flooding（ATT+Dflooding.py）**  
   讀取 Phase 1 產出的最佳超參數，以固定設定重複訓練 18 次，保留 top 3 模型。

3. **Epoch 監控**  
   後台 `tail -f` log 檔，即時解析每個 epoch 耗時。若連續 2 個 epoch 超過閾值（預設 10 秒），判斷 GPU 進入異常狀態：
   - 終止當前訓練 process
   - 以 exit code 42 退出 WSL
   - Windows 端 `.bat` 偵測到 42 → `wsl --terminate` → 等待 10 秒 → 重新啟動

   **特殊規則（避免誤判）**：
   - 每當偵測到 `Search: Running Trial #N` 或 `[Trial N] start:` 行，立即 **重置** epoch 計數與連續慢計數。
   - 每個 trial 的 **Epoch 1 一律忽略**（XLA JIT compile 正常會 >10s，並非 GPU hang），log 中會顯示 `(ignored: first epoch / XLA compile)`。
   - 從 Epoch 2 起才真正累計 `consecutive_slow`，連續 2 個 >10s 才觸發重啟。
   - 這避免了「新 trial 第一個 epoch XLA compile → 觸發重啟 → 重啟後又要重 compile」的死迴圈。

4. **斷點恢復**  
   透過 `.batch_training_state` 狀態檔記錄已完成的 `(phase, stock, model)` 組合。WSL 重啟後自動跳過已完成項目，從中斷點繼續。

---

## 參數一覽

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `<stock_ids>` | *(必填)* | 逗號分隔的股票代號，如 `3293,2330` |
| `--model-types` | `fundamental,trade,moment,sentiment,tech_trend,macro` | 逗號分隔的因子類型 |
| `--epoch-timeout` | `10` | 單一 epoch 逾時秒數 |
| `--feature-preprocess` | *(互動問)* | `yes` / `no`：整批統一決定是否執行特徵前處理 |
| `--validation-mode` | *(互動問，預設 3)* | `traditional` / `blocking` / `walk_forward_expanding` / `walk_forward_rolling` |
| `--no-preprocess` | — | 等同 `--feature-preprocess no` |
| `--preprocess` | — | 等同 `--feature-preprocess yes` |
| `--reset` | — | 清除狀態檔，重新開始 |

---

## 環境變數

Agent 內部會自動設定以下環境變數（可透過外部 `export` 覆蓋）：

| 環境變數 | 預設值 | 說明 |
|----------|--------|------|
| `TF_GPU_ALLOCATOR` | `cuda_malloc_async` | TF GPU 記憶體分配器 |
| `GPU_MEMORY_LIMIT_MB` | `24576` | GPU 記憶體上限 (MB) |
| `ENABLE_TF32` | `1` | 啟用 TF32 加速 |
| `ENABLE_MIXED_PRECISION` | `0` | 混合精度（RTX 5090 WSL 下建議關閉以穩定） |
| `ENABLE_XLA` | `0` | XLA 編譯（建議關閉避免首 epoch 過慢） |
| `TRAIN_MODE` | `speed` | 訓練模式 |
| `ISOLATE_STOCK_MODEL_RUNS` | `0` | Agent 控制外層迴圈，不需腳本內 subprocess |
| `FIT_VERBOSE` | `2` | 強制 one-line-per-epoch 格式，便於 epoch 時間解析 |
| `FEATURE_PREPROCESS` | *(由 batch 決定注入)* | `1`=啟用特徵前處理；`0`=pass-through。Agent 在按 `--feature-preprocess` / 互動問答後自動 export，避免 `ATT+Flood.py` / `ATT+Dflooding.py` 在子程序再問一次 |
| `VALIDATION_MODE` | *(由 batch 決定注入)* | `blocking` / `walk_forward_expanding` / `walk_forward_rolling`。新批次未指定時會互動選單，預設 `walk_forward_rolling` |

---

## 檔案結構

```
workspace_vscoding/
├── Batch_training.bat        # Windows 啟動器（WSL 重啟迴圈）
├── Batch_training.sh         # WSL 訓練代理（核心邏輯）
├── ATT+Flood.py              # Phase 1: AutoML 超參數搜尋
├── ATT+Dflooding.py          # Phase 2: 固定參數重複訓練
├── .batch_training_state     # 執行狀態檔（自動產生/清除）
├── logs/                     # 訓練 log 輸出目錄
│   └── batch_automl_3293_fundamental_20260419_*.log
└── README_Batch_training.md  # 本文件
```

---

## 常見情境

### GPU 反覆逾時重啟
若 WSL 重啟次數達到上限（預設 20 次），`.bat` 會停止重試。可嘗試：
- 降低 `GPU_MEMORY_LIMIT_MB`
- 設定 `TRAIN_MODE=safe`
- 檢查 GPU 散熱

### 新 trial 第一個 epoch 被判定超時
已在 monitor 中處理：每個 trial 的 Epoch 1 自動忽略（XLA compile 本就偏慢），從 Epoch 2 起才進入連續慢計數。若仍持續觸發，表示 Epoch 2+ 真的每個都 >10s：
- 檢查是否 `lookback_window` 過大（例如 40）導致 step time 本來就 >10s → 提高 `--epoch-timeout`
- 例：`Batch_training.bat 2317 --epoch-timeout 20`

### 想從特定 phase/model 開始
直接編輯 `.batch_training_state` 中的 `completed=` 欄位，加入已完成的標記即可跳過。

### 只想跑 Phase 2（已有 AutoML 結果）
在 `.batch_training_state` 中將所有 `automl:*` 標記為已完成，並設定 `current_phase=dflooding`。

### 批次訓練時想針對某支股票跳過特徵前處理
目前 `--feature-preprocess` 是**整批統一**，無法按股票單獨設定。若需混合模式，請拆成兩次 batch 執行（或直接從 WSL 執行 `FEATURE_PREPROCESS=0 python ATT+Flood.py`）。

### `traditional` 與 `blocking` 有什麼差別？
在 Batch_training 入口中，`traditional` 是為了易懂而提供的同義字，實際會映射為 `blocking` 並傳給子程序。

---

## 相依性

- Windows 10/11 + WSL2
- Conda 環境 `finlab`（含 TensorFlow、Keras Tuner 等）
- RTX 5090（或其他 CUDA GPU）
- Python 套件詳見 `requirements.txt`
