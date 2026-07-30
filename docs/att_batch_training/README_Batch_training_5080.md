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

選擇 validation 方式（預設 walk-forward rolling）：

```bat
Batch_training.bat 3293 --validation rolling --wf-splits 5
Batch_training.bat 3293 --validation expanding
Batch_training.bat 3293 --validation traditional
```

不帶 `--validation` 且未設 env 時，WSL 端會互動提示（`1`/`2`/`3`，10 秒超時預設 3）。詳見 [SKILL.md](SKILL.md#6-validation-策略)。

### 2. 直接在 WSL 內執行（不需要自動 WSL 重啟）

```bash
bash Batch_training.sh 3293
bash Batch_training.sh 3293,2330 --epoch-timeout 12
```

### 3. 清除狀態檔重新開始

```bat
Batch_training.bat --reset
```

---

## 執行流程

> **重要**：每次呼叫 agent 都會**連續跑完 Phase 1 → Phase 2**，不需要人工介入。Phase 1 AutoML 全部結束後，Phase 2 Dynamic Flooding 會自動接著執行（由 `Batch_training.sh` / `run_att_agent.sh` 的 `main()` 迴圈保證）。兩階段都會記錄在 `.batch_training_state` 的 `completed` 列表，WSL 若中途重啟也會從中斷點繼續。

> **單一標的預估時間（RTX 5090 / 5080 實測，兩者差不多）**  
> Phase 1 超參數搜尋 ≈ 3 小時，Phase 2 Dflooding 正式訓練 ≈ 2 小時，後續 DES 集成 ≈ 5 分鐘。  
> 一支股票 6 個面向全跑完約 5 小時（不含 DES）。

1. **Phase 1 — AutoML（ATT+Flood.py）**  
   對每個 `(stock_id, model_type)` 組合執行 Bayesian Optimization 超參數搜尋（stage1: 12 trials / 80 epochs → stage2: 24 trials / 120 epochs）。產出寫入 `D:/hyperbayes_ATT/ATT_{model_type}_{stock_id}/best_trial_summary.json`。

2. **Phase 2 — Dynamic Flooding（ATT+Dflooding.py）**  
   讀取 Phase 1 產出的最佳超參數，以固定設定重複訓練 18 次，保留 top 3 模型寫入 `D:/experiments_ATT/`。兩個 phase 都透過同一個 `STOCK_IDS` 環境變數接收 stock 清單（勿改為其它名稱，見 SKILL.md §7）。

3. **Epoch 監控**  
   後台 `tail -f` log 檔，即時解析每個 epoch 耗時。若連續 2 個 epoch 超過閾值（預設 10 秒），判斷 GPU 進入異常狀態：
   - 終止當前訓練 process
   - 以 exit code 42 退出 WSL
   - Windows 端 `.bat` 偵測到 42 → `wsl --terminate` → 等待 10 秒 → 重新啟動

4. **斷點恢復**  
   透過 `.batch_training_state` 狀態檔記錄已完成的 `(phase, stock, model)` 組合。WSL 重啟後自動跳過已完成項目，從中斷點繼續，直到 Phase 2 最後一個 job 完成才會把 state file 清除。

---

## 參數一覽

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `<stock_ids>` | *(必填)* | 逗號分隔的股票代號，如 `3293,2330` |
| `--model-types` | `fundamental,trade,moment,sentiment,tech_trend,macro` | 逗號分隔的因子類型 |
| `--epoch-timeout` | `10` | 單一 epoch 逾時秒數 |
| `--validation` | `rolling` | `1`/`traditional`、`2`/`expanding`、`3`/`rolling`（walk-forward） |
| `--wf-splits` | `5` | walk-forward fold 數 |
| `--wf-val-ratio` | `0.2` | 每 fold 驗證集比例 |
| `--wf-gap` | `10` | train/val gap（防 leakage） |
| `--reset` | — | 清除狀態檔，重新開始 |

---

## 環境變數

Agent 內部會自動設定以下環境變數（可透過外部 `export` 覆蓋）：

| 環境變數 | 預設值 | 說明 |
|----------|--------|------|
| `TF_GPU_ALLOCATOR` | `cuda_malloc_async` | TF GPU 記憶體分配器 |
| `GPU_MEMORY_LIMIT_MB` | `12288` | GPU 記憶體上限 (MB)，0=不限制 |
| `ENABLE_TF32` | `1` | 啟用 TF32 加速 |
| `ENABLE_MIXED_PRECISION` | `0` | 混合精度（RTX 5080 WSL 下建議關閉以穩定） |
| `ENABLE_XLA` | `0` | XLA 編譯（建議關閉避免首 epoch 過慢） |
| `TRAIN_MODE` | `speed` | 訓練模式 |
| `ISOLATE_STOCK_MODEL_RUNS` | `0` | Agent 控制外層迴圈，不需腳本內 subprocess |
| `FIT_VERBOSE` | `2` | 強制 one-line-per-epoch 格式，便於 epoch 時間解析 |
| `FEATURE_PREPROCESS` | `0` | 預設關閉特徵前處理 (pass-through)，避免批次被互動 prompt 卡住；若要開啟設 `1` |
| `VALIDATION_MODE` | `walk_forward_rolling` | `blocking` / `walk_forward_expanding` / `walk_forward_rolling`（agent 覆蓋 python 預設） |
| `WF_N_SPLITS` | `5` | walk-forward fold 數 |
| `WF_VAL_RATIO` | `0.2` | 每 fold 驗證集比例 |
| `WF_GAP` | `10` | train/val gap |
| `VENV_ACTIVATE` | `$HOME/venvs/finlab/bin/activate` | Python venv 啟用腳本路徑（優先） |
| `CONDA_ENV_NAME` | `finlab` | conda 環境名稱（venv 不存在時才用） |
| `CONDA_SH_PATH` | `$HOME/miniconda3/etc/profile.d/conda.sh` | conda init 腳本（fallback） |

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

### 想從特定 phase/model 開始
直接編輯 `.batch_training_state` 中的 `completed=` 欄位，加入已完成的標記即可跳過。

### 只想跑 Phase 2（已有 AutoML 結果）
在 `.batch_training_state` 中將所有 `automl:*` 標記為已完成，並設定 `current_phase=dflooding`。

---

## 相依性

- Windows 10/11 + WSL2 (distro 名稱預設 `Ubuntu`；若不同請改 `Batch_training.bat` 的 `WSL_DISTRO`)
- WSL 端 Python 環境：優先使用 venv `~/venvs/finlab`；如不存在則 fallback 到 conda `finlab`
- RTX 5080（或其他 CUDA GPU）
- Python 套件詳見 [requirements.txt](requirements.txt)（由 finlab venv `pip freeze` 產生）。
  - 安裝：`python -m venv ~/venvs/finlab && source ~/venvs/finlab/bin/activate && pip install -r requirements.txt`
  - 注意：TensorFlow 為自編 wheel（`tensorflow==2.20.0-dev0+selfbuilt`，含 sm_120 支援，給 RTX 5080/5090 Blackwell 用），不在 PyPI；需另從本地 wheel 安裝或自編。
