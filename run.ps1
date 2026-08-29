<#
.SYNOPSIS
    Windows-native task runner for the TW-50 DESQ pipeline.

.DESCRIPTION
    Mirrors the targets defined in Makefile for reviewers on Windows who do
    not have GNU make. Assumes the `finlab` conda env is already active.

.PARAMETER Target
    Named recipe to run. See -Target help for the full list.

.PARAMETER Stock
    Stock ID for single-stock recipes (default: 2330).

.PARAMETER Trials
    Keras Tuner trial budget for Stage 1 (default: 12).

.PARAMETER Epochs
    Epochs for Stage 1 tuner (default: 80).

.PARAMETER DfloodEpochs
    Epochs for Stage 2 Dynamic Flooding retrain (default: 120).

.PARAMETER Batch
    Batch size (default: 64).

.EXAMPLE
    .\run.ps1 smoke
    .\run.ps1 smoke-oof -Stock 2454
    .\run.ps1 full-2330 -Epochs 60
    .\run.ps1 figures
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('help', 'preflight', 'lint',
                 'prices', 'stage1', 'stage2', 'stage2-oof', 'stage3', 'stage3-strict',
                 'stage4-data', 'stage4-train', 'stage4-backtest',
                 'monitor-smoke', 'monitor-stage2',
                 'smoke', 'smoke-oof',
                 'full-2330', 'full-flagships', 'full-top50',
                 'seed-sweep',
                 'rerun-baselines', 'verify-baselines', 'snapshot-baselines',
                 'verify-prices', 'hash-shipped', 'repro', 'manifest-check',
                 'figures', 'figures-us', 'tables', 'tables-check',
                 'clean-smoke', 'clean-artifacts', 'clean-all')]
    [string]$Target = 'help',

    [string]$Stock = '2330',
    [string]$FlagStocks = '2330,2454',
    [int]$Trials = 12,
    [int]$Epochs = 80,
    [int]$DfloodEpochs = 120,
    [double]$DqnHours = 1.5,
    [int]$Batch = 64,
    [int]$SmokeTrials = 2,
    [int]$SmokeEpochs = 3,
    [int]$SmokeBatch = 128,
    [string]$SweepSeeds = '42,123,456,789,2024',
    [string]$SweepStages = '3'
)

$ErrorActionPreference = 'Stop'
$py = 'python'

function Invoke-Cmd {
    param([string]$Cmd)
    Write-Host ">> $Cmd" -ForegroundColor Cyan
    Invoke-Expression $Cmd
    if ($LASTEXITCODE -ne 0) { throw "Command failed (exit $LASTEXITCODE): $Cmd" }
}

function Show-Help {
    Write-Host @'
TW-50 DESQ run.ps1 -- Windows PowerShell task runner

  .\run.ps1 <target> [-Stock 2330] [-Trials 12] [-Epochs 80] [-Batch 64]

  Quick targets:
    help              print this list
    preflight         environment sanity checks
    lint              ast.parse on all pipeline scripts

    smoke             5-min plumbing test on -Stock (default 2330)
    smoke-oof         smoke + --des-oof (leakage-free DES fit)
    full-2330         production settings for TSMC
    full-flagships    TSMC + MediaTek
    full-top50        complete TW-50 batch
    seed-sweep        multi-seed Stage 3 sweep -> mean +/- std CSV

  Individual stages:
    prices            fetch OHLCV for -Stock
    stage1            Bayesian tuning + static Flooding
    stage2            Dynamic Flooding retrain (in-sample DES-train preds)
    stage2-oof        Dynamic Flooding retrain with --des-oof
    stage3            KNORA-E ensemble + backtest
    stage3-strict     stage3 with --strict-oof leakage guard
    stage4-data       build Double-DQN input from Stage 3 output
    stage4-train      train all five DQN walk-forward folds
    stage4-backtest   evaluate the promoted Stage 4 checkpoint
    monitor-smoke     synthetic Appendix-F Level 0-3 smoke
    monitor-stage2    immutable Stage 2 snapshot for -Stock

  Paper artifacts (no training required):
    figures           regenerate paper Fig 17
    figures-us        regenerate paper Fig 19
    tables            regenerate and audit revised-paper Tables 3-10
    tables-check      run revised-paper table regression tests

  Cleanup:
    clean-smoke       remove smoke-test artifacts for -Stock
    clean-artifacts   remove artifacts/ recursively
    clean-all         remove artifacts/ AND prices/
'@
}

switch ($Target) {
    'help'      { Show-Help }

    'preflight' {
        Invoke-Cmd "$py -c `"import tensorflow as tf, keras_tuner, deslib, sklearn, joblib; print('tf', tf.__version__, 'sklearn', sklearn.__version__, 'deslib', deslib.__version__)`""
        Invoke-Cmd "$py -c `"import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))`""
        Invoke-Cmd "$py -c `"from pathlib import Path; n = len(list(Path('features').glob('*.csv'))); print(f'features: {n} CSVs')`""
        Invoke-Cmd "$py -c `"from pathlib import Path; miss = [s for s in ['$Stock'] if not (Path('prices')/f'{s}.csv').exists()]; print('missing prices:', miss)`""
    }

    'lint' {
        Invoke-Cmd "$py -c `"import ast; [ast.parse(open(f, encoding='utf-8').read()) for f in ['tw50_flood.py','tw50_dflood.py','tw50_des.py','fetch_prices.py']]; print('LINT OK')`""
    }

    'prices'         { Invoke-Cmd "$py fetch_prices.py --stock-ids $Stock" }
    'stage1'         { Invoke-Cmd "$py tw50_flood.py  --stock-ids $Stock --aspect all --trials $Trials --epochs $Epochs --batch-size $Batch" }
    'stage2'         { Invoke-Cmd "$py tw50_dflood.py --stock-ids $Stock --aspect all --epochs $DfloodEpochs --batch-size $Batch" }
    'stage2-oof'    { Invoke-Cmd "$py tw50_dflood.py --stock-ids $Stock --aspect all --epochs $DfloodEpochs --batch-size $Batch --des-oof" }
    'stage3'         { Invoke-Cmd "$py tw50_des.py    --stock-ids $Stock --no-show" }
    'stage3-strict' { Invoke-Cmd "$py tw50_des.py    --stock-ids $Stock --no-show --strict-oof" }
    'stage4-data'    { Push-Location dqn; try { Invoke-Cmd "$py build_dqn_data.py --stock-ids $Stock --overwrite" } finally { Pop-Location } }
    'stage4-train'   { Push-Location dqn; try { Invoke-Cmd "$py src/train_dqn.py --symbol $Stock --fold all --hours $DqnHours" } finally { Pop-Location } }
    'stage4-backtest' { Push-Location dqn; try { Invoke-Cmd "$py src/backtest.py --symbol $Stock --out backtest_summary.csv" } finally { Pop-Location } }
    'monitor-smoke'   { Invoke-Cmd "$py -m monitoring smoke" }
    'monitor-stage2'  { Invoke-Cmd "$py -m monitoring collect-stage2 --stock-id $Stock" }

    'seed-sweep' {
        Invoke-Cmd "$py scripts/run_seed_sweep.py --stock-ids $Stock --seeds $SweepSeeds --stages $SweepStages"
    }

    'rerun-baselines'    { Invoke-Cmd "bash us/baselines/run_all_baselines.sh" }
    'verify-baselines'   { Invoke-Cmd "$py us/baselines/verify_baselines.py" }
    'snapshot-baselines' { Invoke-Cmd "bash us/baselines/run_all_baselines.sh --verify-only --force-snapshot" }

    'verify-prices' { Invoke-Cmd "$py reproducibility/verify_public_prices.py --stock-ids $Stock" }
    'hash-shipped'  { Invoke-Cmd "$py reproducibility/hash_shipped.py" }
    'manifest-check' { Invoke-Cmd "$py reproducibility/check_manifest.py" }
    'repro' {
        Invoke-Cmd "$py reproducibility/hash_shipped.py"
        Invoke-Cmd "$py reproducibility/verify_public_prices.py --stock-ids $Stock"
        & $PSCommandPath -Target 'smoke-oof' -Stock $Stock
    }

    'smoke' {
        Invoke-Cmd "$py fetch_prices.py --stock-ids $Stock"
        Invoke-Cmd "$py tw50_flood.py   --stock-ids $Stock --aspect all --trials $SmokeTrials --epochs $SmokeEpochs --batch-size $SmokeBatch"
        Invoke-Cmd "$py tw50_dflood.py  --stock-ids $Stock --aspect all --epochs 5 --batch-size $SmokeBatch"
        Invoke-Cmd "$py tw50_des.py     --stock-ids $Stock --no-show"
        Write-Host "Smoke test complete. See artifacts/des/backtest/summary.csv" -ForegroundColor Green
    }

    'smoke-oof' {
        Invoke-Cmd "$py fetch_prices.py --stock-ids $Stock"
        Invoke-Cmd "$py tw50_flood.py   --stock-ids $Stock --aspect all --trials $SmokeTrials --epochs $SmokeEpochs --batch-size $SmokeBatch"
        Invoke-Cmd "$py tw50_dflood.py  --stock-ids $Stock --aspect all --epochs 5 --batch-size $SmokeBatch --des-oof"
        Invoke-Cmd "$py tw50_des.py     --stock-ids $Stock --no-show --strict-oof"
        Write-Host "OOF smoke test complete." -ForegroundColor Green
    }

    'full-2330' {
        Invoke-Cmd "$py fetch_prices.py --stock-ids 2330"
        Invoke-Cmd "$py tw50_flood.py   --stock-ids 2330 --aspect all --trials $Trials --epochs $Epochs"
        Invoke-Cmd "$py tw50_dflood.py  --stock-ids 2330 --aspect all --epochs $DfloodEpochs --des-oof"
        Invoke-Cmd "$py tw50_des.py     --stock-ids 2330 --no-show --strict-oof"
    }

    'full-flagships' {
        Invoke-Cmd "$py fetch_prices.py --stock-ids $FlagStocks"
        Invoke-Cmd "$py tw50_flood.py   --stock-ids $FlagStocks --aspect all --trials $Trials --epochs $Epochs"
        Invoke-Cmd "$py tw50_dflood.py  --stock-ids $FlagStocks --aspect all --epochs $DfloodEpochs --des-oof"
        Invoke-Cmd "$py tw50_des.py     --stock-ids $FlagStocks --no-show --strict-oof"
    }

    'full-top50' {
        Invoke-Cmd "$py fetch_prices.py --top50 --sleep 0.4"
        Invoke-Cmd "$py tw50_flood.py   --top50 --aspect all --trials $Trials --epochs $Epochs"
        Invoke-Cmd "$py tw50_dflood.py  --top50 --aspect all --epochs $DfloodEpochs --des-oof"
        Invoke-Cmd "$py tw50_des.py     --top50 --no-show --strict-oof"
    }

    'figures'    { Invoke-Cmd "$py evaluation/render_figure_backtest.py" }
    'figures-us' { Invoke-Cmd "$py us/baselines/combined/combined_comparison.py" }
    'tables'      { Invoke-Cmd "$py evaluation/paper/generate_tables.py" }
    'tables-check' { Invoke-Cmd "$py -m unittest discover -s evaluation/paper/tests -v" }

    'clean-smoke' {
        Write-Host "Removing smoke-test artifacts for STOCK=$Stock ..." -ForegroundColor Yellow
        Get-ChildItem -Path artifacts/des/backtest,artifacts/des/pred,artifacts/des/models,artifacts/dflood/pred,artifacts/dflood/models `
            -Filter "*$Stock*" -Recurse -ErrorAction SilentlyContinue | Remove-Item -Force
    }

    'clean-artifacts' {
        if (Test-Path artifacts) { Remove-Item artifacts -Recurse -Force }
    }

    'clean-all' {
        if (Test-Path artifacts) { Remove-Item artifacts -Recurse -Force }
        if (Test-Path prices)    { Remove-Item prices    -Recurse -Force }
        Write-Host "All generated artifacts removed. features/ is retained." -ForegroundColor Yellow
    }
}
