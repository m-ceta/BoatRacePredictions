# BoatRacePredictions

ボートレースの過去データ `B*.TXT` / `K*.TXT` から学習データを作成し、当日レースの順位予測と三連単候補を出力するプロジェクトです。

## 現在の主な機能

- `rowdata/` の不足分ダウンロード
- `data/processed/training_table.parquet` の生成
- ベースモデル学習
- 三連単 v2/v3 モデル学習、rerank 最適化、校正
- full valid 評価
- 当日レース予測
- 直近レースのバックテスト
- Streamlit Web UI
- Google Drive 上の zip からの復元と zip への出力

## ディレクトリ構成

- `rowdata/`: 元データ。`B*.TXT` は番組表、`K*.TXT` は結果です。
- `data/processed/`: build 後の Parquet データです。
- `artifacts/`: 学習済みモデル、特徴量定義、校正器、評価結果です。
- `configs/train.yaml`: 学習・評価・rerank 最適化の設定です。
- `app/streamlit_app.py`: フル機能 Web UI です。
- `app/community_today_app.py`: Streamlit Community Cloud 向けの当日予測専用 UI です。
- `sh/`: Linux / Bash 用スクリプトです。
- `bat/`: Windows 用バッチです。

## セットアップ

### Windows

既存の Python / Conda 環境を使う場合:

```bat
python -m pip install -e .
```

Conda 環境を作る場合:

```bat
bat\env_setup.bat
```

Conda 環境を開く場合:

```bat
bat\open_conda_shell.bat
```

### Linux / Google Cloud

```bash
chmod +x sh/*.sh
sh/env_setup.sh
```

`sh/env_setup.sh` は Debian パッケージ、Miniforge、Conda 環境、依存ライブラリ、必要に応じて rclone と Google Drive mount を設定します。既定では 16GB の swap を作成しますが、空き容量が 25GB 未満の場合はスキップします。

主な環境変数:

- `ENV_NAME`: Conda 環境名。既定は `boatrace-predictions` です。
- `ENABLE_SWAP`: `0` で swap 作成を無効化します。
- `SWAP_SIZE`: swap サイズ。既定は `16G` です。
- `SWAP_MIN_FREE_GB`: swap 作成に必要な最小空き容量。既定は `25` です。
- `CONFIGURE_RCLONE`: `0` で rclone 設定確認をスキップします。
- `MOUNT_GDRIVE`: `0` で Google Drive mount をスキップします。

例:

```bash
CONFIGURE_RCLONE=0 MOUNT_GDRIVE=0 sh/env_setup.sh
```

## 基本ワークフロー

通常の更新・学習は次の順序です。

```bash
boatrace-backfill-rowdata --rowdata rowdata
boatrace-build --rowdata rowdata --output data/processed
boatrace-train --config configs/train.yaml
boatrace-train-trifecta-v2 --config configs/train.yaml --max-races 0 --eval-max-races 10000 --optimize-rerank
boatrace-eval-trifecta-full --config configs/train.yaml
```

役割:

- `boatrace-build`: `rowdata/` から `data/processed/` を作成します。
- `boatrace-train`: ベースモデルのみを学習します。
- `boatrace-train-trifecta-v2`: flow / staged / v2 / v3 と rerank 関連を学習・最適化します。
- `boatrace-eval-trifecta-full`: valid 全体で三連単評価を実行します。

## CLI コマンド

### rowdata 補完

```bash
boatrace-backfill-rowdata --rowdata rowdata
```

期間を指定する場合:

```bash
boatrace-backfill-rowdata --rowdata rowdata --start 2026-05-01 --end 2026-05-31
```

主なオプション:

- `--start`: 開始日です。省略時は既存ファイルから自動判定します。
- `--end`: 終了日です。省略時は日本時刻基準で既定日を決めます。
- `--kinds`: 取得対象です。既定は `BK` です。
- `--overwrite`: 既存ファイルを上書きします。

### 学習データ生成

```bash
boatrace-build --rowdata rowdata --output data/processed
```

日付上限を指定する場合:

```bash
boatrace-build --rowdata rowdata --output data/processed --max-date 2026-05-24
```

主な出力:

- `data/processed/race_entries.parquet`
- `data/processed/race_results.parquet`
- `data/processed/training_table.parquet`

`base_buckets/` と `history_months/` は build 中間ファイルです。学習完了時に削除されます。

### ベースモデル学習

```bash
boatrace-train --config configs/train.yaml
```

GPU を使う場合:

```bash
boatrace-train --config configs/train.yaml --training-device gpu
```

`boatrace-train` はベースモデル学習のみです。三連単 v2/v3 の追加学習は実行しません。

### 三連単 v2/v3 学習

```bash
boatrace-train-trifecta-v2 --config configs/train.yaml --max-races 0 --eval-max-races 10000
```

rerank 最適化付き:

```bash
boatrace-train-trifecta-v2 --config configs/train.yaml --max-races 0 --eval-max-races 10000 --optimize-rerank
```

rerank 最適化の候補評価を並列化する場合:

```bash
boatrace-train-trifecta-v2 --config configs/train.yaml --max-races 0 --eval-max-races 10000 --optimize-rerank --optimize-rerank-workers 4
```

利用可能コア数から 1 を引いた数で自動指定する場合:

```bash
boatrace-train-trifecta-v2 --config configs/train.yaml --max-races 0 --eval-max-races 10000 --optimize-rerank --optimize-rerank-workers 0
```

最適化を最初からやり直す場合:

```bash
boatrace-train-trifecta-v2 --config configs/train.yaml --max-races 0 --eval-max-races 10000 --optimize-rerank --reset-rerank-optimization
```

主なオプション:

- `--max-races`: v2/v3 学習に使う最大レース数です。
- `--eval-max-races`: 最適化・評価に使う最大レース数です。
- `--optimize-rerank`: rerank 重み、top_n、ペナルティ、校正窓を最適化します。
- `--optimize-rerank-workers`: rerank 最適化の候補評価並列数です。既定は `1`、`0` は `CPUコア数 - 1` です。
- `--reset-rerank-optimization`: `artifacts/rerank_optimization_checkpoint.json` を破棄して再最適化します。

補足:

- top_n は CLI 引数では指定しません。
- top_n 候補は `configs/train.yaml` の `phase3.rerank.top_n_grid` で定義します。
- 最適化された top_n はモデル・メタデータ側に保存され、予測時は保存値が優先されます。
- rerank 最適化の途中経過は `artifacts/rerank_optimization_checkpoint.json` に保存されます。

### full valid 評価

```bash
boatrace-eval-trifecta-full --config configs/train.yaml
```

期間を絞る場合:

```bash
boatrace-eval-trifecta-full --config configs/train.yaml --date-from 2026-01-01 --date-to 2026-05-31
```

### 当日レース予測

```bash
boatrace-predict-today --config configs/train.yaml --venue 23 --race-no 1
```

### CSV 入力予測

順位予測のみ:

```bash
boatrace-predict --config configs/train.yaml --features artifacts/feature_columns.json --input future_races.csv --output ranking_predictions.csv
```

三連単候補も出力:

```bash
boatrace-predict --config configs/train.yaml --features artifacts/feature_columns.json --input future_races.csv --output ranking_predictions.csv --trifecta-output trifecta_predictions.csv
```

オッズ CSV を併用する場合:

```bash
boatrace-predict --config configs/train.yaml --features artifacts/feature_columns.json --input future_races.csv --output ranking_predictions.csv --trifecta-output trifecta_predictions.csv --odds odds.csv
```

### 直近バックテスト

```bash
boatrace-backtest-recent-week --config configs/train.yaml --rowdata rowdata
```

主なオプション:

- `--days`: 対象日数です。既定は `7` です。
- `--top-k`: 1レースあたりの購入候補数です。既定は `1` です。
- `--stake`: 1点あたり購入額です。既定は `100` です。
- `--start` / `--end`: 対象期間を明示します。
- `--report-dir`: レポート出力先です。

### zip 復元・出力

Google Drive 共有リンクからダウンロードして復元する場合:

```bash
boatrace-package-download
```

ローカルまたは mount 済み Drive の zip から復元する場合:

```bash
boatrace-package-restore-local --project-root . --source-dir "$HOME/gdrive/gcolab_workdir/btp"
```

zip を作成して出力する場合:

```bash
boatrace-package-export --project-root . --output-dir "$HOME/gdrive/gcolab_workdir/btp"
```

対象 zip:

- `rowdata.zip`
- `data.zip`
- `artifacts.zip`

共通オプション:

- `--skip-rowdata`
- `--skip-data`
- `--skip-artifacts`

## Web UI

### フル機能 UI

```bash
boatrace-webui
```

または:

```bash
streamlit run app/streamlit_app.py
```

機能:

- 当日レース予測
- 共有データ取得
- rowdata 更新
- 学習データ更新
- ベースモデル再学習
- 三連単 v2/v3 学習
- 直近バックテスト

Windows 用:

```bat
bat\full_ui.bat
```

Linux / Bash 用:

```bash
sh/full_ui.sh
```

### 当日予測専用 UI

```bash
boatrace-webui-today
```

または:

```bash
streamlit run app/community_today_app.py
```

Windows 用:

```bat
bat\today_ui.bat
```

Linux / Bash 用:

```bash
sh/today_ui.sh
```

## Streamlit Community Cloud

Community Cloud では `app/community_today_app.py` を entrypoint にします。予測専用アプリであり、学習や rowdata 更新 UI は含みません。

必要ファイル:

- `app/community_today_app.py`
- `app/requirements.txt`
- `src/`
- `configs/train.yaml`

Secrets 例:

```toml
data_drive_file_url = "https://drive.google.com/file/d/your-data-file-id/view?usp=drive_link"
artifacts_drive_file_url = "https://drive.google.com/file/d/your-artifacts-file-id/view?usp=drive_link"
```

補足:

- `rowdata.zip` は Community Cloud 予測専用 UI では不要です。
- `data.zip` と `artifacts.zip` は公開共有リンクで取得できる必要があります。
- Cloud 専用依存関係は `app/requirements.txt` に置きます。

## Google Drive / Google Cloud 運用

Drive mount の既定:

- remote: `gdrive:`
- mount 先: `$HOME/gdrive`
- zip 配置先: `$HOME/gdrive/gcolab_workdir/btp`

Drive を mount する場合:

```bash
sh/drive_mount.sh
```

Drive 上の zip を復元し、rowdata 差分を更新する場合:

```bash
sh/zip_update_local.sh
```

zip を作成して Drive へ出力する場合:

```bash
sh/zip_upload.sh
```

フルパイプラインを実行する場合:

```bash
sh/train_full.sh
```

rerank 最適化なしでフルパイプラインを実行する場合:

```bash
sh/train_without_rerank.sh
```

途中終了後に再開する場合:

```bash
sh/train_full_resume.sh
```

`train_full.sh` の実行順序:

1. `drive_mount`
2. `zip_update_local`
3. `boatrace-build`
4. `boatrace-train`
5. `boatrace-train-trifecta-v2 --optimize-rerank --reset-rerank-optimization`
6. `boatrace-eval-trifecta-full`
7. `zip_upload`

`train_full_resume.sh` は `.gcloud_pipeline_state/` の完了マーカーを見て、完了済みステップをスキップします。Drive mount は再開時も毎回確認します。

主な環境変数:

- `DRIVE_PACKAGE_DIR`: zip 配置先です。既定は `$HOME/gdrive/gcolab_workdir/btp` です。
- `PIPELINE_STATE_DIR`: 再開用マーカーの保存先です。既定は `.gcloud_pipeline_state` です。
- `MAX_RACES`: 三連単 v2/v3 学習の `--max-races` です。既定は `0`、つまり全件です。
- `EVAL_MAX_RACES`: 三連単 v2/v3 最適化の `--eval-max-races` です。既定は `10000` です。final_eval metrics は既定で全件です。
- `OPTIMIZE_RERANK_WORKERS`: rerank 最適化の並列数です。`train_full` / `train_full_resume` の既定は `2`、`0` は `CPUコア数 - 1` です。
- `RESTORE_ROWDATA` / `RESTORE_DATA` / `RESTORE_ARTIFACTS`: `0` で復元をスキップします。
- `UPDATE_ROWDATA`: `0` で zip 復元後の rowdata 更新をスキップします。
- `EXPORT_ROWDATA` / `EXPORT_DATA` / `EXPORT_ARTIFACTS`: `0` で zip 出力をスキップします。

Windows でも同名の `bat` を使えます。

```bat
bat\drive_mount.bat
bat\zip_update_local.bat
bat\zip_upload.bat
bat\train_full.bat
bat\train_without_rerank.bat
bat\train_full_resume.bat
```

Windows の Drive mount は rclone と WinFsp が必要です。

## スクリプト一覧

### Windows

- `bat/_common.bat`
- `bat/drive_mount.bat`
- `bat/env_setup.bat`
- `bat/exec.bat`
- `bat/full_ui.bat`
- `bat/open_conda_prompt.bat`
- `bat/open_conda_shell.bat`
- `bat/today_ui.bat`
- `bat/train_full.bat`
- `bat/train_full_resume.bat`
- `bat/train_without_rerank.bat`
- `bat/zip_update_local.bat`
- `bat/zip_upload.bat`

### Linux / Bash

- `sh/_common.sh`
- `sh/drive_mount.sh`
- `sh/env_setup.sh`
- `sh/exec.sh`
- `sh/full_ui.sh`
- `sh/open_conda_shell.sh`
- `sh/today_ui.sh`
- `sh/train_full.sh`
- `sh/train_full_resume.sh`
- `sh/train_without_rerank.sh`
- `sh/zip_update_local.sh`
- `sh/zip_upload.sh`

## 学習期間と rerank 設定

`configs/train.yaml` では相対窓を使います。

- `data.rolling_years: 3`
- `split.valid_months: 4`

最新 `race_date` を基準に、学習対象は直近 3 年、valid は直近 4 か月へ追従します。

rerank 最適化の主な設定:

- `phase3.rerank.weight_grid`
- `phase3.rerank.top_n_grid`
- `phase3.rerank.rank_penalty_strength_grid`
- `phase3.rerank.log_loss_max_delta_vs_v1`
- `phase3.calibration.window_days_options`

## 必要メモリの目安

- 当日予測: 4GB 以上、8GB 推奨
- `boatrace-build`: 12GB 以上、16GB 推奨
- `boatrace-train`: 16GB 以上、24GB 以上推奨
- `boatrace-train-trifecta-v2`: 24GB 以上、32GB 以上推奨
- full pipeline: 長時間実行になるため、Cloud VM では 32GB 以上を推奨

8GB 環境で実行する場合は swap を有効化してください。ただし swap は完走率を上げるための対策であり、学習時間は大きく伸びます。

## 補足

- `boatrace-build` は streaming build です。
- `boatrace-train` と `boatrace-train-trifecta-v2` は役割が分離されています。
- 予測に必要な最低限の成果物は `configs/`, `artifacts/`, `data/processed/training_table.parquet` です。
- 進捗ログの時刻は日本時間で出力されます。
