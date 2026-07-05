# BoatRacePredictions

ボートレースの過去 `B*.TXT` / `K*.TXT` を学習データとして使い、当日レースの順位予測と三連単候補を出力するプロジェクトです。

## 主な機能

- `rowdata` の不足分補完
- `training_table.parquet` の生成
- ranker / classifier / flow / staged / Phase3 の学習
- 当日レース予測
- Streamlit Web UI
- Google Drive 共有リンクからの `rowdata / data / artifacts` 復元

## ディレクトリ

- `rowdata/`
  - 元データ `B*.TXT` / `K*.TXT`
- `data/processed/`
  - `race_entries.parquet`
  - `race_results.parquet`
  - `training_table.parquet`
- `artifacts/`
  - 学習済みモデル、校正器、特徴量定義、メトリクス
- `configs/train.yaml`
  - 学習・評価設定
- `app/streamlit_app.py`
  - フル機能 Web UI
- `app/community_today_app.py`
  - Streamlit Community Cloud 用の当日予測専用アプリ

## セットアップ

### Conda

Windows:

```bat
bat\conda_setup.bat
conda activate boatrace-predictions
```

Linux / Bash:

```bash
chmod +x sh/*.sh
sh/conda_setup.sh
conda activate boatrace-predictions
```

### pip

```bash
pip install -e .
```

## 主なコマンド

### rowdata 補完

```bash
boatrace-backfill-rowdata --rowdata rowdata
```

補足:

- `--end` 省略時は時間帯で既定終了日が変わります
- `07:00` から `20:59` は前日分まで
- `21:00` から `06:59` は当日分まで

期間指定:

```bash
boatrace-backfill-rowdata --rowdata rowdata --start 2026-05-01 --end 2026-05-31
```

### 学習データ生成

```bash
boatrace-build --rowdata rowdata --output data/processed
```

日付上限を切る場合:

```bash
boatrace-build --rowdata rowdata --output data/processed --max-date 2026-05-24
```

生成される最終成果物:

- `data/processed/race_entries.parquet`
- `data/processed/race_results.parquet`
- `data/processed/training_table.parquet`

`base_buckets/` と `history_months/` は build 中間ファイルです。学習や予測には不要で、学習完了時に自動削除されます。

### モデル再学習

```bash
boatrace-train --config configs/train.yaml
```

### 三連単最適化学習

```bash
boatrace-train-trifecta-v2 --config configs/train.yaml --max-races 1000 --eval-max-races 3000
```

rerank 最適化付き:

```bash
boatrace-train-trifecta-v2 --config configs/train.yaml --max-races 1000 --eval-max-races 3000 --optimize-rerank
```

`--optimize-rerank` 実行中は標準出力に進捗が表示されます。rerank 最適化の途中経過は `artifacts/rerank_optimization_checkpoint.json` に逐次保存され、同じグリッド・同じ評価レース数で再実行すると未完了の組み合わせから再開します。

最初から最適化をやり直す場合:

```bash
boatrace-train-trifecta-v2 --config configs/train.yaml --max-races 1000 --eval-max-races 3000 --optimize-rerank --reset-rerank-optimization
```

### full valid 再評価

```bash
boatrace-eval-trifecta-full --config configs/train.yaml
```

### 過去1週間分予測

直近7日間の学習データを使って過去レースを再予測し、各レースで予想上位の買い目を購入した想定で正解率と回収率を集計します。

```bash
boatrace-backtest-recent-week --config configs/train.yaml --rowdata rowdata
```

主なオプション:

```bash
boatrace-backtest-recent-week --config configs/train.yaml --rowdata rowdata --days 7 --top-k 1 --stake 100
```

- `--days`: 対象日数。既定値は `7`
- `--top-k`: 1レースごとに購入する予想上位件数。既定値は `1`
- `--stake`: 1点あたり購入額。既定値は `100`

### 当日レース予測

```bash
boatrace-predict-today --config configs/train.yaml --venue 23 --race-no 1
```

### CSV 入力予測

順位予測:

```bash
boatrace-predict --config configs/train.yaml --features artifacts/feature_columns.json --input future_races.csv --output ranking_predictions.csv
```

三連単候補も出力:

```bash
boatrace-predict --config configs/train.yaml --features artifacts/feature_columns.json --input future_races.csv --output ranking_predictions.csv --trifecta-output trifecta_predictions.csv
```

## Google Drive 共有リンクからの復元

### `boatrace-package-download`

```bash
boatrace-package-download
```

このコマンドは次の3つを扱います。

- `rowdata.zip` -> `rowdata/`
- `data.zip` -> `data/`
- `artifacts.zip` -> `artifacts/`

指定例:

```bash
boatrace-package-download \
  --rowdata-url "https://drive.google.com/file/d/..." \
  --data-url "https://drive.google.com/file/d/..." \
  --artifacts-url "https://drive.google.com/file/d/..."
```

一部だけ復元する例:

```bash
boatrace-package-download --skip-rowdata
boatrace-package-download --skip-data
boatrace-package-download --skip-artifacts
```

補足:

- `data.zip` と `artifacts.zip` の URL が同じ場合でも、内部では 1 回だけダウンロードして再利用します
- 次の環境変数があれば既定値として使います
  - `BOATRACE_ROWDATA_DRIVE_FILE_URL`
  - `BOATRACE_DATA_DRIVE_FILE_URL`
  - `BOATRACE_ARTIFACTS_DRIVE_FILE_URL`

### セキュリティ

Google Drive へのアップロード処理は削除しています。現在はダウンロード復元のみです。

## Streamlit Web UI

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
- モデル再学習
- 三連単最適化学習
- 過去1週間分予測

### Streamlit Community Cloud 用アプリ

エントリポイント:

```bash
streamlit run app/community_today_app.py
```

またはローカル用ラッパー:

```bash
boatrace-webui-today
```

## Streamlit Community Cloud note

- Community Cloud deploys `app/community_today_app.py`.
- Put Cloud-only Python dependencies in `app/requirements.txt`.
- Keep `environment.yml` for local Conda setup.
- Because Community Cloud searches the entrypoint directory before the repo root, `app/requirements.txt` will be used instead of the root `environment.yml`.

用途:

- 当日予測専用
- Community Cloud 上で `data.zip` / `artifacts.zip` を取得してから予測実行
- `rowdata` の復元や学習系の UI は含みません

### Streamlit Community Cloud へのデプロイ

Community Cloud では `requirements.txt` を使って依存ライブラリを自動インストールします。

必要ファイル:

- `requirements.txt`
- `app/community_today_app.py`
- `src/`
- `configs/train.yaml`

Secrets の例:

```toml
data_drive_file_url = "https://drive.google.com/file/d/your-data-file-id/view?usp=drive_link"
artifacts_drive_file_url = "https://drive.google.com/file/d/your-artifacts-file-id/view?usp=drive_link"
```

雛形:

- `.streamlit/secrets.toml.example`

デプロイ手順:

1. このリポジトリを GitHub に push
2. Streamlit Community Cloud で新しい app を作成
3. Main file path に `app/community_today_app.py` を指定
4. 必要なら `data_drive_file_url` と `artifacts_drive_file_url` を Secrets に設定
5. Deploy

補足:

- Community Cloud 用アプリは予測専用です
- `rowdata.zip` は不要です
- `data.zip` と `artifacts.zip` は公開共有リンクで取得できる必要があります

## Google Cloud + Google Drive

Google Cloud VM で Google Drive をマウントして学習する場合は、`rclone` を使います。

Conda を使う最小構成の環境セットアップ:

```bash
chmod +x sh/*.sh
sh/gcloud_conda_min_setup.sh
```

`sh/gcloud_conda_min_setup.sh` は Debian パッケージ、Miniforge、Conda 環境、実行時依存関係を設定します。`environment.yml` は使わず、開発用の `pytest` / `ruff` は入れません。
また、既定では `rclone` の Google Drive remote 設定確認と `$HOME/gdrive` へのマウントも実行します。remote 名は `gdrive` を想定しています。

セットアップ後、シェルを開き直すか次を実行します。

```bash
source ~/.bashrc
conda activate boatrace-predictions
```

初回のみ VM 上で Google Drive remote を作成します。

```bash
rclone config
```

`sh/gcloud_conda_min_setup.sh` 実行時に remote が未設定の場合は、この `rclone config` が自動で起動します。ブラウザなし VM では `Use auto config? n` を選び、表示された URL を手元 PC のブラウザで開いて認証コードを VM に貼り付けます。

Drive 設定やマウントをセットアップから外す場合:

```bash
CONFIGURE_RCLONE=0 MOUNT_GDRIVE=0 sh/gcloud_conda_min_setup.sh
```

remote 名やマウント先を変える場合:

```bash
RCLONE_REMOTE_NAME="mydrive" GDRIVE_MOUNT_DIR="$HOME/gdrive" sh/gcloud_conda_min_setup.sh
```

remote 名を `gdrive` にした場合、次のコマンドで Drive を `$HOME/gdrive` にマウントします。

```bash
sh/gcloud_drive_mount.sh
```

remote 名やマウント先を変える場合:

```bash
RCLONE_REMOTE_PATH="gdrive:" GDRIVE_MOUNT_DIR="$HOME/gdrive" sh/gcloud_drive_mount.sh
```

Drive 側の zip 配置先は既定で `$HOME/gdrive/BoatRacePredictions` です。このフォルダに次の zip を置きます。

- `rowdata.zip`
- `data.zip`
- `artifacts.zip`

マウント済み Drive から zip を復元し、`rowdata` 差分を更新する場合:

```bash
DRIVE_PACKAGE_DIR="$HOME/gdrive/BoatRacePredictions" sh/gcloud_drive_restore_update.sh
```

復元対象を絞る場合:

```bash
RESTORE_ROWDATA=0 RESTORE_DATA=1 RESTORE_ARTIFACTS=1 sh/gcloud_drive_restore_update.sh
```

ビルド、学習、三連単最適化、zip 作成、Drive へのアップロードまで実行する場合:

```bash
DRIVE_PACKAGE_DIR="$HOME/gdrive/BoatRacePredictions" sh/gcloud_build_train_upload.sh
```

`sh/gcloud_build_train_upload.sh` は次を実行します。

1. `boatrace-build --rowdata rowdata --output data/processed`
2. `boatrace-train --config configs/train.yaml`
3. `boatrace-train-trifecta-v2 --config configs/train.yaml --max-races 1000 --eval-max-races 3000 --optimize-rerank`
4. `boatrace-package-export --output-dir "$DRIVE_PACKAGE_DIR"`

Debian 環境で Drive マウントからアップロードまでを順番に実行する場合:

```bash
DRIVE_PACKAGE_DIR="$HOME/gdrive/BoatRacePredictions" sh/gcloud_debian_full_pipeline.sh
```

このスクリプトは次を順番に実行します。

1. `sh/gcloud_drive_mount.sh`
2. `sh/gcloud_drive_restore_update.sh`
3. `sh/data_build.sh`
4. `sh/train.sh`
   - `boatrace-train-trifecta-v2` は `--optimize-rerank --reset-rerank-optimization` 付きで新規最適化
5. `sh/gcloud_build_train_upload.sh`
   - 前段でビルドと学習済みのため、zip 作成と Drive 出力のみ実行

途中終了後に続きから実行する場合:

```bash
DRIVE_PACKAGE_DIR="$HOME/gdrive/BoatRacePredictions" sh/gcloud_debian_resume_pipeline.sh
```

再開用スクリプトは `.gcloud_pipeline_state/` の完了マーカーを見て完了済みステップをスキップします。Drive マウントは VM 再起動で外れることがあるため、再開時も毎回確認実行します。rerank 最適化は `artifacts/rerank_optimization_checkpoint.json` から再開します。

主な環境変数:

- `DRIVE_PACKAGE_DIR`: Drive 側の zip 配置先。既定は `$HOME/gdrive/BoatRacePredictions`
- `MAX_RACES`: 三連単学習の `--max-races`。既定は `1000`
- `EVAL_MAX_RACES`: 三連単評価の `--eval-max-races`。既定は `3000`
- `OPTIMIZE_RERANK`: `1` で `--optimize-rerank` を有効化。既定は `1`
- `RESET_RERANK_OPTIMIZATION`: `1` で rerank 最適化チェックポイントを破棄して再実行。既定は `0`
- `EXPORT_ROWDATA` / `EXPORT_DATA` / `EXPORT_ARTIFACTS`: `0` で該当 zip の出力をスキップ

zip のローカル復元だけを直接実行する場合:

```bash
boatrace-package-restore-local --project-root . --source-dir "$HOME/gdrive/BoatRacePredictions"
```

zip の作成と Drive への出力だけを直接実行する場合:

```bash
boatrace-package-export --project-root . --output-dir "$HOME/gdrive/BoatRacePredictions"
```

## バッチ / シェル

### Windows

- `bat\data_build.bat`
  - `boatrace-backfill-rowdata`
  - `boatrace-build`
- `bat\train.bat`
  - `boatrace-train`
  - `boatrace-train-trifecta-v2 --max-races 1000 --eval-max-races 3000`
- `bat\opt.bat`
  - `boatrace-train-trifecta-v2 --max-races 1000 --eval-max-races 3000 --optimize-rerank`
  - `artifacts/rerank_optimization_checkpoint.json` から再開
- `bat\eval.bat`
  - `boatrace-eval-trifecta-full`
- `bat\data_download.bat`
  - `boatrace-package-download`
- `bat\ui.bat`
  - `boatrace-webui`
- `bat\today_ui.bat`
  - `boatrace-webui-today`

### Linux / Bash

- `sh/data_build.sh`
- `sh/train.sh`
  - `boatrace-train`
  - `boatrace-train-trifecta-v2 --max-races 1000 --eval-max-races 3000`
- `sh/opt.sh`
  - `boatrace-train-trifecta-v2 --max-races 1000 --eval-max-races 3000 --optimize-rerank`
  - `artifacts/rerank_optimization_checkpoint.json` から再開
- `sh/eval.sh`
- `sh/data_download.sh`
- `sh/ui.sh`
- `sh/today_ui.sh`
- `sh/gcloud_drive_mount.sh`
  - `rclone mount` で Google Drive をマウント
- `sh/gcloud_conda_min_setup.sh`
  - Google Cloud Debian 用の最小 Conda 環境を構築
- `sh/gcloud_drive_restore_update.sh`
  - マウント済み Drive から `rowdata.zip` / `data.zip` / `artifacts.zip` を復元
  - `boatrace-backfill-rowdata`
- `sh/gcloud_build_train_upload.sh`
  - `boatrace-build`
  - `boatrace-train`
  - `boatrace-train-trifecta-v2 --optimize-rerank`
  - `boatrace-package-export`
- `sh/gcloud_debian_full_pipeline.sh`
  - Debian VM 向けの全工程実行
  - rerank 最適化は新規実行
- `sh/gcloud_debian_resume_pipeline.sh`
  - `.gcloud_pipeline_state/` と rerank チェックポイントを使って続きから実行

## 学習期間ルール

`configs/train.yaml` では相対窓を使っています。

- `data.rolling_years: 3`
- `split.valid_months: 4`

最新 `race_date` を基準に、実行時に自動で

- 学習対象: 直近 3 年
- valid: 直近 4 か月

へ追従します。

## 必要メモリの目安

### 当日予測

- 最低ライン: `2GB`
- 現実的な下限: `4GB`
- 安心目安: `8GB`

### `boatrace-build`

- 最低ライン: `8GB`
- 現実的な下限: `12GB`
- 安心目安: `16GB`

### `boatrace-train`

- 最低目安: `16GB`
- 安全目安: `24GB`
- 推奨: `32GB`

### `boatrace-train-trifecta-v2`

- 最低目安: `24GB`
- 安全目安: `32GB`
- 推奨: `48GB`
- `--optimize-rerank` 付きの長時間実行では `32GB` 以上を推奨します。Cloud VM では `8 vCPU / 64GB` 程度を目安にしてください。

## 補足

- `boatrace-build` は streaming build 化されており、一括 build よりメモリを抑えています
- rerank 最適化の途中経過は `artifacts/rerank_optimization_checkpoint.json` に保存されます
- 学習完了時には `data/processed/base_buckets` と `data/processed/history_months` を自動削除します
- 予測専用で別 PC に持っていく場合は、最低限 `configs/`, `artifacts/`, `data/processed/training_table.parquet` が必要です
