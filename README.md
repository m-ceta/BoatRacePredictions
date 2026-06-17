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
boatrace-train-trifecta-v2 --config configs/train.yaml --max-races 1000 --eval-max-races 1000 --eval-rerank-top-n 10
```

rerank 最適化付き:

```bash
boatrace-train-trifecta-v2 --config configs/train.yaml --max-races 1000 --eval-max-races 1000 --eval-rerank-top-n 10 --optimize-rerank
```

### full valid 再評価

```bash
boatrace-eval-trifecta-full --config configs/train.yaml
```

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

## バッチ / シェル

### Windows

- `bat\data_build.bat`
  - `boatrace-backfill-rowdata`
  - `boatrace-build`
- `bat\train.bat`
  - `boatrace-train`
  - `boatrace-train-trifecta-v2`
- `bat\opt.bat`
  - `boatrace-train-trifecta-v2 --optimize-rerank`
- `bat\eval.bat`
  - `boatrace-eval-trifecta-full`
- `bat\data_download.bat`
  - `boatrace-package-download`
- `bat\ui.bat`
  - `boatrace-webui`

### Linux / Bash

- `sh/data_build.sh`
- `sh/train.sh`
- `sh/opt.sh`
- `sh/eval.sh`
- `sh/data_download.sh`
- `sh/ui.sh`

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

## 補足

- `boatrace-build` は streaming build 化されており、一括 build よりメモリを抑えています
- 学習完了時には `data/processed/base_buckets` と `data/processed/history_months` を自動削除します
- 予測専用で別 PC に持っていく場合は、最低限 `configs/`, `artifacts/`, `data/processed/training_table.parquet` が必要です
