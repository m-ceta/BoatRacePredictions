# BoatRacePredictions

ボートレースの過去 `B*.TXT` / `K*.TXT` を学習データに使い、当日レースの順位予測と三連単予測を行う Python プロジェクトです。

主な機能:

- `rowdata` の不足分補完
- 学習データ生成
- 順位予測モデル再学習
- 三連単 `Phase3` 最適化学習
- 三連単評価
- 当日レース予測
- Streamlit Web UI
- 学習成果物の zip 化と Google Drive へのアップロード

## 構成

- `rowdata/`
  - 過去の `B*.TXT` / `K*.TXT`
- `data/processed/`
  - `race_entries.parquet`
  - `race_results.parquet`
  - `training_table.parquet`
- `artifacts/`
  - 学習済みモデル、校正器、メトリクス
- `configs/train.yaml`
  - 学習・評価・推論設定
- `app/streamlit_app.py`
  - Web UI

## セットアップ

### Conda

```bat
bat\conda_setup.bat
conda activate boatrace-predictions
```

環境を有効化したコマンドプロンプトを開く場合:

```bat
bat\open_conda_prompt.bat
```

Linux / Bash 環境:

```bash
chmod +x sh/*.sh
sh/conda_setup.sh
conda activate boatrace-predictions
```

Conda 環境を有効化したシェルを開く場合:

```bash
sh/open_conda_shell.sh
```

### pip

```bash
pip install -e .
```

## 主なコマンド

### 1. `rowdata` 補完

最新既存日付の翌日から当日まで補完:

```bash
boatrace-backfill-rowdata --rowdata rowdata
```

期間指定:

```bash
boatrace-backfill-rowdata --rowdata rowdata --start 2026-05-01 --end 2026-05-31
```

`B` のみ:

```bash
boatrace-backfill-rowdata --rowdata rowdata --start 2026-05-01 --end 2026-05-31 --kinds B
```

### 2. 学習データ生成

```bash
boatrace-build --rowdata rowdata --output data/processed
```

日付上限付き:

```bash
boatrace-build --rowdata rowdata --output data/processed --max-date 2026-05-24
```

補足:

- `boatrace-build` は streaming build です
- 全件を一度に Python list / DataFrame へ積み上げず、中間 parquet を使って処理します

### 3. ビルド結果比較

```bash
boatrace-compare-processed --expected data/processed_old --actual data/processed
```

### 4. モデル再学習

```bash
boatrace-train --config configs/train.yaml
```

補足:

- `training_table.parquet` の最新 `race_date` を見て、学習上限日は実行時に自動調整されます
- `configs/train.yaml` を毎回手で更新しなくても、`rowdata` 補完後の最新日まで追従します

### 5. 三連単 `Phase3` 学習

```bash
boatrace-train-trifecta-v2 --config configs/train.yaml --max-races 1000 --eval-max-races 1000 --eval-rerank-top-n 10
```

rerank 最適化込み:

```bash
boatrace-train-trifecta-v2 --config configs/train.yaml --max-races 1000 --eval-max-races 1000 --eval-rerank-top-n 10 --optimize-rerank
```

`--optimize-rerank` の意味:

- 付けない:
  - `Phase3` モデルを学習する
- 付ける:
  - `Phase3` 学習に加えて rerank 設定の探索も行う
  - その分かなり重くなる

### 6. 三連単 full valid 評価

```bash
boatrace-eval-trifecta-full --config configs/train.yaml
```

### 7. CSV から予測

順位予測:

```bash
boatrace-predict --config configs/train.yaml --features artifacts/feature_columns.json --input future_races.csv --output ranking_predictions.csv
```

三連単まで出力:

```bash
boatrace-predict --config configs/train.yaml --features artifacts/feature_columns.json --input future_races.csv --output ranking_predictions.csv --trifecta-output trifecta_predictions.csv
```

オッズ込み:

```bash
boatrace-predict --config configs/train.yaml --features artifacts/feature_columns.json --input future_races.csv --output ranking_predictions.csv --trifecta-output trifecta_predictions.csv --odds odds.csv
```

### 8. 当日レース予測

```bash
boatrace-predict-today --config configs/train.yaml --venue 23 --race-no 1
```

補足:

- `future_races.csv` を手で作る必要はありません
- 内部で番組表取得と特徴量生成を行います
- オッズ取得に成功した場合は買い候補も表示します

### 9. Web UI

```bash
boatrace-webui
```

または:

```bash
streamlit run app/streamlit_app.py
```

既定 URL:

- `http://localhost:8501`
- `http://127.0.0.1:8501`

Web UI でできること:

- 当日レース予測
- `rowdata` 補完
- 学習データ更新
- モデル再学習
- 三連単最適化学習
- 学習成果物アップロード

学習系 UI の補足:

- `モデル再学習`
  - `boatrace-train`
- `三連単最適化学習`
  - `boatrace-train-trifecta-v2`
- どちらも `CPU` / `GPU` を選択可能
- 既定値は `CPU`

### 10. 学習成果物の zip 化と Google Drive アップロード

```bash
boatrace-package-upload
```

このコマンドで次を行います。

1. `rowdata/` を `rowdata.zip` に圧縮
2. `artifacts/`, `configs/`, `data/` を `drp.zip` に圧縮
3. Google Drive の指定フォルダへ同名上書きアップロード

既定のアップロード先:

- `https://drive.google.com/drive/folders/19HHxA5r4T_IqMrDyNqU3qRUoRkhT87OL?usp=drive_link`

認証:

- `google_drive_credentials.json` をルートへ配置
- 初回実行時にブラウザ認証
- トークンは `artifacts/google-drive-token.json` に保存

明示指定例:

```bash
boatrace-package-upload --project-root . --drive-folder https://drive.google.com/drive/folders/19HHxA5r4T_IqMrDyNqU3qRUoRkhT87OL?usp=drive_link --credentials google_drive_credentials.json --token artifacts/google-drive-token.json
```

### 11. パッケージダウンロードと復元

```bash
boatrace-package-download
```

このコマンドで次を行います。

1. Google Drive 共有リンクから `rowdata.zip` をダウンロード
2. Google Drive 共有リンクから `brp.zip` をダウンロード
3. `rowdata.zip` から `rowdata/` を上書き復元
4. `brp.zip` から `data/` と `artifacts/` を上書き復元

既定の共有リンク:

- `brp.zip`
  - `https://drive.google.com/file/d/14w8W6xqi-NmnePs7waYhrUrxYD378YHq/view?usp=drive_link`
- `rowdata.zip`
  - `https://drive.google.com/file/d/1mtjumyk9k43UlGa7c2URfAZmUFgt_9En/view?usp=drive_link`

必要に応じて URL を上書きできます。

```bash
boatrace-package-download --brp-url https://drive.google.com/file/d/.../view?usp=drive_link --rowdata-url https://drive.google.com/file/d/.../view?usp=drive_link
```

## 入力ファイル

### `future_races.csv`

CSV 予測用の入力です。

- 1レースあたり 6 行
- 1行 = 1艇
- `race_id` は同一レースで共通
- 学習時と同じ特徴量列を持つ

通常の当日予測では内部生成されるため、手作成は不要です。

### `odds.csv`

最小列:

- `race_id`
- `trifecta`
- `odds`

例:

```csv
race_id,trifecta,odds
20260522_24_01,1-2-3,18.4
20260522_24_01,1-3-2,21.7
20260522_24_01,2-1-3,35.9
```

## 学習済み成果物

主な `artifacts/`:

- `catboost_ranker.cbm`
- `lightgbm_ranker.txt`
- `feature_columns.json`
- `ensemble_weights.json`
- `trifecta_isotonic.joblib`
- `trifecta_v2_isotonic.joblib`
- `trifecta_v3_isotonic.joblib`
- `trifecta_v2_model.joblib`
- `metrics.json`
- `classifiers/*.txt`
- `staged/*.txt`
- `flow_lightgbm.txt`
- `flow_classes.json`

## 別 PC へ持っていく最小セット

予測だけなら次で十分です。

- `artifacts/`
- `configs/`
- `data/processed/training_table.parquet`
- ソースコード一式

## 日次・週次・月次運用

### 日次

1. `boatrace-backfill-rowdata --rowdata rowdata`
2. `boatrace-build --rowdata rowdata --output data/processed`
3. 当日予測

### 週次

1. `rowdata` 補完
2. `data/processed` 更新

### 月次

1. `rowdata` 補完
2. `data/processed` 更新
3. `boatrace-train`
4. `boatrace-train-trifecta-v2`
5. 必要に応じて `boatrace-eval-trifecta-full`

## タスクスケジューラ用バッチ

タスクスケジューラから定期実行する前提で、Conda 仮想環境 `boatrace-predictions` を有効化してから処理を流すバッチを用意しています。

### `bat\data_build.bat`

実行内容:

1. `boatrace-backfill-rowdata --rowdata rowdata`
2. `boatrace-build --rowdata rowdata --output data/processed`
3. `boatrace-package-upload`

### `sh/data_build.sh`

実行内容:

1. `boatrace-backfill-rowdata --rowdata rowdata`
2. `boatrace-build --rowdata rowdata --output data/processed`
3. `boatrace-package-upload`

### `bat\train.bat`

実行内容:

1. `boatrace-train --config configs/train.yaml`
2. `boatrace-train-trifecta-v2 --config configs/train.yaml --max-races 1000 --eval-max-races 1000 --eval-rerank-top-n 10`
3. `boatrace-package-upload`

### `sh/train.sh`

実行内容:

1. `boatrace-train --config configs/train.yaml`
2. `boatrace-train-trifecta-v2 --config configs/train.yaml --max-races 1000 --eval-max-races 1000 --eval-rerank-top-n 10`
3. `boatrace-package-upload`

### `bat\opt.bat`

実行内容:

1. `boatrace-train-trifecta-v2 --config configs/train.yaml --max-races 1000 --eval-max-races 1000 --eval-rerank-top-n 10 --optimize-rerank`
2. `boatrace-package-upload`

### `sh/opt.sh`

実行内容:

1. `boatrace-train-trifecta-v2 --config configs/train.yaml --max-races 1000 --eval-max-races 1000 --eval-rerank-top-n 10 --optimize-rerank`
2. `boatrace-package-upload`

### `bat\eval.bat`

実行内容:

1. `boatrace-eval-trifecta-full --config configs/train.yaml`

### `sh/eval.sh`

実行内容:

1. `boatrace-eval-trifecta-full --config configs/train.yaml`

### `bat\ui.bat`

実行内容:

1. `boatrace-webui`

### `sh/ui.sh`

実行内容:

1. `boatrace-webui`

### `bat\data_download.bat`

実行内容:

1. `boatrace-package-download`

### `sh/data_download.sh`

実行内容:

1. `boatrace-package-download`

### タスクスケジューラ設定のポイント

- `プログラム/スクリプト`
  - `bat\*.bat` を指定
- `開始`
  - `bat` フォルダ、またはリポジトリルート
- 実行ユーザーの環境で `conda.bat` が見つかること

## メモリと処理負荷の目安

### `boatrace-build`

- streaming build に変更済み
- 旧一括 build より OOM しにくい

### `boatrace-train`

- 目安: `24GB` 以上推奨
- `16GB` でも条件次第では可能

### `boatrace-train-trifecta-v2`

- 目安: `32GB` 以上推奨
- `16GB` ではかなり厳しい

### GPU

- 学習時のみ GPU 指定可能
- 推論は CPU
- `CatBoost` は GPU 時に `NDCG` 評価を外す実装
- `LightGBM` だけ GPU にする運用も可能

## 注意

- `artifacts/`, `rowdata/`, `data/` は Git 管理対象外です
- `google_drive_credentials.json` と `artifacts/google-drive-token.json` は Git 管理対象外です
- `.lzh` 展開は `lhafile` を優先し、必要に応じて 7-Zip へフォールバックします
- 初回セットアップ後に依存を更新した場合は、`bat\conda_setup.bat` または `pip install -e .` を再実行してください

## テスト

```bash
python -m pytest -q
```
