# BoatRacePredictions

ボートレースの過去 `B*.TXT` / `K*.TXT` から学習し、当日レースや将来レースに対して

- 順位予測
- 三連単確率
- オッズ取得時の買い候補

を出力する Python プロジェクトです。

現在の主な構成は次です。

- 順位予測: `CatBoostRanker` + `LightGBM Ranker`
- 補助モデル: `is_win / is_top2 / is_top3` classifier、`flow`、`staged`
- 三連単予測: `v1` + `Phase3` rerank
- 当日予測: mbrace / boatrace.jp から番組表・オッズを取得
- Web UI: Streamlit

`.lzh` 展開は Python の `lhafile` を優先して使うため、通常は 7-Zip 不要です。

## ディレクトリ

- `rowdata/`
  - 元データ `B*.TXT` / `K*.TXT`
- `data/processed/`
  - 中間生成物
  - `training_table.parquet` など
- `artifacts/`
  - 学習済みモデル、特徴量定義、校正器、メトリクス
- `configs/train.yaml`
  - 学習・推論設定
- `app/streamlit_app.py`
  - Web UI

## セットアップ

### 1. Conda 推奨

```bat
conda_setup.bat
conda activate boatrace-predictions
```

既に環境がある場合も `conda_setup.bat` で更新できます。

環境を有効化したコマンドプロンプトを直接開く場合:

```bat
open_conda_prompt.bat
```

### 2. pip で入れる場合

```bash
pip install -e .
```

## 主なコマンド

### rowdata の不足分を補完

最新既存日の翌日から当日までを補完:

```bash
boatrace-backfill-rowdata --rowdata rowdata
```

期間指定:

```bash
boatrace-backfill-rowdata --rowdata rowdata --start 2026-05-14 --end 2026-05-22
```

`B` のみ:

```bash
boatrace-backfill-rowdata --rowdata rowdata --start 2026-05-14 --end 2026-05-22 --kinds B
```

### 学習データを再生成

```bash
boatrace-build --rowdata rowdata --output data/processed
```

出力:

- `data/processed/race_entries.parquet`
- `data/processed/race_results.parquet`
- `data/processed/training_table.parquet`

### モデル再学習

```bash
boatrace-train --config configs/train.yaml
```

補足:

- `boatrace-build` 後の `training_table.parquet` に入っている最新 `race_date` を見て、
  学習側の `max_date` / `valid_end_date` は実行時に自動同期されます。
- そのため、`rowdata` 更新後に毎回 `configs/train.yaml` の日付を手で直す必要はありません。

### 三連単 `Phase3` 更新

```bash
boatrace-train-trifecta-v2 --config configs/train.yaml --max-races 1000 --eval-max-races 1000 --eval-rerank-top-n 10 --optimize-rerank
```

### 三連単 full valid 評価

```bash
boatrace-eval-trifecta-full --config configs/train.yaml
```

### 特徴量化済み CSV から予測

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

### 当日レース予測

```bash
boatrace-predict-today --config configs/train.yaml --venue 23 --race-no 1
```

補足:

- `future_races.csv` は不要です
- 内部で当日の番組表を取得し、`training_table.parquet` の履歴から特徴量を生成します
- オッズが取得できた場合は、買い候補も出します

### Web UI

```bash
boatrace-webui
```

または:

```bash
streamlit run app/streamlit_app.py
```

### URL を定期的に開く補助コマンド

```bash
```

回数と間隔を指定する場合:

```bash
```

### 学習成果物を zip 化して Google Drive へアップロード

次を 1 回で実行します。

1. `rowdata/` を `rowdata.zip` に固める
2. `artifacts/`, `configs/`, `data/` を `drp.zip` に固める
3. Google Drive の指定フォルダへ同名上書きアップロードする

コマンド:

```bash
boatrace-package-upload
```

既定のアップロード先:

- `https://drive.google.com/drive/folders/19HHxA5r4T_IqMrDyNqU3qRUoRkhT87OL?usp=drive_link`

前提:

- ルートに OAuth クライアント JSON を `google_drive_credentials.json` として置く
- 初回実行時にブラウザ認証を行う
- 認証後のトークンは `artifacts/google-drive-token.json` に保存される

必要なら引数で変更できます。

```bash
boatrace-package-upload --project-root . --drive-folder https://drive.google.com/drive/folders/19HHxA5r4T_IqMrDyNqU3qRUoRkhT87OL?usp=drive_link --credentials google_drive_credentials.json --token artifacts/google-drive-token.json
```

## 入力ファイル

### `future_races.csv`

特徴量化済みの予測入力です。

- 1レースあたり 6 行
- 1行 = 1艇
- `race_id` 単位で 6 艇をまとめる
- 学習時と同じ特徴量列を持つ

通常の当日予測では内部生成されるため、手で作る必要はありません。

### `odds.csv`

最小列は次です。

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

## 当日予測の出力

当日予測では、概ね次を表示します。

- 予想着順
- 各着の予想確率
- 三連単本命
- 三連単予想確率
- 予想信頼度
- オッズ取得時:
  - オッズ評価上位
  - 買い目安オッズ
  - 判定
  - 買い候補一覧

## 主な出力ファイル

### `artifacts/`

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

### 予測専用で別 PC に持っていく最小セット

予測だけなら、概ね次をコピーすれば使えます。

- `artifacts/`
- `configs/`
- `data/processed/training_table.parquet`
- ソースコード一式

### PATH が通っていない環境について

`weekly_update.bat` と `monthly_update.bat` は `boatrace-*` コマンドが `PATH` に無くても動きます。
内部で次の順に Python を探して `src.cli` を直接呼び出します。

1. `.venv\Scripts\python.exe`
2. `venv\Scripts\python.exe`
3. `py -3`
4. `python`

## 運用の考え方

### 日次

1. `boatrace-backfill-rowdata --rowdata rowdata`
2. `boatrace-build --rowdata rowdata --output data/processed`
3. 当日予測

### 週次

1. `rowdata` 補完
2. `data/processed` 再生成

### 月次

1. `rowdata` 補完
2. `data/processed` 再生成
3. `boatrace-train`
4. `boatrace-train-trifecta-v2`
5. 必要に応じて評価

## テスト

```bash
python -m pytest -q
```

## 注意

- `artifacts/` は Git 管理対象外です
- `rowdata/` と `data/` も Git 管理対象外です
- 当日予測と `rowdata` 補完はネットワーク接続が必要です
- 既存環境で `.lzh` 展開を Python 側へ切り替えるには、依存更新のため一度 `conda_setup.bat` または `pip install -e .` を再実行してください
## Web UI で実行できる操作

`boatrace-webui` または `streamlit run app/streamlit_app.py` で起動する Web UI から、次の操作を実行できます。

- 当日レース予測
- `rowdata` 補完
- 学習データ更新
- `boatrace-train`
- `boatrace-train-trifecta-v2`
- `boatrace-package-upload`

### Web UI 上の学習系操作

`boatrace-train` タブ:

- `configs/train.yaml` を指定して ranker / classifier / flow / staged / Phase3 基本モデルを再学習します
- 学習デバイスは `CPU` / `GPU` を選択できます
- Web UI の既定値は `CPU` です

`boatrace-train-trifecta-v2` タブ:

- `max-races`
- `eval-max-races`
- `eval-rerank-top-n`
- `optimize-rerank`
- 学習デバイス `CPU` / `GPU`

を指定して、三連単 `Phase3` rerank の追加最適化を実行します。

`boatrace-package-upload` タブ:

- `rowdata.zip`
- `drp.zip`

を作成し、Google Drive の指定フォルダへ同名上書きアップロードします。

いずれの操作も Web UI 上で実行ログを確認できます。
