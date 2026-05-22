# BoatRacePredictions

`rowdata` 配下の番組表 `B*.TXT` とレース結果 `K*.TXT` を使い、ボートレースの順位予測と三連単確率予測を行う Python パッケージです。

## 概要

- 1レース6艇を `group` とするランキング学習を採用
- 既存の `CatBoostRanker` / `LightGBM Ranker` / 加重アンサンブルを維持
- 三連単確率は以下の2系統を出力
  - `probability_v1`: 既存の Plackett-Luce ベース
  - `probability_v2`: `win_prob` / `top2_prob` / `top3_prob` と ranking score を組み合わせた拡張版
- `trifecta_isotonic` による三連単確率校正を維持
- オッズがある場合は期待値列を追加できる

## 順位予測と三連単予測の違い

- 順位予測:
  - 各艇の強さをレース内で比較し、`predicted_rank` と `win_probability_like` を出力します。
- 三連単確率:
  - `1-2-3` のような 120 通りの並びごとに確率を出力します。
  - `probability_v1` は既存方式、`probability_v2` は拡張方式です。
- 期待値判定:
  - オッズが与えられた場合のみ、`expected_value = probability * odds` を計算し、`is_value_bet` や `stake_fraction` を追加します。

## ディレクトリ

- `src/parsers`: B/K テキストパーサ
- `src/features`: 学習テーブル生成
- `src/models`: ranker / classifier / flow model
- `src/models/staged.py`: 1着 / 2着 / 3着 の段階モデル
- `src/evaluation`: 三連単・期待値評価指標
- `src/odds`: 期待値計算
- `src/live`: 当日番組表取得と予測
- `configs/train.yaml`: 学習設定

## セットアップ

```bash
pip install -e .
```

## Web UI

Streamlit ベースの Web UI を起動できます。

```bash
boatrace-webui
```

または:

```bash
streamlit run app/streamlit_app.py
```

## データセット生成

```bash
boatrace-build --rowdata rowdata --output data/processed
```

## rowdata 補完

不足している `B*.TXT` / `K*.TXT` を mbrace の日次 `.lzh` から補完できます。

最新既存日付の翌日から本日分までを補完:

```bash
boatrace-backfill-rowdata --rowdata rowdata
```

任意期間を補完:

```bash
boatrace-backfill-rowdata --rowdata rowdata --start 2026-05-14 --end 2026-05-22
```

番組表だけ補完:

```bash
boatrace-backfill-rowdata --rowdata rowdata --start 2026-05-14 --end 2026-05-22 --kinds B
```

生成物:

- `data/processed/race_entries.parquet`
- `data/processed/race_results.parquet`
- `data/processed/training_table.parquet`

## 学習

```bash
boatrace-train --config configs/train.yaml
```

生成物:

- `artifacts/catboost_ranker.cbm`
- `artifacts/lightgbm_ranker.txt`
- `artifacts/classifiers/*.txt`
- `artifacts/flow_lightgbm.txt`
- `artifacts/flow_classes.json`
- `artifacts/staged/*.txt`
- `artifacts/trifecta_v2_logreg.joblib`
- `artifacts/feature_columns.json`
- `artifacts/ensemble_weights.json`
- `artifacts/trifecta_isotonic.joblib`
- `artifacts/metrics.json`

## 予測

順位予測のみ:

```bash
boatrace-predict --config configs/train.yaml --features artifacts/feature_columns.json --input future_races.csv --output ranking_predictions.csv
```

順位予測に加えて三連単確率を出力:

```bash
boatrace-predict --config configs/train.yaml --features artifacts/feature_columns.json --input future_races.csv --output ranking_predictions.csv --trifecta-output trifecta_predictions.csv
```

オッズ付き三連単確率を出力:

```bash
boatrace-predict --config configs/train.yaml --features artifacts/feature_columns.json --input future_races.csv --output ranking_predictions.csv --trifecta-output trifecta_predictions.csv --odds odds.csv
```

`odds.csv` の最低限の想定列:

- `race_id`
- `trifecta`
- `odds`

`trifecta_predictions.csv` には以下の列が入ります。

- `probability_v1`
- `probability_v2`
- `probability`
- `odds` オッズ入力時のみ
- `expected_value` オッズ入力時のみ
- `market_rank` オッズ入力時のみ
- `is_value_bet` オッズ入力時のみ
- `stake_fraction` オッズ入力時のみ

## trifecta v2 のみ再学習

既存の ranker / classifier / calibrator を使い、`flow model`、`staged model`、`trifecta v2 combiner` だけを更新する軽量コマンドです。

```bash
boatrace-train-trifecta-v2 --config configs/train.yaml --max-races 300
```

このコマンドは主に `v2` 改善実験用で、フルの `boatrace-train` より短いサイクルで回す用途を想定しています。

## 当日レース予測

```bash
boatrace-predict-today --config configs/train.yaml --venue 24 --race-no 12
```

## Python API

```python
from src.api import backfill_rowdata_files, load_bundle, predict_ranking, predict_today, predict_trifecta
import pandas as pd

backfill_report = backfill_rowdata_files("rowdata", start_date="2026-05-14", end_date="2026-05-22")
print(backfill_report.to_dict())

bundle = load_bundle("configs/train.yaml")
future_df = pd.read_csv("future_races.csv")

ranking = predict_ranking(bundle, future_df)
trifecta = predict_trifecta(bundle, future_df, top_n=20)

odds_df = pd.read_csv("odds.csv")
trifecta_with_odds = predict_trifecta(bundle, future_df, top_n=20, odds_df=odds_df, use_v2=True)

today = predict_today("24", 12, "configs/train.yaml")
print(today.text)
```

## メトリクス

`artifacts/metrics.json` には以下を保存します。

- `ranker_metrics`
- `trifecta_v1_metrics`
- `trifecta_v2_metrics`
- `classifier_metrics`
- `flow_model_metrics`
- `staged_model_metrics`
- `expected_value_backtest_metrics`

## テスト

```bash
python -m pytest -q
```

## 注意

- 時系列リークを避けるため、特徴量は過去情報のみで作成します。
- Web のオッズ取得やリアルタイム取得は別層で扱い、ローカル予測フローは `B*.TXT` / `K*.TXT` だけで完結する設計です。

## Windows Batch Shortcuts

Weekly batch:

```bat
weekly_update.bat
```

Runs:

- `boatrace-backfill-rowdata --rowdata rowdata`
- `boatrace-build --rowdata rowdata --output data/processed`

Monthly batch:

```bat
monthly_update.bat
```

Runs:

- `boatrace-backfill-rowdata --rowdata rowdata`
- `boatrace-build --rowdata rowdata --output data/processed`
- `boatrace-train --config configs/train.yaml`
- `boatrace-train-trifecta-v2 --config configs/train.yaml --max-races 1000 --eval-max-races 1000 --eval-rerank-top-n 10 --optimize-rerank`
