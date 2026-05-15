# BoatRacePredictions

`rowdata` 配下の番組表 `B*.TXT` とレース結果 `K*.TXT` を学習データに変換し、ボートレースの着順予想を行うための初期実装です。

## 方針

- 1レース6艇を `group` とするランキング学習
- 目的モデルは `CatBoostRanker`
- 時系列リークを防ぐため、特徴量は各レース日時より前の履歴だけから作成
- 未来レース予想では、番組表相当の出走データを入力にしてレース内順位を返す

## ディレクトリ

- `src/parsers`: B/K テキストパーサ
- `src/features`: 特徴量生成
- `src/models`: 学習と推論
- `src`: CLI と設定ロード
- `configs/train.yaml`: 学習設定

## セットアップ

```bash
pip install -e .
```

## データセット作成

```bash
boatrace-build --rowdata rowdata --output data/processed
```

出力:

- `data/processed/race_entries.parquet`
- `data/processed/race_results.parquet`
- `data/processed/training_table.parquet`

## 学習

```bash
boatrace-train --config configs/train.yaml
```

出力:

- `artifacts/catboost_ranker.cbm`
- `artifacts/lightgbm_ranker.txt`
- `artifacts/feature_columns.json`
- `artifacts/ensemble_weights.json`
- `artifacts/trifecta_isotonic.joblib`
- `artifacts/metrics.json`

## 予想

未来レース用の入力CSVを作成して推論します。列定義は `training_table.parquet` の説明変数と同系です。

```bash
boatrace-predict --config configs/train.yaml --features artifacts/feature_columns.json --input future_races.csv
```

本日開催レースを直接予想する CLI:

```bash
boatrace-predict-today --config configs/train.yaml --venue 大村 --race-no 12
```

## Python API

別の Python プログラムからは `src.api` 経由で直接呼び出せます。

```python
from src.api import load_bundle, predict_ranking, predict_today, predict_trifecta
import pandas as pd

bundle = load_bundle("configs/train.yaml")
future_df = pd.read_csv("future_races.csv")

ranking = predict_ranking(bundle, future_df)
trifecta = predict_trifecta(bundle, future_df, top_n=20)
today = predict_today("大村", 12, "configs/train.yaml")
print(today.text)
```

学習の呼び出し例:

```python
from src.api import train_from_config

metrics = train_from_config("configs/train.yaml")
print(metrics["ensemble"])
```

## 今回の実装範囲

- B/K 形式の基礎パース
- 履歴集計ベースの特徴量
- ランキング学習の骨組み
- ローカルCSV入力による推論CLI

未実装:

- Webからの番組表自動取得
- オッズ統合
- 3連単確率の明示モデル
- フォーマット差分の完全吸収
