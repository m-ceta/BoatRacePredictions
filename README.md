# BoatRacePredictions

ボートレースの出走表・結果データからランキングモデルと三連単 v1 予測を学習し、当日予測・簡易バックテスト・Web UI を提供します。

## 主な機能

- `rowdata` の取得・更新
- `data/processed` への学習用データ生成
- ranker / classifier / 三連単 v1 calibrator の学習
- 当日レースの順位予測・三連単候補出力
- 荒れ度、穴候補、決着パターンの表示
- 直近期間の簡易バックテスト
- Google Drive zip 連携用スクリプト

v2/v3 追加学習、Phase3 rerank 最適化、full trifecta 評価コマンドは削除済みです。学習と予測は v1 を主系統として実行します。

## セットアップ

```bash
bash sh/env_setup.sh
bash sh/open_conda_shell.sh
```

Windows では以下を使用します。

```bat
bat\env_setup.bat
bat\open_conda_shell.bat
```

## 基本コマンド

```bash
boatrace-backfill-rowdata --rowdata rowdata
boatrace-build --rowdata rowdata --output data/processed
boatrace-train --config configs/train.yaml
boatrace-predict-today --config configs/train.yaml --venue 15 --race-no 1
boatrace-predict-today --config configs/train.yaml --venue 15 --race-no 1 --courses 213456
boatrace-predict-today --config configs/train.yaml --venue 15 --race-no 1 --course-overrides 1=2,2=1
boatrace-webui
boatrace-webui-today
```

## 学習パイプライン

Linux:

```bash
bash sh/train_full.sh
bash sh/train_full_resume.sh
```

Windows:

```bat
bat\train_full.bat
bat\train_full_resume.bat
```

## Neural / GPU Setup

`env_setup` はニューラル variant 用の依存関係をデフォルトでインストールします。`PYTORCH_DEVICE` のデフォルトは `auto` です。

CPU版PyTorchを使う場合:

```bash
PYTORCH_DEVICE=cpu bash sh/env_setup.sh
```

CUDA版PyTorchを使う場合:

```bash
PYTORCH_DEVICE=gpu PYTORCH_CUDA_VERSION=cu121 bash sh/env_setup.sh
```

デフォルト動作（`nvidia-smi` がある場合だけGPUを使う）:

```bash
PYTORCH_DEVICE=auto bash sh/env_setup.sh
```

Windows:

```bat
set PYTORCH_DEVICE=gpu
set PYTORCH_CUDA_VERSION=cu121
bat\env_setup.bat
```

`train_full` は以下の順で実行します。

1. Google Drive mount
2. zip からローカル更新
3. `boatrace-build`
4. `boatrace-train`
5. zip を Google Drive へ更新

`train_full_resume` は同じステップを marker ファイルでスキップしながら再開します。

## 主要設定

- `configs/train.yaml`: データ期間、学習期間、モデル、アンサンブル、artifact path を定義します。
- `models.lightgbm_variants`: LightGBM variant の有効化と並列数を定義します。
- `models.xgboost_variants`: XGBoost variant の有効化と並列数を定義します。
- `models.ensemble`: v1 ranker ensemble の探索設定を定義します。
- `artifacts.trifecta_calibrator_path`: 三連単 v1 calibrator の保存先です。

## スクリプト

- `sh/train_full.sh`, `bat/train_full.bat`: フル学習
- `sh/train_full_resume.sh`, `bat/train_full_resume.bat`: 再開可能なフル学習
- `sh/zip_update_local.sh`, `bat/zip_update_local.bat`: Drive zip から復元・rowdata 更新
- `sh/zip_upload.sh`, `bat/zip_upload.bat`: rowdata / data / artifacts を zip 化して Drive に更新
- `sh/full_ui.sh`, `bat/full_ui.bat`: 通常 Web UI
- `sh/today_ui.sh`, `bat/today_ui.bat`: 当日予測 UI
- `notebooks/colab_train_full.ipynb`: Google Colab で `train_full` 相当を実行するNotebook

## 注意

- v2/v3 用の古い artifact が `artifacts` に残っていても、通常の学習・予測経路では読み込まれません。
- パッケージを editable install 済みの環境では、コマンド一覧を更新するために `python -m pip install -e .` を再実行してください。
