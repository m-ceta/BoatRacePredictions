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
boatrace-train --config configs/train.yaml --skip-evaluation
boatrace-train --config configs/train.yaml --skip-variant-evaluation
boatrace-predict-today --config configs/train.yaml --venue 15 --race-no 1
boatrace-predict-today --config configs/train.yaml --venue 15 --race-no 1 --courses 213456
boatrace-predict-today --config configs/train.yaml --venue 15 --race-no 1 --course-overrides 1=2,2=1
boatrace-webui
boatrace-webui-today
```

## 学習パイプライン

### Top3的中確率の評価

アンサンブルの `valid_calibrated` / `test_calibrated` は、当日予測と同じ確率補正表を適用した後の
120候補を使って、的中率・回収率・Top3的中確率を評価します。
Top3モデルの入力特徴量は学習・当日予測と共通の処理で生成します。同率候補は買い目の数値順で並べ、
UI・正解ラベル・評価の順位を統一します。`valid_raw` / `test_raw` は未校正の比較用であり、
補正後確率で学習したTop3モデルの確率評価は行いません。
各variant単体は本番アンサンブル用の確率補正表・Top3モデルを流用せず、従来どおり単体比較します。

修正前の結果がある環境では、以下で順位モデル・STモデルなどの完了済み学習を再利用できます。

```bash
boatrace-train --config configs/train.yaml --resume --skip-variant-evaluation
```

同率時の扱いをそろえるため、旧版のTop3的中確率モデルだけは再作成し、古い最終評価も再計算します。
`evaluation_version: 2` / `probability_source: adjusted_probability` が更新済み評価の目印です。
`train_full_resume` はtrainステップの完了markerがあるとtrain自体をスキップするため、
評価更新だけが目的の場合は上記コマンドを直接実行してください。
設定・データが変わっていれば通常のチェックポイント判定に従い再学習されます。

なお、Top3モデルの学習レポートは学習データへの当てはまりです。
処理経路の一致だけで未知期間への性能が保証されるわけではありません。

### ST予測の比較実験

`models.st_prediction.enabled: true` の場合、`boatrace-train`（`train_full` を含む）の最後に、
軽量なST回帰モデルと比較用の順位モデルを学習します。現行アンサンブルのweight・当日予測は変更しません。

- STモデルの入力は選手の過去ST、場、艇番、実進入コースなどのレース前情報のみです。本番STは正解としてのみ使用します。
- 比較実験は時系列分割です。train期間の先頭80%を候補学習期間とし、それ以後の日付のvalidレースだけを評価します。
- 順位学習用の予測STは日付単位の時系列OOFで作成します。最初の期間は履歴学習用とし、比較する両順位モデルから共通で除外します。
- 同一レース群・同一設定のLightGBMで、既存proxyのみ／予測ST特徴7項目追加を比較します。既存proxyは残します。
- 初期値は学習候補最大30,000レース、評価最大3,000レース、OOF 3分割、各モデル80反復、CPU 2スレッドです。
- 進入コースや着順が不明・重複するレースは除外し、ST正解の欠損・F/Lなども除外します。実使用件数をレポートへ記録します。

結果は `artifacts/metrics.json` の `st_prediction_experiment` と、
`artifacts/st_prediction_experiment.json` に保存します（保存先は既存metricsと同じディレクトリ）。
STのMAE/RMSE、隣艇より先に出るかの正解率、1位艇的中率、三連単Top1/3/5/10/12/20的中率、
log_loss、追加前後の差分、Top3/Top12の改善・悪化レース数を確認できます。
ログの `delta=+...pp` は的中率の差（パーセントポイント）です。

これは**順位モデル単体から未校正の同一方式で三連単へ変換した比較**であり、
classifier・確率補正を含む本番アンサンブルの改善を保証する評価ではありません。
まずこの差分と標準誤差を確認し、有効なら本番への組み込みを検討してください。
STモデル・両順位モデルは `st_prediction_experiment.joblib` に保存します。
`src.models.st_prediction.predict_experiment_scores` から保存した実験モデルを再利用できます。

`--resume` は完了済みの比較実験を再利用します。`--skip-variant-evaluation` でもこの比較は実行し、
`--skip-evaluation` は実験モデルの学習・保存のみ行います。比較自体が不要なら `enabled: false` にしてください。

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

Skip all post-training metrics and run model training/artifact saving only:

Linux:

```bash
SKIP_TRAIN_EVALUATION=1 bash sh/train_full.sh
SKIP_TRAIN_EVALUATION=1 bash sh/train_full_resume.sh
```

Windows:

```bat
set SKIP_TRAIN_EVALUATION=1
bat\train_full.bat
```

Skip individual model/variant metrics while keeping ensemble metrics:

Linux:

```bash
BOATRACE_TRAIN_SKIP_EVALUATION=1 bash sh/train_full.sh
```

Windows:

```bat
set BOATRACE_TRAIN_SKIP_EVALUATION=1
bat\train_full.bat
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
