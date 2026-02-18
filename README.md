# 胸部X線画像の不均衡データに対するサンプリング手法とAUC最適化の相乗効果に関する研究

**Research on Synergistic Effects of Sampling Methods and AUC Optimization for Imbalanced Chest X-ray Image Classification**

修士論文 / Master's Thesis — 高知大学大学院 理工学専攻 情報科学コース 續木研究室

---

## 概要 / Abstract

胸部X線画像の病変分類では陽性例が少なく、クラス不均衡が学習と評価を不安定化させる。本研究は、不均衡対策としてのサンプリングと、LibAUCによるROC-AUC直接最適化を同一枠組みで組み合わせ、AUC最適化学習におけるサンプリング戦略の影響（相互作用）を検証した。

DenseNet121をベースに、ROS・SMOTE・ADASYN等と比率制御（0.5〜1.0）を組み合わせ、単一病変からマルチラベルへ段階的に評価した。

**主な結果:**
- 単一病変では浮腫でROS（比率0.9）がAUCを+0.0251改善
- マルチラベルでは単一病変の知見を直接転用しても集約指標の改善は得られず、ラベル共起などの交絡が影響する可能性を示唆

---

## プロジェクト構成 / Project Structure

```
.
├── src/                          # ソースコード
│   ├── multilabel/               # マルチラベル分類（Phase 3）
│   │   ├── datasets/             # データセット (CheXpert, CXR8)
│   │   ├── losses/               # AUC損失関数
│   │   ├── metrics/              # 評価指標
│   │   ├── samplers/             # サンプリング手法
│   │   │   ├── base.py           # 抽象基底クラス
│   │   │   ├── baseline.py       # Approach 0: ベースライン
│   │   │   ├── binary_relevance.py  # Approach A: Binary Relevance
│   │   │   ├── manifold_mixup.py    # Approach B: Manifold Mixup
│   │   │   ├── weighted_sampler.py  # Approach C: 重み付きサンプリング
│   │   │   └── ml_smote.py          # Approach D: ML-SMOTE
│   │   ├── train_multilabel.py   # マルチラベル学習
│   │   ├── evaluate_multilabel.py # 評価
│   │   └── run_experiments.py    # 実験実行
│   ├── pipelines/                # 学習パイプライン（Phase 2）
│   │   ├── pretrain.py           # 事前学習
│   │   ├── finetune.py           # AUC最適化ファインチューニング
│   │   ├── evaluate.py           # 評価
│   │   └── train_and_evaluate.py # 統合パイプライン
│   ├── utils/                    # ユーティリティ
│   │   ├── config.py             # 設定管理
│   │   ├── experiment_tracker.py # 実験追跡
│   │   └── feasibility.py        # 実行可能性チェック
│   └── analysis/
│       └── statistical_analysis.py # 統計解析
├── scripts/                      # 実験スクリプト
│   ├── run_phase2_experiments.sh        # Phase 2 全実験実行
│   ├── run_all_pathologies_experiments.sh
│   ├── build_multilabel_paper_assets.py # 論文用図表生成（Phase 3）
│   ├── generate_phase2_paper_assets.py  # 論文用図表生成（Phase 2）
│   ├── analyze_sampling_ratios.py       # サンプリング比率分析
│   ├── summarize_evaluation_results.py  # 結果サマリー
│   └── monitor_experiments.py           # 実験モニタリング
├── configs/                      # 設定ファイル
│   ├── config.yaml               # メイン設定
│   └── config_multilabel.yaml    # マルチラベル設定
├── notebooks/                    # Jupyter Notebook
│   └── 07_Optimizing_Multi_Label_AUROC_Loss_with_DenseNet121_on_CheXpert.ipynb
├── results/                      # 論文成果物（図・表）
│   ├── phase2/                   # Phase 2: 単一病変実験
│   │   ├── figures/              # 図（PNG）
│   │   └── tables/               # 表（LaTeX）
│   └── multilabel/               # Phase 3: マルチラベル実験
│       ├── figures/              # 図（PNG）
│       └── tables/               # 表（LaTeX）
├── requirements.txt              # Python依存ライブラリ
└── .gitignore
```

---

## 実験設定 / Experimental Setup

### 対象疾患 / Target Pathologies
| ID | 疾患名 | 英語名 |
|---|---|---|
| 0 | 心肥大 | Cardiomegaly |
| 1 | 浮腫 | Edema |
| 2 | 肺硬化 | Consolidation |
| 3 | 無気肺 | Atelectasis |
| 4 | 胸水 | Pleural Effusion |

### サンプリングアプローチ / Sampling Approaches
| アプローチ | 手法 |
|---|---|
| Approach 0 | ベースライン（サンプリングなし） |
| Approach A | Binary Relevance サンプリング |
| Approach B | Manifold Mixup |
| Approach C | 重み付きサンプリング (Weighted Resampling) |
| Approach D | Multi-Label SMOTE (ML-SMOTE) |

### モデル / Model
- **アーキテクチャ**: DenseNet121（事前学習済み）
- **AUC最適化**: [LibAUC](https://libauc.org/) (AUCMLoss + PESG optimizer)
- **再現性シード**: 42, 123, 456

### データセット / Datasets
- **学習**: CheXpert (Stanford) ― [データセット取得](https://stanfordmlgroup.github.io/competitions/chexpert/)
- **テスト**: CXR8 / ChestX-ray14 (NIH) ― [データセット取得](https://nihcc.app.box.com/v/ChestXray-NIHCC)

> **Note**: データセットはライセンスの関係上このリポジトリに含まれていません。上記リンクから各自取得してください。

---

## セットアップ / Setup

### 1. 環境構築

```bash
# conda環境の作成（推奨）
conda create -n xray_research python=3.10
conda activate xray_research

# 依存ライブラリのインストール
pip install -r requirements.txt
```

### 2. データセットの準備

```bash
# configs/config.yaml の以下を編集
dataset:
  root: "/path/to/CheXpert/CheXpert-v1.0/"
  test_dataset: "/path/to/CXR8/dataset/test_dataset/"
```

### 3. GPU確認

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 実験実行 / Running Experiments

### Phase 2: 単一病変実験

```bash
# 全疾患・全サンプリング手法の実験
bash scripts/run_phase2_experiments.sh

# 単一疾患のみ
bash scripts/run_single_pathology.sh --pathology Edema --method RandomOverSampler --ratio 0.9

# サンプリング比率の網羅実験
bash scripts/run_sampling_ratio_experiments.sh
```

### Phase 3: マルチラベル実験

```bash
# マルチラベル実験の実行
python src/multilabel/run_experiments.py --config configs/config_multilabel.yaml

# 特定アプローチのみ
python src/multilabel/run_experiments.py --approach C --seeds 42 123 456
```

### 結果の可視化

```bash
# Phase 2 論文用図表の生成
python scripts/generate_phase2_paper_assets.py

# Phase 3 論文用図表の生成
python scripts/build_multilabel_paper_assets.py

# 結果サマリー
python scripts/summarize_evaluation_results.py
```

---

## 主な結果 / Key Results

論文で使用した図・表は [results/](results/) ディレクトリに格納されています。

| 実験 | 最良手法 | AUC改善 |
|---|---|---|
| Edema (Phase 2) | ROS (ratio=0.9) | +0.0251 |
| Multi-label (Phase 3) | Approach D (ML-SMOTE) | - |

---

## 依存ライブラリ / Dependencies

| ライブラリ | 用途 |
|---|---|
| PyTorch + torchvision | ディープラーニング |
| LibAUC | AUC最適化 |
| imbalanced-learn | サンプリング手法 (SMOTE等) |
| scikit-learn | 評価指標 |
| pandas, numpy | データ処理 |
| PyYAML | 設定管理 |
| Pillow | 画像処理 |
| tqdm | 進捗表示 |

---

## 引用 / Citation

本コードを研究で利用する場合は以下を引用してください：

```bibtex
@mastersthesis{yamaichi2025,
  author  = {山市 裕也},
  title   = {胸部X線画像の不均衡データに対するサンプリング手法とAUC最適化の相乗効果に関する研究},
  school  = {高知大学大学院 総合人間自然科学研究科 理工学専攻},
  year    = {2025},
}
```

---

## ライセンス / License

本コードはMITライセンスの下で公開されています。
ただし、使用するデータセット（CheXpert・CXR8）については各データセットのライセンスに従ってください。
