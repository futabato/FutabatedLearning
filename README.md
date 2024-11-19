# FutabatedLearning

## プロジェクトの概要

FutabatedLearning は、連合学習（Federated Learning）のアルゴリズム検証を目的としたフレームワークです。
特に、セキュリティの研究者向けに設計されており、連合学習の実験環境を提供することで、効率的に仮説検証に取り組むことができます。

FutabatedLearning は、連合学習の研究を支援するために、以下の機能を提供します：

- マルチホスト環境での連合学習
- より効率的にデータ分析を実施するための Jupyter Notebook テンプレート
- ビザンチン耐性のある集約手法の実装
- ビザンチン攻撃手法の実装

FutabatedLearning は、研究目的での使用を前提としており、以下の範囲で利用可能です：

- 学術研究やプロトタイプ開発
- 連合学習アルゴリズムの評価と比較
- セキュリティ研究における実験的検証

本プロダクトは、商用利用や大規模な実運用を目的としていません。
あくまで研究目的であるため、連合学習の問題設定として一部厳密さを欠いた設計になっています。

## 本プロジェクトの構成

本節では、FutabatedLearning が提供する具体的な機能について説明します。

### ディレクトリ構造

```
FutabatedLearning/
 ├── data/ # データセットを格納するディレクトリ
 ├── notebooks/ # Jupyter Notebook テンプレート
 ├── src/
 │    ├── attack/ # 攻撃手法の実装
 │    └── federatedlearning/ # 連合学習の実装
 │         └── server/
 │              └── aggregations/ # 集約手法の実装
 ├── outputs/ # 実験で生成されるデータの出力先
 └── mlruns/ # MLflow のロギングデータ
```

### 集約手法

以下は、現在 FutabatedLearning に実装されている集約手法とその提案論文の一覧です：

| 集約手法 | 提案論文                                                                                                                                                                        |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| FedAvg   | [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)                                                                   |
| Krum     | [Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent](https://proceedings.neurips.cc/paper_files/paper/2017/file/f4b9ec30ad9f68f89b29639786cb62ef-Paper.pdf) |

### 攻撃手法

以下は、現在 FutabatedLearning に実装されている攻撃手法とその提案論文の一覧です：

| 攻撃手法          | 提案論文                                                                                            |
| ----------------- | --------------------------------------------------------------------------------------------------- |
| Label Flip Attack | [Poisoning Attacks against Support Vector Machines](https://dl.acm.org/doi/10.5555/3042573.3042761) |

## 環境構築

```
docker compose build
```

## Run an experiment

The baseline experiment with MNIST on CNN model using GPU (if `gpu:0` is available)

```
python src/federatedlearning/main.py
```

### Override configuration from the command line

Example

```
python src/federatedlearning/main.py \
    mlflow.run_name=exp001 \
    federatedlearning.num_byzantines=0 federatedlearning.num_clients=10
```

### Parameter Search with Optuna

Example

```
python src/federatedlearning/main.py \
    --multirun 'federatedlearning.num_byzantines=range(8,13)'
```

## Visualize, Search, Compare experiments

```
mlflow ui
```
