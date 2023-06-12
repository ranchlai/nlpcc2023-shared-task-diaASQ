<!-- <p align="center"> -->
<!-- </p> -->
# DiaASQ

This repository contains data and code for the first-place solution to the
nlpcc2023 shared task: DiaASQ. See the [project page](https://conasq.pages.dev/results) for more details.

Our solution is a modified version of [DiaASQ](https://github.com/unikcc/DiaASQ)

## Installation
To clone and install the repository, please run the following command:
```bash
git clone https://github.com/Joint-Laboratory-of-HUST-and-PAIC/nlpcc2023-shared-task-diaASQ.git
cd nlpcc2023-shared-task-diaASQ
conda create -n diaasq python=3.9 -y
conda activate diaasq
pip install -r requirements.txt
```


To 主办方:

1. 请参考[Installation](#installation)安装环境
2. 从 [Google dirve](https://drive.google.com/file/d/1UoWxWCDS8kjBD6UUHPLZDzrY2aNZ-xeJ/view?usp=drive_link)下载模型并解压至recipes下的en和zh目录
3. 在 [recipes](./recipes) 目录下运行 [inference_at_once.sh](./recipes/inference_at_once.sh)，实现中英文的预测和基于规则的修正,并生成如下文件：
- 基于规则修正的[英文预测](recipes/final_submission/pred_test_en_model_fused_top3_rules.json)和 [中文预测](recipes/final_submission/pred_test_zh_model_fused_top3_rules.json)
- 未使用规则修正的 [英文预测](recipes/final_submission/pred_test_en_model_fused_top3.json)和 [中文预测](recipes/final_submission/pred_test_zh_model_fused_top3.json)
4. 请使用带规则的预测作为最终的提交文件 也可以测试一下未使用规则的预测在测试集的表现（我们只观察到基于rules的后处理修正了几个case，但并没有实际提升验证集效果，所以使用rules后处理的效果未知, 希望主办方能够测试一下未使用rules的效果，如果效果接近，我们希望不使用rules）
5. 关于训练/推理/规则更多的细节，请参考[Recipe的README.md](./recipes/README.md)



## News 🎉


## Quick Links
- [Overview](#overview)
- [Requirements](#requirements)
- [Data Preparation](#data-preparation)
- [Model Usage](#model-usage)
- [Citation](#citation)


## Overview
The architecture of our model is shown below:
<center>
<img src="./res/fig_sample.png" width="50%" />
</center>
We modified the baseline in the following aspects:
+ We use the [MacBERT] for both English and Chinese.
+ The English version is transfered from the final Chinese weights to achieve cross-lingual transfer.
+ We modified the loss weigths to make the model more robust.
+ We replaced multi-view interaction with three consecutive multi-head attention modules.
+ Cross-validation is used to select the best model and ensemble the models.


## Requirements

The model is implemented using PyTorch. The versions of the main packages used in our experiments are listed below:ss

+ torch==2.0.1
+ transformers==4.29.1

Install the other required packages:
``` bash
pip install -r requirements.txt
```
We recommend using conda python 3.9 for all experiments.

## Training and Evaluation

See [Recipe](./recipes/README.md) for more details.


## Model Usage
You can download the pretrained model from [Google dirve](https://drive.google.com/file/d/1UoWxWCDS8kjBD6UUHPLZDzrY2aNZ-xeJ/view?usp=drive_link) and put it in [./recipes/en/model_fused_top3.tar](./recipes/en/model_fused_top3.tar) or [./zh/model_fused_top3.tar](./zh/model_fused_top3.tar).
You can do inference with the following command:

```bash
cd recipes
bash kfold_inference.sh zh
bash kfold_inference.sh en
bash extract_and_apply_rules.sh # optional step, apply rules, improvement uknown,
```
+ GPU memory requirements

| Dataset | Batch size | GPU Memory |
| --- | --- | --- |
| Chinese | 1 |  11GB. |
| English | 1 | 11GB. |

In all our experiments, we use a single RTX 3090 12GB.

## Results

Our final submission on the test set achieves the following results(slig):

Chinese:

| Item  | Prec.  | Rec.   | F1     | TP  | Pred. | Gold |
|-------|--------|--------|--------|-----|-------|------|
| Micro | 0.4339 | 0.3431 | 0.3832 | 187 | 431   | 545  |
| Iden  | 0.4988 | 0.3945 | 0.4406 | 215 | 431   | 545  |
| Avg F1|        |        | 0.4119 |     |       |      |


English:

| Item   | Prec.  | Rec.   | F1     | TP  | Pred. | Gold |
|--------|--------|--------|--------|-----|-------|------|
| Micro  | 0.4887 | 0.3871 | 0.4320 | 216 | 442   | 558  |
| Iden   | 0.5226 | 0.4140 | 0.4620 | 231 | 442   | 558  |
| Avg F1 |        |        | 0.4470 |     |       |      |


And the average F1 score for en/zh is 0.4295.

## Citation
If you use our dataset, please cite the following paper:
```
@article{lietal2022arxiv,
  title={DiaASQ: A Benchmark of Conversational Aspect-based Sentiment Quadruple Analysis},
  author={Bobo Li, Hao Fei, Fei Li, Yuhan Wu, Jinsong Zhang, Shengqiong Wu, Jingye Li, Yijiang Liu, Lizi Liao, Tat-Seng Chua, Donghong Ji}
  journal={arXiv preprint arXiv:2211.05705},
  year={2022}
}
```
