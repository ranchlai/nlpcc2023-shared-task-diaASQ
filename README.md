<!-- <p align="center"> -->
<!-- </p> -->
# DiaASQ

This repository contains data and code for the first-place solution to the
nlpcc2023 shared task: DiaASQ. See the [project page](https://conasq.pages.dev/results) for more details.

Our solution is a modified version of [DiaASQ](https://github.com/unikcc/DiaASQ)

To clone and install the repository, please run the following command:

```bash
git clone https://github.com/ranchlai/nlpcc2023-shared-task-diaASQ.git
cd nlpcc2023-shared-task-diaASQ
pip install -r requirements.txt
```


To 主办方:
1. 先安装环境，pip install -r requirements.txt
2. 从[Google dirve](https://drive.google.com/file/d/1UoWxWCDS8kjBD6UUHPLZDzrY2aNZ-xeJ/view?usp=drive_link)]下载模型并放在[./recipes/en/model_fused_top3.tar](./recipes/en/model_fused_top3.tar)或[./zh/model_fused_top3.tar](./zh/model_fused_top3.tar)
2. 在 [recipes](./recipes) 下运行 inference_at_once.sh，实现中英文的预测和基于规则的修正
3. 请使用en_with_rules和zh_with_rules作为最终的提交文件， 也可以测试一下en_without_rules和zh_without_rules在测试集的表现（由于提交次数限制，我们没有提交这两个文件）
4. 关于训练/推理/规则更多的细节，请参考[Recipe](./recipes/README.md)



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
