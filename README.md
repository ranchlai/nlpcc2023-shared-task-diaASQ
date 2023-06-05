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
pip install -e .
```



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
<img src="./data/fig_sample.png" width="50%" />
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

## Data Preparation
The data is placed in the `data` directory. The directory structure is as follows:
./data/jsons_zh/train.json
./data/jsons_zh/valid.json


##### Parsed data
Download the parsed data in JSON format from [Google Drive Link](https://drive.google.com/file/d/1MsY8LqbnQ40te-i_OmL5wOT6vQr6PuQi/view?usp=share_link).
Unzip the files and place them under the data directory like the following:

```bash
data/dataset/jsons_zh
data/dataset/jsons_en
```

The dataset currently only includes the train and valid sets. The test set will be released at a later date; refer to [this issue](https://github.com/unikcc/DiaASQ/issues/5#issuecomment-1495612887) for more information.

<!--
##### Build data manually
You can also manually run the scripts to transform the ann and txt format to json format.
1. Download the source data (ann and txt) from [Google Drive Link]
2. Then, unzip the files and place them under the data directory like the following:
```
./data/dataset/annotation_zh
./data/dataset/annotation_en

```
3. Run the following commands, then you will obtain the parsed file in JSON format.
```bash
python src/prepare_data.py
python src/prepare_data.py --lang en
``` -->

## Model Usage

+ Train && Evaluate for Chinese dataset
  ```bash
  bash scripts/train_zh.sh
  ```

+ Train && Evaluate for English dataset
  ```bash
  bash scripts/train_en.sh
  ```

+ If you do not have a `test` set yet, you can run the following command to train and evaluate the model on the `valid` set.
  ```bash
  bash scripts/train_zh_notest.sh
  bash scripts/train_en_notest.sh
  ```

+ GPU memory requirements

| Dataset | Batch size | GPU Memory |
| --- | --- | --- |
| Chinese | 2 |  8GB. |
| English | 2 | 16GB. |

+ Customized hyperparameters:
You can set hyperparameters in `main.py` or `src/config.yaml`, and the former has a higher priority.


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
