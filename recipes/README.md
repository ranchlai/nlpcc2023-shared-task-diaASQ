# Parameter search and model fusion for Chinese/English dataset



## Environment
set up language environment
```bash
export lang=zh # or en
```

```
## Data Preparation
Run ```./kfold_prepare_data.sh``` to prepare the data for k-fold cross validation.
```bash
python prepare_k_fold.py \
    --data ../../data/jsons_zh/orig/ \
    --k 5 \
    --seed 42 \
    --output ../../data/jsons_zh \
    --ratio 0.95 
    # --only_train # use all data (train and valid) for k-fold
```
It will generate  train and valid files in the ``` ../../data/jsons_zh ``` folder.

## Training
Run [kfold_train.sh](kfold_train.sh)to train the model for k-fold cross validation.
```bash
bash kfold_train.sh $lang
```

## Model fusion
We use top 3 models to create the final model using simple weight averaging
```
bash kfold_fuse_model.sh $lang
```
The final model will be saved to [./$lang/fold/model_fused_top3.tar]
## Inference

You can also download the final model from [Google dirve](https://drive.google.com/file/d/1UoWxWCDS8kjBD6UUHPLZDzrY2aNZ-xeJ/view?usp=drive_link) and put it in [./en/model_fused_top3.tar](./en/model_fused_top3.tar) or [./zh/model_fused_top3.tar](./zh/model_fused_top3.tar).
After downloading the final model, you can run the inference script to generate the prediction file for test set:
```bash
bash kfold_inference.sh $lang 
```

The final data is located at 
- [en_with_rules](final_submission/pred_test_en_no_mask_model_fused_top3_rules.json)
- [zh_with_rules](final_submission/pred_test_zh_no_mask_model_fused_top3_rules.json)
- [en_without_rules](final_submission/pred_test_en_no_mask_model_fused_top3.json)
- [zh_without_rules](final_submission/pred_test_zh_no_mask_model_fused_top3.json)

As the rules did correct some obvious errors（although not seeing any improvement on avearge F1), we use the prediction files with rules for submission.

To 主办方:
1. 先安装环境，pip 
1. 请使用en_with_rules和zh_with_rules作为最终的提交文件
2. 主办方可以测试一下en_without_rules和zh_without_rules在测试集的表现
3. 


## Rules to correct sentiment prediction

As model prediction is not perfect and hard to cover corner cases, we extract rules from training files and apply them to fix the sentiment prediction. 

```bash
bash extrace_and_apply_rules.sh
```
The final files are the following:
- [en_with_rules](final_submission/pred_test_en_no_mask_model_fused_top3_rules.json)
- [zh_with_rules](final_submission/pred_test_zh_no_mask_model_fused_top3_rules.json)
- [en_without_rules](final_submission/pred_test_en_no_mask_model_fused_top3.json)
- [zh_without_rules](final_submission/pred_test_zh_no_mask_model_fused_top3.json)

As the rules did correct some obvious errors（although not seeing any improvement on avearge F1), we use the prediction files with rules for submission.


As we do not have the ground truth for test set, we can not evaluate the performance of the rules. However, we can evaluate the performance of the rules on the validation set . The results are shown in the following table:

|rules | Language | average F1 |
| -------- | -------- | -------- | 
| No | zh | 0.4739 |
| Yes | zh | 0.4739 |
| Yes | en | 0.3900 |
| No | en | 0.3900 |
The rules did not improve the performance on the validation set. However, we can see that the rules can correct some obvious errors. For example, the following examples are corrected by the rules:

```bash
before correct sentiment: [54, 57, 58, 60, 70, 71, 'pos', 'n 10 p', '屏 幕', '差']
after correct sentiment: [54, 57, 58, 60, 70, 71, 'neg', 'n 10 p', '屏 幕', '差']
===================
save to pred_valid_zh_no_mask_rules.json

before correct sentiment: [236, 237, 231, 232, 233, 234, 'neg', 'iPhone', 'experience', 'better']
after correct sentiment: [236, 237, 231, 232, 233, 234, 'pos', 'iPhone', 'experience', 'better']
===================
before correct sentiment: [20, 21, 24, 25, 17, 18, 'neg', 'iPhone', 'processor', 'better']
after correct sentiment: [20, 21, 24, 25, 17, 18, 'pos', 'iPhone', 'processor', 'better']
===================
save to pred_valid_en_no_mask_rules.json
```

Note that the rules are extracted from the training set. Therefore, the rules may not be able to correct all the errors in the test set. The rules are built automatically and we DIDNOT analyze the test set case by case. 