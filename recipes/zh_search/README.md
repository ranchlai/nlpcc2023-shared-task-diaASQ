# Parameter search and model fusion for Chinese dataset

## Data Preparation
Run ```./kfold_prepare_data.sh``` to prepare the data for k-fold cross validation.
```bash
python ../prepare_k_fold.py \
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

## Inference
First fuse the models trained in the previous step by running [kfold_fuse_model.sh](kfold_fuse_model.sh).
Then run [recipes/zh_search](recipes/zh_search) to get the final prediction.

The final model is located at [../final_submission/pred_test_zh_no_mask_model_fused_top3.tar.json](../final_submission/pred_test_zh_no_mask_model_fused_top3.tar.json)