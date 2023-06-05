#
lang=zh
folder="exps/fold"
echo "folder $folder"
cd ../data/jsons_$lang
rm train.json
ln -s ./orig/train.json train.json
rm valid.json
ln -s ./orig/valid.json valid.json
cd ../../recipes/zh_search
config=$folder/config_infer.yaml
echo using config  $config
model=model_fused_top3.tar

python ../../main.py \
    --lang $lang \
    --cuda_index 0 \
    --input_files "train valid test" \
    --config $config \
    --target_dir $folder/outputs \
    --log_file $folder/log.txt \
    --resume_from ./exps/fold/$model \
    --scale_factor 1.0 \
    --action "pred"
mv ./pred_test_zh_no_mask.json ../final_submission/pred_test_zh_no_mask_$model.json
echo saved to ../final_submission/pred_test_zh_no_mask_$model.json
