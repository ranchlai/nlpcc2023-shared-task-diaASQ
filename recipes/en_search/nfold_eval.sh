#
lang=en
folder="exps/fold"
cd ../data/jsons_$lang
rm train.json
ln -s ./orig/train.json train.json
rm valid.json
ln -s ./orig/valid.json valid.json
cd ../../recipes/en_search
model=model_fused_top3.tar
python ../../main.py \
    --lang $lang \
    --cuda_index 0 \
    --input_files "train valid test" \
    --config $folder/config.yaml \
    --target_dir $folder/outputs \
    --log_file $folder/log.txt \
    --resume_from ./exps/fold/$model \
    --scale_factor 1.0 \
    --action "pred"



# name=`echo $model | awk -F ".tar" '{print $1}'`
# mv ./pred_test_en_no_mask.json ../final_submission/pred_test_en_no_mask_$name.json
# echo saved to ../final_submission/pred_test_en_no_mask_$name.json
