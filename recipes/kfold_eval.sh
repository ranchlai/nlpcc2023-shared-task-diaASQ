
if [ $# -ne 1 ]; then
    echo "Usage: $0 <lang>"
    exit 1
fi
lang=$1
folder="fold"
echo "folder $folder"

if [ -f "../data/jsons_$lang/train.json" ]; then
    rm ../data/jsons_$lang/train.json
    rm ../data/jsons_$lang/valid.json
fi

cd ../data/jsons_$lang
ln -s ./orig/train.json train.json
ln -s ./orig/valid.json valid.json
cd ../../recipes/zh_search

config=$folder/config.yaml
echo using config  $config
model=model_fused_top3.tar

python ../../main.py \
    --lang $lang \
    --cuda_index 0 \
    --input_files "train valid test" \
    --config $config \
    --target_dir $folder/outputs \
    --log_file $folder/log.txt \
    --resume_from ./fold/$model \
    --scale_factor 1.0 \
    --action "eval"
mv ./pred_valid_zh.json ../final_submission/pred_valid_zh_$model.json
echo move pred_valid_zh.json to ../final_submission/pred_valid_zh_$model.json
