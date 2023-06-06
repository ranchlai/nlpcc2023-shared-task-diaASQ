
if [ $# -ne 1 ]; then
    echo "Usage: $0 <lang>"
    exit 1
fi
lang=$1
folder="${lang}/fold"
echo "folder $folder"

model=model_fused_top3.tar

# check if model exists
if [ ! -f ${lang}/${model} ]; then
    echo "model not found, train the model or download the model first"
    exit 1
fi

# remove the soft link to the k-fold data
if [ -f "../data/jsons_${lang}/train.json" ]; then
    rm ../data/jsons_${lang}/train.json
    rm ../data/jsons_${lang}/valid.json
fi

# create soft link to the original data
cd ../data/jsons_${lang}
ln -s ./orig/train.json train.json
ln -s ./orig/valid.json valid.json
cd ../../recipes/

config=$folder/config.yaml
echo using config  $config

python ../main.py \
    --lang ${lang} \
    --cuda_index 0 \
    --input_files "train valid test" \
    --config $config \
    --target_dir $folder/outputs \
    --log_file $folder/log.txt \
    --resume_from ${lang}/${model} \
    --scale_factor 1.0 \
    --action "pred"

# remove "tar" in model name
model=`echo ${model} | awk -F ".tar" '{print $1}'`
mv ./pred_test_${lang}.json ./final_submission/pred_test_${lang}_${model}.json
echo move pred_test_${lang}.json to ./final_submission/pred_test_${lang}_${model}.json
