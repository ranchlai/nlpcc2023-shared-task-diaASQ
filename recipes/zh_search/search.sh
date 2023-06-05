#
lang=zh
folder=./outputs/$RANDOM
mkdir -p $folder
cp config.yaml $folder/config.yaml
echo "target_dir: $folder"
python ../../main.py \
    --lang $lang \
    --cuda_index 0 \
    --input_files "train valid test" \
    --config config.yaml \
    --target_dir  $folder \
    --log_file $folder/log.txt \
