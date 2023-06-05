#
lang=zh
folders=`find ./exps -maxdepth 2 -mindepth 1 -type d`
folders=exps/exp21
for folder in $folders; do
    echo "target_dir: $folder"
    diff $folder/config.yaml ./config_base.yaml
    python ../../main.py \
        --lang $lang \
        --cuda_index 0 \
        --input_files "train valid test" \
        --config $folder/config.yaml \
        --target_dir $folder/outputs \
        --log_file $folder/log.txt

done
