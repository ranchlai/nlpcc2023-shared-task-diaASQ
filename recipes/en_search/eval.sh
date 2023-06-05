#
lang=en
folders="exps/exp14"
for folder in $folders; do
    echo "target_dir: $folder"
    python ../../main.py \
        --lang $lang \
        --cuda_index 0 \
        --input_files "train valid test" \
        --config $folder/config_inference.yaml \
        --target_dir $folder/outputs \
        --log_file $folder/log.txt \
        --action "eval"

done
