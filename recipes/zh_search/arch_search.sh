#
lang=zh
folders=`find ./archs -maxdepth 2 -mindepth 1 -type d`
# sort the folders by the number in the folder name
folders=$(echo "$folders" | sort -V)
for folder in $folders; do
    echo "target_dir: $folder"
    cp $folder/model.py ../../src/model.py
    diff $folder/model.py ../../src/model.py
    python ../../main.py \
        --lang $lang \
        --cuda_index 0 \
        --input_files "train valid test" \
        --config archs/config.yaml \
        --target_dir $folder/outputs \
        --log_file $folder/log.txt

done
