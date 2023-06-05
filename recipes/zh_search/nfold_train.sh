#!/usr/bin/bash



# watch for the process that run main.py  to finish

process=`ps aux | grep main.py | grep -v grep | awk '{print $2}'`
echo "process: $process"
while [ ! -z "$process" ]
do
    echo "sleep 10s"
    sleep 10s
    process=`ps aux | grep main.py | grep -v grep | awk '{print $2}'`
    echo "process: $process"
done

lang=zh
folder=exps/fold
for fold in $(seq 2 2); do
    echo "fold: $fold"
    rm ../data/jsons_zh/train.json
    rm ../data/jsons_zh/valid.json
    cd ../data/jsons_zh
    ln -s ./full/train${fold}.json train.json
    ln -s ./full/valid${fold}.json valid.json
    cd ../../recipes/zh_search

    # find best model in each fold

    # sort model by score
    score=`find $folder/$fold/outputs/ |grep tar | awk -F "score" '{print $2}'|awk -F ".pth" '{print $1}'|sort -n|tail -n1`
    model=`find $folder/$fold/outputs/ |grep $score`
    echo "found model: $model"

    # rm ../data/preprocessed/zh_notest_hfl-chinese-macbert-large.pkl
    echo "target_dir: $folder"
    echo "fold: $fold"
    mkdir -p $folder/$fold/outputs
    python ../../main.py \
        --lang $lang \
        --cuda_index 0 \
        --input_files "train valid" \
        --config $folder/config.yaml \
        --target_dir $folder/$fold/outputs \
        --log_file $folder/$fold/log.txt \
        --resume_from $model

done
