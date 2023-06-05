lang=en
folder=exps/fold

process=`ps aux | grep main.py | grep -v grep | awk '{print $2}'`
echo "process: $process"
while [ ! -z "$process" ]
do
    echo "sleep 30s"
    sleep 30s
    process=`ps aux | grep main.py | grep -v grep | awk '{print $2}'`
    echo "process: $process"
done


for fold in $(seq 3 3); do
    echo "fold: $fold"
    echo data folder "../data/jsons_$lang"
    rm ../data/jsons_$lang/train.json
    rm ../data/jsons_$lang/valid.json
    cd ../data/jsons_$lang
    ln -s full/train${fold}.json train.json
    ln -s full/valid${fold}.json valid.json
    cd ../../recipes/en_search

    echo `pwd`

    # sort model by score
    score=`find $folder/$fold/outputs/ |grep tar | awk -F "score" '{print $2}'|awk -F ".pth" '{print $1}'|sort -n|tail -n1`
    if [ -z "$score" ]; then
        echo "score not found"
        resume=""
    else

        model=`find $folder/$fold/outputs/ |grep $score`
        echo "found model: $model"

        if [ -z "$model" ]; then
            echo "model not found"
            resume=""
        else
            echo "model found"
            resume="--resume_from $model"
        fi
    fi

    # rm ../data/preprocessed/$lang_notest_hfl-chinese-macbert-large.pkl
    echo "target_dir: $folder"
    echo "fold: $fold"
    mkdir -p $folder/$fold/outputs
    echo "working dir:" `pwd`
    python ../../main.py \
        --lang $lang \
        --cuda_index 0 \
        --input_files "train valid" \
        --config $folder/$fold/config.yaml \
        --target_dir $folder/$fold/outputs \
        --log_file $folder/$fold/log.txt \
        $resume

done
