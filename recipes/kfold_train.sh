if [ $# -ne 1 ]; then
    echo "Usage: $0 lang"
    exit 1
fi

lang=$1
echo "lang: ${lang}"


folder=./fold
for fold in $(seq 0 0); do
    echo "fold: $fold"
    echo data folder "../data/jsons_${lang}"
    if [ -f "../data/jsons_${lang}/train.json" ]; then
        rm ../data/jsons_${lang}/train.json
        rm ../data/jsons_${lang}/valid.json
    fi

    cd ../data/jsons_${lang}
    ln -s train${fold}.json train.json
    ln -s valid${fold}.json valid.json
    cd ../../recipes/${lang}_search

    mkdir -p $folder/$fold/outputs/
    # sort model by score
    score=`find $folder/$fold/outputs/ |grep tar | awk -F "score" '{print $2}'|awk -F ".pth" '{print $1}'|sort -n|tail -n1`

    echo "found score: $score"

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

    # rm ../data/preprocessed/${lang}_notest_hfl-chinese-macbert-large.pkl
    echo "target_dir: $folder"
    mkdir -p $folder/$fold/outputs
    echo "working dir:" `pwd`
    cp ./$folder/config.yaml $folder/$fold/config.yaml
    python ../../main.py \
        --lang ${lang} \
        --cuda_index 0 \
        --input_files "train valid" \
        --config $folder/$fold/config.yaml \
        --target_dir $folder/$fold/outputs \
        --log_file $folder/$fold/log.txt \
        $resume

done
