# Description: prepare data for k-fold cross validation
if [ $# -ne 1 ]; then
    echo "Usage: $0 lang"
    exit
fi
lang=$1
python prepare_k_fold.py \
    --data ../data/jsons_$lang/orig/ \
    --k 5 \
    --seed 42 \
    --output ../data/jsons_$lang \
    --ratio 0.95
# --only_train # use all data (train and valid) for k-fold
