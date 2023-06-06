if [ $# -ne 1 ]; then
    echo "Usage: $0 <lang>"
    exit
fi
lang=$1
models=`find $lang/fold/*/|grep tar`
if [ -z "$models" ]; then
    echo "models not found"
    echo "run kfold_train.sh first"
    exit
fi
top_n=3
echo top_n $top_n
output="./$lang/model_fused_top${top_n}.tar"

python ./fuse_models.py --models $models --output $output --top_n ${top_n}
