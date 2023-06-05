models=`find exps/fold/*/|grep tar`
# echo found models: $models
# echo found models: $models

top_n=3
echo top_n $top_n
output="./exps/fold/model_fused_top${top_n}.tar"

python ../fuse_models.py --models $models --output $output --top_n ${top_n}
