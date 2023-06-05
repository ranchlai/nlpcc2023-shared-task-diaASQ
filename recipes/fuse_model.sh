models=`find en_search/exp14/outputs/|grep tar`
echo found models: $models
output="./model_en_fused.tar"
python fuse_models.py --models $models --output $output --top_n 1
