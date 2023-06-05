models=`find ./|grep tar `
for model in $models;do
python remove_opti_state.py --model $model --output $model
done
