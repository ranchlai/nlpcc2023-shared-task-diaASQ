for i in `seq 0 5`;do
mkdir -p ./exps/exp$i
if [ -f ./exps/exp$i/config.yaml ];then
    echo "config_best.yaml exists, skip"
    continue
fi
cp ./config_best.yaml ./exps/exp$i/config.yaml
done
