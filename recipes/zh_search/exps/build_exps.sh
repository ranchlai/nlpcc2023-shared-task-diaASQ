for i in `seq 0 9`;do
mkdir -p exp$i
if [ -f ./exp$i/config.yaml ];then
    echo "config.yaml exists, skip"
    continue
fi
cp ../config_base.yaml ./exp$i/config.yaml
done
