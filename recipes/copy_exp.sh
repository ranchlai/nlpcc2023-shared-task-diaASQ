src=$1
dst=$2
if [ -f $dst ]; then
    echo "File $dst exists, please remove it first."
    exit 1
fi
mkdir -p $dst
cp $src/config.yaml $dst/config.yaml
