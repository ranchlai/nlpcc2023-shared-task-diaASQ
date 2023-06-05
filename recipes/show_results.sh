# find all subfolders in outputs
folders=$(find exps/fold -mindepth 1 -maxdepth 1 -type d)
# sort the folders by the number in the folder name
folders=$(echo "$folders" | sort -V)
config0=config_best.yaml
for folder in $folders; do
    log=$folder/log.txt
    config=$folder/config.yaml
    echo "============================================================================"
    # exp=$(echo $log | cut -d '/' -f 2)
    echo $folder
    cat  $log |grep "score" |awk -F "- " '{print $2}'
    # diff $config0 $config
    echo "============================================================================"


done
