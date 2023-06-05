# find all subfolders in outputs
folders=$(find archs -mindepth 1 -maxdepth 1 -type d)
# sort the folders by the number in the folder name
folders=$(echo "$folders" | sort -V)

config0=./archs/model.py
for folder in $folders; do
    log=$folder/log.txt
    config=$folder/model.py
    echo "============================================================================"
    exp=$(echo $log | cut -d '/' -f 2)
    echo $exp
    cat  $log |grep "micro-F1 score" |awk -F "- " '{print $2}'
    diff $config0 $config
    echo "============================================================================"


done
