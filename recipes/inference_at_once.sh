set -e -x -o pipefail
./kfold_inference.sh zh
./kfold_inference.sh en
./extract_and_apply_rules.sh
