
folder="first_submission/"



echo "=======en============"
echo "before masking"
python ../src/run_eval.py -g ./data/jsons_en/valid.json -p ./$folder/pred_valid_en_no_mask.json > $folder/eval_en_no_mask.txt
cat $folder/eval_en_no_mask.txt
f1_en_no_mask=`cat $folder/eval_en_no_mask.txt | grep "Avg F1" |awk -F ": "  '{print $2}'`
echo "after masking"
python ../src/run_eval.py -g ./data/jsons_en/valid.json -p ./$folder/pred_en_valid.json > $folder/eval_en_mask.txt
cat $folder/eval_en_mask.txt
f1_en_mask=`cat $folder/eval_en_mask.txt | grep "Avg F1" | awk -F ': ' '{print $2}'`
echo "mask improvement: `echo "$f1_en_mask - $f1_en_no_mask" | bc`"

echo "=======zh============"
echo "before masking"
python ../src/run_eval.py -g ./data/jsons_zh/valid.json -p ./$folder/pred_valid_zh_no_mask.json > $folder/eval_zh_no_mask.txt
cat $folder/eval_zh_no_mask.txt
f1_zh_no_mask=`cat $folder/eval_zh_no_mask.txt | grep "Avg F1" |awk -F ": "  '{print $2}'`
echo "after masking"
python ../src/run_eval.py -g ./data/jsons_zh/valid.json -p ./$folder/pred_zh_valid.json > $folder/eval_zh_mask.txt
cat $folder/eval_zh_mask.txt
f1_zh_mask=`cat $folder/eval_zh_mask.txt | grep "Avg F1" | awk -F ': ' '{print $2}'`
echo "mask improvement: `echo "$f1_zh_mask - $f1_zh_no_mask" | bc`"


#  1996  python ../src/run_eval.py -g ./data/jsons_zh/valid.json -p ./$folder/pred_zh_valid.json
