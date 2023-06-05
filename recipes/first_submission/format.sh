for file in `ls *.json`;do
python ../../format_json.py \
    --json_file $file \
    --output $file
done
