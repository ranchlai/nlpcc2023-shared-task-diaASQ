# parser.add_argument('--input', type=str, default='zh_search/pred_test.json')
# parser.add_argument('--output', type=str, default='zh_search/pred_test_mask.json')
# parser.add_argument('--mask_out_prob', type=float, default=0.15)

prob=0.10
echo "using prob: $prob"
python prepare_submission.py \
    --input zh_search/pred_test.json \
    --output pred_zh_test_mask.json \
    --mask_out_prob $prob


python prepare_submission.py \
    --input en_search/pred_test.json \
    --output pred_en_test.json \
    --mask_out_prob $prob



python prepare_submission.py \
    --input zh_search/pred_valid.json \
    --output pred_zh_valid.json \
    --mask_out_prob $prob


python prepare_submission.py \
    --input en_search/pred_valid.json \
    --output pred_en_valid.json \
    --mask_out_prob $prob


echo "cuassion: using prob: $prob"
