# parser = argparse.ArgumentParser()
#     parser.add_argument("--input", type=str, default="zh_search/pred_test.json",required=True)
#     parser.add_argument("--output", type=str, default="zh_search/pred_test_mask.json",required=True)
#     parser.add_argument("--mask_out_prob", type=float, default=-1,required=True)
#     parser.add_argument("--correct_sentiment_by_rules", action="store_true",required=True)
#     parser.add_argument("--pos_words", type=str, default="pos_words.txt",required=True)
#     parser.add_argument("--neg_words", type=str, default="neg_words.txt",required=True)
#     #

# python ../prepare_submission.py \
#     --input "pred_zh_test_no_mask_0531.json" \
#     --output "pred_zh_test_no_mask_0531_rules.json" \
#     --mask_out_prob -1 \
#     --correct_sentiment_by_rules \
#     --pos_words "../data/pos_words_zh.txt" \
#     --neg_words "../data/neg_words_zh.txt" \
#     --lang zh


python ../prepare_submission.py \
    --input "pred_test_zh_no_mask_model_fused_top3.tar.json" \
    --output "pred_test_zh_no_mask_model_fused_top3_rules.json" \
    --mask_out_prob -1 \
    --correct_sentiment_by_rules \
    --pos_words "../data/pos_words_zh.txt" \
    --neg_words "../data/neg_words_zh.txt" \
    --lang zh



# python ../prepare_submission.py \
#     --input "pred_test_en_no_mask_model_fused_top3.json" \
#     --output "pred_test_en_no_mask_model_fused_top3_rules.json" \
#     --mask_out_prob -1 \
#     --correct_sentiment_by_rules \
#     --pos_words "../data/pos_words_en.txt" \
#     --neg_words "../data/neg_words_en.txt" \
#     --lang en


    # python ../prepare_submission.py \
    # --input "pred_valid_zh_no_mask.json" \
    # --output "pred_valid_zh_no_mask_rules.json" \
    # --mask_out_prob -1 \
    # --correct_sentiment_by_rules \
    # --pos_words "../data/pos_words_zh.txt" \
    # --neg_words "../data/neg_words_zh.txt" \
    # --lang zh


    # python ../prepare_submission.py \
    # --input "pred_valid_en_no_mask.json" \
    # --output "pred_valid_en_no_mask_rules.json" \
    # --mask_out_prob -1 \
    # --correct_sentiment_by_rules \
    # --pos_words "../data/pos_words_en.txt" \
    # --neg_words "../data/neg_words_en.txt" \
    # --lang en
