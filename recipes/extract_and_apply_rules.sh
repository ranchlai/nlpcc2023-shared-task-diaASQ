
python extract_rules.py --lang zh \
         --top_n 512 \
         --train ../data/jsons_zh/train.json \
          --valid ../data/jsons_zh/train.json \
          --output ../data/


python extract_rules.py --lang en \
    --top_n 512 \
    --train ../data/jsons_en/train.json \
    --valid ../data/jsons_en/train.json \
    --output ../data/


lang=zh
python prepare_submission.py \
    --input final_submission/pred_test_${lang}_model_fused_top3.json \
    --output final_submission/pred_test_${lang}_model_fused_top3_rules.json \
    --mask_out_prob -1 \
    --correct_sentiment_by_rules \
    --pos_words ../data/pos_words_${lang}.txt \
    --neg_words ../data/neg_words_${lang}.txt \
    --lang ${lang}

lang=en
python prepare_submission.py \
    --input final_submission/pred_test_${lang}_model_fused_top3.json \
    --output final_submission/pred_test_${lang}_model_fused_top3_rules.json \
    --mask_out_prob -1 \
    --correct_sentiment_by_rules \
    --pos_words ../data/pos_words_${lang}.txt \
    --neg_words ../data/neg_words_${lang}.txt \
    --lang ${lang}
