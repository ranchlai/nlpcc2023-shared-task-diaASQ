for lang in zh en; do

    python ../prepare_submission.py \
        --input pred_valid_${lang}_no_mask.json \
        --output pred_valid_${lang}_no_mask_rules.json \
        --mask_out_prob -1 \
        --correct_sentiment_by_rules \
        --pos_words ../../data/pos_words_${lang}.txt \
        --neg_words ../../data/neg_words_${lang}.txt \
        --lang ${lang}

    echo "before applying rules"
    python ../../src/run_eval.py -p pred_valid_${lang}_no_mask.json -g ../../data/jsons_${lang}/valid.json
    echo "after applying rules"
    python ../../src/run_eval.py -p pred_valid_${lang}_no_mask_rules.json -g ../../data/jsons_${lang}/valid.json
done
