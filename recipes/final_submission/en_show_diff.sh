python ../show_diff.py pred_test_en_no_mask_model_fused_top3_rules.json pred_test_en_no_mask_0525.json ../data/jsons_en/test.json

# in _0530.json not in _0525.json {'[95, 96, 87, 88, 93, 94, "neg", "k40", "price", "expensive"]', '[36, 37, 34, 35, 38, 39, "pos", "Xiaomi", "Optimization", "weakness"]', '[89, 90, 87, 88, 93, 94, "pos", "neo5", "price", "expensive"]'}
# in _0525.json not in _0530.json {'[36, 37, 34, 35, 38, 39, "neg", "Xiaomi", "Optimization", "weakness"]', '[89, 90, 104, 105, 103, 104, "pos", "neo5", "pictures", "better"]'}
# in both {'[146, 147, 150, 151, 149, 150, "pos", "neo5", "reputation", "good"]'}
