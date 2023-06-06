"""extract pos and neg words from dataset"""
import argparse

import json
from collections import Counter

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="en",required=True)
    parser.add_argument("--top_n", type=int, default=512,required=True)
    parser.add_argument("--train", type=str, default="train.json",required=True)
    parser.add_argument("--valid", type=str, default="valid.json",required=True)
    parser.add_argument("--output", type=str, default="./data/",required=True)
    
    return parser.parse_args()



if __name__ == "__main__":
        
    args = get_args()
    lang = args.lang
    train = args.train
    valid = args.valid
    top_n = args.top_n
    
    with open(train, "r") as f:
        data = json.load(f)
        
    with open(valid, "r") as f:
        data += json.load(f)
            
    good = []
    bad = []
    if lang == "en":
        delimiter = " "
    else:
        delimiter = ""
        
    for d in data:
        good += [delimiter.join(item[-2:]) for item in d["triplets"] if item[6] == 'pos' and item[-2] != '']
        bad += [delimiter.join(item[-2:]) for item in d["triplets"] if item[6] == 'neg' and item[-2] != '']
        
    # keep only top 20
    good = [item[0] for item in Counter(good).most_common(top_n)]
    bad = [item[0] for item in Counter(bad).most_common(top_n)]

    intersection = set(good).intersection(set(bad))
    print("intersection: {}".format(intersection))
    # remove intersection
    good = [item for item in good if item not in intersection]
    bad = [item for item in bad if item not in intersection]


    with open(f"{args.output}/pos_words_{lang}.txt", "w") as fp:
        fp.write("\n".join(good))
        
    with open(f"{args.output}/neg_words_{lang}.txt", "w") as fp:
        fp.write("\n".join(bad))
        
    print(f"file saved to {args.output}/pos_words_{lang}.txt and {args.output}/neg_words_{lang}.txt")