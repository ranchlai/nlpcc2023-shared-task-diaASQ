# -*- coding: utf-8 -*-
import argparse
import json
import random


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, default="zh_search/pred_test.json", required=True
    )
    parser.add_argument(
        "--output", type=str, default="zh_search/pred_test_mask.json", required=True
    )
    parser.add_argument("--mask_out_prob", type=float, default=-1, required=True)
    parser.add_argument(
        "--correct_sentiment_by_rules", action="store_true", required=True
    )
    parser.add_argument("--pos_words", type=str, default="pos_words.txt", required=True)
    parser.add_argument("--neg_words", type=str, default="neg_words.txt", required=True)
    parser.add_argument("--lang", type=str, default="en", required=True)

    return parser.parse_args()


sent_map = {"pos": "neg", "neg": "pos", "other": "neg"}


specical_rules = [
    ("power|system|battery", "quickly|fast|drop", "neg"),
    ("battery", "small", "neg"),
    ("battery", "big", "pos"),
    ("*", "hot", "neg"),
]


def check_overlay(str1, str2):
    if str1 == "*":
        return True
    for key in str1.split("|"):
        if key in str2.split() and (
            "not" not in str2.split() and "n't" not in str2.split()
        ):
            return True
    return False


def check_spection_rules(aspect, opinion, sentiment):
    # for rule in specical_rules:
    #     if (
    #         check_overlay(rule[0], aspect)
    #         and check_overlay(rule[1], opinion)
    #         and sentiment != rule[2]
    #     ):
    #         return True, rule[2]
    return False, sentiment


def mask_triple(triple, mask_out_prob):
    if random.random() < mask_out_prob:
        triple[6] = sent_map[triple[6]]
        print("mask out triple: {}".format(triple))

    return triple


def join_aspect_opinion(aspect, opinion, lang="en"):
    if lang == "en":
        return " ".join(aspect.split() + opinion.split())
    else:
        return "".join(aspect.split() + opinion.split())


if __name__ == "__main__":
    args = get_args()
    with open(args.input, "r") as f:
        data = json.load(f)

    if args.mask_out_prob > 0:
        print("warning: using mask_out")

    lang = args.lang

    if args.correct_sentiment_by_rules:
        pos_words = open(args.pos_words).read().split("\n")
        neg_words = open(args.neg_words).read().split("\n")
        pos_words = [w.strip() for w in pos_words if w.strip()]
        neg_words = [w.strip() for w in neg_words if w.strip()]
        for i in range(len(data)):
            trps = data[i]["triplets"]
            for j in range(len(trps)):
                rule_hit, sent = check_spection_rules(
                    trps[j][-2], trps[j][-1], trps[j][6]
                )
                if rule_hit:
                    print("before correct sentiment: {}".format(trps[j]))
                    trps[j][6] = sent
                    print("after correct sentiment: {}".format(trps[j]))
                    print("========Rule===========")
                    continue

                if (
                    join_aspect_opinion(trps[j][-2], trps[j][-1], lang) in pos_words
                    and trps[j][6] == "neg"
                ):
                    print("before correct sentiment: {}".format(trps[j]))
                    trps[j][6] = "pos"
                    print("after correct sentiment: {}".format(trps[j]))
                    print("===================")

                elif (
                    join_aspect_opinion(trps[j][-2], trps[j][-1], lang) in neg_words
                    and trps[j][6] == "pos"
                ):
                    print("before correct sentiment: {}".format(trps[j]))
                    trps[j][6] = "neg"
                    print("after correct sentiment: {}".format(trps[j]))
                    print("===================")

    for i in range(len(data)):
        trps = data[i]["triplets"]
        trps = [mask_triple(t, args.mask_out_prob) for t in trps]
        data[i]["triplets"] = trps

    with open(args.output, "w") as f:
        json.dump(data, f, ensure_ascii=False)

    print("save to {}".format(args.output))
