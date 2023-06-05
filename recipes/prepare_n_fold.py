# -*- coding: utf-8 -*-
"""Prepare for n fold data cross validation """
import argparse
import json
import random


def get_args():
    parser = argparse.ArgumentParser(
        description="Prepare for n fold data cross validation"
    )
    parser.add_argument(
        "--data", type=str, default="data/jsons_zh", help="path to the data directory"
    )
    parser.add_argument("--n", type=int, default=5, help="number of folds")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--output", type=str, default="data/jsons_zh")
    parser.add_argument("--only_train", action="store_true")

    parser.add_argument("--ratio", type=float, default=0.9)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    print(args)
    print("Prepare for n fold data cross validation")

    data = []

    train_file = "{}/train.json".format(args.data)
    print("Load data from {}".format(train_file))
    with open(train_file, "r") as f:
        data = json.load(f)

    if not args.only_train:
        with open("{}/valid.json".format(args.data), "r") as f:
            data.extend(json.load(f))
            print("Load data from {}".format("{}/valid.json".format(args.data)))

    random.seed(args.seed)

    for i in range(args.n):
        random.shuffle(data)

        train_data = data[: int(len(data) * args.ratio)]
        valid_data = data[int(len(data) * args.ratio) :]
        with open("{}/train{}.json".format(args.output, i), "w") as f:
            json.dump(train_data, f, indent=4, ensure_ascii=False)

        with open("{}/valid{}.json".format(args.output, i), "w") as f:
            json.dump(valid_data, f, indent=4, ensure_ascii=False)

        print("Fold {} done".format(i))
        print("Size of train data: {}".format(len(train_data)))
        print("Size of valid data: {}".format(len(valid_data)))
        print("Save to {}/train{}.json".format(args.output, i))
        print("Save to {}/valid{}.json".format(args.output, i))
