# -*- coding: utf-8 -*-
"""Compare two json files by doc id and print the differences"""

import argparse
import json


def get_args():
    """Get command line arguments"""
    parser = argparse.ArgumentParser(
        description="Compare two json files by doc id and print the differences"
    )
    parser.add_argument("file1", help="First file to compare")
    parser.add_argument("file2", help="Second file to compare")
    parser.add_argument("truth", help="truth file to load contents from")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    with open(args.file1) as f1, open(args.file2) as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)

    with open(args.truth) as f:
        truth = json.load(f)

    # first = args.file1.split("mask")[-1]
    # second = args.file2.split("mask")[-1]
    first = args.file1
    second = args.file2

    for i, (d1, d2) in enumerate(zip(data1, data2)):
        assert d1["doc_id"] == d2["doc_id"]
        print("==================================")
        print("doc_id", d1["doc_id"])

        print("\n".join(truth[i]["sentences"]))
        t1 = [json.dumps(d, ensure_ascii=False) for d in d1["triplets"]]
        t2 = [json.dumps(d, ensure_ascii=False) for d in d2["triplets"]]
        setdiff1 = set(t1) - set(t2)
        setdiff2 = set(t2) - set(t1)

        if setdiff1 == set() and setdiff2 == set():
            continue

        print(f"[{first}]", setdiff1)
        print(f"[{second}]", setdiff2)
        inboth = set(t2) & set(t1)
        print("in both", inboth)
