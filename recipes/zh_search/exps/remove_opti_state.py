# -*- coding: utf-8 -*-
"""remove optimizer state in torch model"""
import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(
        description="remove optimizer state in torch model"
    )
    parser.add_argument("--model", type=str, help="model path")
    parser.add_argument("--output", type=str, help="output path")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    model = torch.load(args.model)
    if "optimizer" in model:
        model.pop("optimizer", None)
        torch.save(model, args.output)
        # print(model.keys())
        print("model saved to {}".format(args.output))
    else:
        print("no optimizer state in model {}".format(args.model))
