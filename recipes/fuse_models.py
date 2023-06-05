# -*- coding: utf-8 -*-
"""fulse a list of model weights"""

import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, nargs="+", default=[])
    parser.add_argument("--output_path", type=str, default="model.pt.tar")
    parser.add_argument("--top_n", type=int, default=3)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    files = []
    # sort the files by score
    # en_search/exp14/outputs/lang-en_epoch53_score0.3749.pth.tar
    print("=========    all models    =========")
    for file in args.models:
        print(file)
    for file in args.models:
        score = float(file.split("score")[-1].split(".pth")[0])
        score = float(file.split("_")[-1].split(".")[1]) / 10000
        files.append((file, float(score)))
    files = sorted(files, key=lambda x: x[1], reverse=True)
    files = files[: args.top_n]
    files = [x[0] for x in files]
    print("========using models========")
    for file in files:
        print(file)
    model0 = torch.load(files[0], map_location="cpu")["model"]
    for i, file in enumerate(files[1:]):
        model = torch.load(file, map_location="cpu")["model"]
        for k in model.keys():
            # only fuse the bias and weight
            if "bias" in k or "weight" in k:
                model0[k] = model0[k] * (i + 1) / (i + 2) + model[k] / (i + 2)

    torch.save({"model": model0}, args.output_path)
    print("file saved to {}".format(args.output_path))
