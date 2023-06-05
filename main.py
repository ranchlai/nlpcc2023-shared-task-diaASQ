#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml
from attrdict import AttrDict
from loguru import logger
from torch.optim import AdamW
from tqdm import tqdm
from transformers import (
    get_constant_schedule,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from src.common import ScoreManager, set_seed, update_config
from src.model import BertWordPair
from src.utils import MyDataLoader, RelationMetric

# support multi-gpu training


class Main:
    def __init__(self, args):

        logger.add(args.log_file)
        config = AttrDict(
            yaml.load(
                open(args.config, "r", encoding="utf-8"),
                Loader=yaml.FullLoader,
            )
        )

        for k, v in vars(args).items():
            setattr(config, k, v)

        config = update_config(config)

        set_seed(config.seed)
        if not os.path.exists(args.target_dir):
            os.makedirs(args.target_dir)

        config.device = torch.device(
            "cuda:{}".format(config.cuda_index)
            if torch.cuda.is_available() and config.cuda_index >= 0
            else "cpu"
        )
        self.config = config
        self.best_score = 0
        self.global_epoch = 0
        self.best_epoch = 0

    def train_iter(self):
        self.model.train()

        train_data = tqdm(
            self.trainLoader, total=self.trainLoader.__len__(), file=sys.stdout
        )
        losses = []

        self.model.zero_grad()

        max_len = 0

        for i, data in enumerate(train_data):

            loss, _ = self.model(**data)
            if data["ent_matrix"].shape[1] > max_len:
                max_len = data["ent_matrix"].shape[1]
            losses.append([w.tolist() for w in loss])
            loss_sum = sum(loss)
            loss_sum.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            if (i + 1) % self.config.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()

            ent_loss, rel_loss, pol_loss = np.mean(losses, 0)
            lr1, _ = self.scheduler.get_last_lr()
            description = f"Epoch {self.global_epoch}, \
                lr1: {lr1:.02e}, \
                entity loss:{ent_loss:.5f}, \
                    rel loss: {rel_loss:.5f}, pol loss: {pol_loss:.5f}"
            train_data.set_description(description)
        print("max_len: ", max_len)

    def evaluate_iter(self, dataLoader=None):

        logger.info(f"scale factor: {self.config.scale_factor}")

        if self.global_epoch == 0 and self.config.action == "train":
            logger.info("Evaluate on valid set before training is not allowed")
            # return
        self.model.eval()
        dataLoader = self.validLoader if dataLoader is None else dataLoader
        dataiter = tqdm(dataLoader, total=dataLoader.__len__(), file=sys.stdout)
        for i, data in enumerate(dataiter):
            with torch.no_grad():
                _, (
                    pred_ent_matrix,
                    pred_rel_matrix,
                    pred_pol_matrix,
                ) = self.model(**data)
                self.relation_metric.add_instance(
                    data,
                    pred_ent_matrix,
                    pred_rel_matrix,
                    pred_pol_matrix,
                    self.config.scale_factor,
                )

    def evaluate(self):
        # PATH = os.path.join(self.config.target_dir, "{}_{}.pth.tar").format(
        #     self.config.lang, epoch
        # # )
        # self.model.load_state_dict(
        #     torch.load(model_path, map_location=self.config.device)["model"]
        # )
        self.model.eval()
        self.evaluate_iter(self.validLoader)

        self.relation_metric.save2file(
            self.config.valid_pred_file, self.config.valid_gold_file
        )
        self.relation_metric.compute(
            self.config.valid_pred_file, self.config.valid_gold_file
        )

    def test(self):
        # PATH = os.path.join(self.config.target_dir, "{}_{}.pth.tar").format(
        #     self.config.lang, epoch
        # # )
        # self.model.load_state_dict(
        #     torch.load(model_path, map_location=self.config.device)["model"]
        # )
        self.model.eval()
        self.evaluate_iter(self.testLoader)
        self.relation_metric.save2file(
            self.config.test_pred_file, self.config.test_gold_file
        )

        # result = self.relation_metric.compute(self.config.test_pred_file,
        # self.config.test_pred_file)

        # if action == "eval":
        #     score, res = result
        #     logger.info(
        #         "Evaluate on valid set, micro-F1 score: {:.4f}%".format(score * 100)
        #     )
        #     print(res)

    def resume(self):
        logger.info("Resume from {}".format(self.config.resume_from))
        PATH = self.config.resume_from
        checkpoint = torch.load(PATH, map_location=self.config.device)
        model = checkpoint["model"]
        self.model.load_state_dict(model)
        self.global_epoch = 0
        try:
            self.best_score = 0  # checkpoint["best_score"]
            self.best_epoch = checkpoint["best_epoch"]
            self.global_epoch = checkpoint["best_epoch"] + 1
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        except KeyError:
            logger.info("Optimizer not found in checkpoint")

    def train(self):

        best_score, best_iter = self.best_score, 0

        start_epoch = self.global_epoch
        save_paths = []
        if self.config.resume_from is not None:
            self.evaluate_iter()
        for epoch in range(start_epoch, self.config.epoch_size):

            if epoch >= self.config.freeze_bert_epoch:
                for n, p in self.model.named_parameters():
                    if "bert" in n:
                        p.requires_grad = True

                print("bert unfreeze")
                n_trainable = sum(
                    p.numel() for p in self.model.parameters() if p.requires_grad
                )
                print("n_trainable: ", n_trainable)

            self.global_epoch = epoch
            self.train_iter()
            self.evaluate_iter()

            self.relation_metric.save2file(
                self.config.valid_pred_file, self.config.valid_gold_file
            )
            score, res = self.relation_metric.compute(
                self.config.valid_pred_file, self.config.valid_gold_file
            )

            self.score_manager.add_instance(score, res)
            logger.info(
                "Epoch {}, avearge F1 score: {:.4f}%".format(epoch, score * 100)
            )
            logger.info(res)

            if score > best_score:
                best_score, best_iter = score, epoch
                filename = os.path.join(
                    self.config.target_dir,
                    f"lang-{self.config.lang}_epoch{epoch:02}_score{score:.04f}.pth.tar",  # noqa
                )
                logger.info("saving model to {}".format(filename))
                torch.save(
                    {
                        "epoch": epoch,
                        "model": self.model.cpu().state_dict(),
                        # "optimizer": self.optimizer.state_dict(),
                        "best_score": best_score,
                        "best_epoch": best_iter,
                    },
                    filename,
                )
                if self.config.save_limits > 0:
                    save_paths.append(filename)
                    if len(save_paths) > self.config.save_limits:
                        logger.info("remove old model, {}".format(save_paths[0]))
                        os.remove(save_paths.pop(0))

                self.model.to(self.config.device)
            elif epoch - best_iter > self.config.patience:
                print(
                    "Not upgrade for {} steps, early stopping...".format(
                        self.config.patience
                    )
                )
                break
            self.model.to(self.config.device)

        self.best_iter = best_iter

    def load_param(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = [
            # {
            #     "params": [p for n, p in param_optimizer if "bert" in n],
            #     "lr": float(self.config.bert_lr) ,
            # },
            # {
            #     "params": [
            #         p for n, p in param_optimizer if "bert" not in n
            #     ],
            #     "lr": float(self.config.lr),
            # }
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": float(self.config.weight_decay),
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0,
            },
        ]

        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=float(self.config.lr),
            eps=float(self.config.adam_epsilon),
            weight_decay=float(self.config.weight_decay),
        )
        if self.config.scheduler == "linear_with_warmup":
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=self.config.epoch_size * self.trainLoader.__len__(),
            )
        elif self.config.scheduler == "constant":
            self.scheduler = get_constant_schedule(self.optimizer)
        else:
            self.scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_cycles=self.config.num_cycles,
                num_training_steps=self.config.epoch_size * self.trainLoader.__len__(),
            )

    def forward(self):
        (
            self.trainLoader,
            self.validLoader,
            self.testLoader,
            config,
        ) = MyDataLoader(self.config).getdata()

        self.model = BertWordPair(self.config).to(config.device)
        # warp to multi-GPUs
        # if config.n_gpu > 1:
        #     self.model = nn.DataParallel(self.model)

        self.score_manager = ScoreManager()
        self.relation_metric = RelationMetric(self.config)
        self.load_param()

        # check if need to resume
        if self.config.resume_from is not None:
            self.resume()

        if self.config.action == "eval":
            logger.info("Start to {}...".format(self.config.action))
            self.evaluate()
            return

        if self.config.action == "pred":
            logger.info("Start to {}...".format(self.config.action))
            self.test()
            return

        logger.info("Start training...")

        if (
            self.config.freeze_bert_epoch > 0
            and self.best_epoch < self.config.freeze_bert_epoch
        ):
            for n, p in self.model.named_parameters():
                # print(n)
                if "bert" in n:
                    p.requires_grad = False

        n_trainable = (
            sum([p.numel() for p in self.model.parameters() if p.requires_grad])
            / 1000000000
        )
        # print trainable parameters
        print(
            f"Trainable parameters:{n_trainable} billion",
        )

        self.train()
        logger.info("Training finished..., best epoch is {}...".format(self.best_iter))
        # if "test" in self.config.input_files:
        #     logger.info("Start evaluating...")
        #     self.evaluate(self.best_iter)


#

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--lang",
        type=str,
        default="en",
        choices=["zh", "en"],
        help="language selection",
    )

    parser.add_argument("-c", "--cuda_index", type=int, default=0, help="CUDA index")
    parser.add_argument(
        "-i",
        "--input_files",
        type=str,
        default="train valid test",
        help="input file names",
    )
    parser.add_argument(
        "-a",
        "--action",
        type=str,
        default="train",
        choices=["train", "eval", "pred"],
        help="choose to train, evaluate, or predict",
    )
    parser.add_argument(
        "-b",
        "--best_iter",
        type=int,
        default=0,
        help="best iter to run test, only used when action is eval or pred",
    )
    parser.add_argument("-s", "--seed", type=int, default=42, help="random seed")
    parser.add_argument("--config", type=str, default="config.yaml", help="config file")
    parser.add_argument(
        "--target_dir",
        type=str,
        default="saved_models",
        help="target directory to save models",
    )

    parser.add_argument(
        "--log_file",
        type=str,
        default="log.txt",
    )
    parser.add_argument(
        "--resume_from", type=str, default=None, help="resume from model"
    )
    parser.add_argument("--scale_factor", type=float, default=1.0, help="scale factor")

    args = parser.parse_args()
    main = Main(args)
    main.forward()
