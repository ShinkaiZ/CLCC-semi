import os

import sys
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim

from config import get_cfg_defaults
from utils import valid_model, test_model, get_model, train_clcc
from datasets import get_dataset
from lr_scheduler import LR_Scheduler
from helpers import setup_determinism
from losses import get_loss
import warnings

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="",
                        help="config yaml path")
    parser.add_argument("--load", type=str, default="",
                        help="path to model weight")
    parser.add_argument("--valid", action="store_true",
                        help="enable evaluation mode for validation")
    parser.add_argument("--test", action="store_true",
                        help="enable evaluation mode for testset")
    parser.add_argument("-m", "--mode", type=str, default="train",
                        help="model runing mode (train/valid/test)")

    args = parser.parse_args()
    if args.valid:
        args.mode = "valid"
    elif args.test:
        args.mode = "test"

    return args


def setup_logging(args, cfg):
    if not os.path.isdir(cfg.DIRS.LOGS):
        os.mkdir(cfg.DIRS.LOGS)

    head = '{asctime}:{levelname}: {message}'
    handlers = [logging.StreamHandler(sys.stderr), logging.FileHandler(
        os.path.join(cfg.DIRS.LOGS, f'{cfg.EXP}_{cfg.MODEL.NAME}_{args.mode}_fold{cfg.TRAIN.FOLD}.log'),
        mode='a')]
    logging.basicConfig(format=head, style='{', level=logging.DEBUG, handlers=handlers)
    logging.info(f'===============================')
    logging.info(f'\n\nStart with config {cfg}')
    logging.info(f'Command arguments {args}')


def main(args, cfg):
    logging.info(f"=========> {cfg.EXP} <=========")

    # Declare variables
    start_epoch = 0
    best_metric = 0.

    # Create model
    model = get_model(cfg)

    if cfg.MODEL.WEIGHT != "":
        weight = cfg.MODEL.WEIGHT
        model.load_state_dict(torch.load(weight)["state_dict"], strict=False)

    # Define Loss and Optimizer
    train_criterion = get_loss(cfg)
    valid_criterion = get_loss(cfg)

    if cfg.SYSTEM.CUDA:
        model = model.cuda()
        for i in range(len(train_criterion)):
            train_criterion[i] = train_criterion[i].cuda()
        for i in range(len(valid_criterion)):
            valid_criterion[i] = valid_criterion[i].cuda()

    # #optimizer
    if cfg.OPT.OPTIMIZER == "adamw":
        optimizer = optim.AdamW(params=model.parameters(),
                                lr=cfg.OPT.BASE_LR,
                                weight_decay=cfg.OPT.WEIGHT_DECAY)
    elif cfg.OPT.OPTIMIZER == "adam":
        optimizer = optim.Adam(params=model.parameters(),
                               lr=cfg.OPT.BASE_LR,
                               weight_decay=cfg.OPT.WEIGHT_DECAY)
    elif cfg.OPT.OPTIMIZER == "sgd":
        optimizer = optim.SGD(params=model.parameters(),
                              lr=cfg.OPT.BASE_LR,
                              weight_decay=cfg.OPT.WEIGHT_DECAY)
    else:
        raise Exception('OPT.OPTIMIZER should in ["adamw", "adam", "sgd"]')

    # Load checkpoint
    if args.load != "":
        if os.path.isfile(args.load):
            print(f"=> loading checkpoint {args.load}")
            ckpt = torch.load(args.load, "cpu")
            model.load_state_dict(ckpt.pop('state_dict'))
            print("resuming optimizer ...")
            optimizer.load_state_dict(ckpt.pop('optimizer'))
            start_epoch, best_metric = ckpt['epoch'], ckpt['best_metric']
            print('start, best_metric =', start_epoch, best_metric)
            logging.info(
                f"=> loaded checkpoint '{args.load}' (epoch {ckpt['epoch']}, best_metric: {ckpt['best_metric']})")
        else:
            logging.info(f"=> no checkpoint found at '{args.load}'")

    if cfg.SYSTEM.MULTI_GPU:
        model = nn.DataParallel(model)

    # Load data
    train_loader = get_dataset('train', cfg, trainsize=cfg.DATA.SIZE)
    valid_loader = get_dataset('valid', cfg, trainsize=cfg.DATA.SIZE)
    test_loader = get_dataset('test', cfg, trainsize=cfg.DATA.SIZE)

    # Load scheduler
    scheduler = LR_Scheduler("cos", cfg.OPT.BASE_LR, cfg.TRAIN.EPOCHS, iters_per_epoch=len(train_loader),
                             warmup_epochs=cfg.OPT.WARMUP_EPOCHS)

    if args.mode == "train":
        train_dict = {'CLCC': train_clcc}
        train_dict[cfg.TRAIN.METHOD](logging.info, cfg, model, train_loader, valid_loader, train_criterion,
                                     valid_criterion, optimizer, scheduler, start_epoch, best_metric, test_loader)
    elif args.mode == "valid":
        valid_model(logging.info, cfg, model, valid_criterion, valid_loader)
    else:
        test_model(logging.info, cfg, model, test_loader, weight=cfg.MODEL.WEIGHT)


if __name__ == "__main__":

    args = parse_args()
    cfg = get_cfg_defaults()

    if args.config != "":
        cfg.merge_from_file(args.config)
    cfg.freeze()

    for _dir in ["WEIGHTS", "OUTPUTS"]:
        if not os.path.isdir(cfg.DIRS[_dir]):
            os.mkdir(cfg.DIRS[_dir])

    setup_logging(args, cfg)
    setup_determinism(cfg.SYSTEM.SEED)
    main(args, cfg)
