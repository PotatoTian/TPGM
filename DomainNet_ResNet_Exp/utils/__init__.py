import logging
import os
from datetime import datetime
import json
import pandas as pd
import torch



def get_logger(logdir, name="cnpt"):
    logger = logging.getLogger(name)
    ts = str(datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    file_path = os.path.join(logdir, "run_{}.log".format(ts))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger


def dump_logs(log_dir, logs):
    try:
        log_path = os.path.join(log_dir, "log_stats.txt")
        with open(log_path, "a") as f:
            f.write(json.dumps(logs) + "\n")
    except Exception:
        print("write failed")

def dump_ratio_csv(log_dir, data):
    log_path = os.path.join(log_dir, "tpgm_constraints.csv")
    df = pd.DataFrame(data)
    try:
        with open(log_path, "a") as f:
            df.to_csv(f, mode="a", index=True, header=False)
    except Exception:
        print("write fail")

def log_constraints(logdir, tpgm, init=False, save=False):
    if logdir is not None:
        if init:
            csv_param = {
                str(i): [name]
                for i, name in enumerate(tpgm.constraints_name)
            }
            dump_ratio_csv(logdir, csv_param)
        if save:
            csv_param = {
                str(i): [tpgm.activation(constraint.mean()).item()]
                for i, constraint in enumerate(tpgm.constraints)
            }
            dump_ratio_csv(logdir, csv_param)


def save_on_master(state, logdir, epoch, is_best):
    if (epoch + 1) % 200 == 0 or is_best:
        if not is_best:
            filename = "{}/ckpt.{}.pth.tar".format(logdir, epoch)
        else:
            filename = "{}/ckpt.best.pth.tar".format(logdir)

        with open(filename, "wb") as h_file:
            torch.save(state, h_file)


class AverageMeter(object):
    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res