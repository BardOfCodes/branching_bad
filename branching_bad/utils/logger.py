import torch as th
from tabulate import tabulate
from torch.utils.tensorboard import SummaryWriter


class Logger:

    def __init__(self, config):
        self.writer = SummaryWriter(config.LOG_DIR)

    def log_statistics(self, statistics, log_iter, prefix):
        for key, value in statistics.items():
            self.writer.add_scalar(f"{prefix}/{key}", value, log_iter)
        entries = [list(statistics.values())]
        entries[0].insert(0, "")
        headers = list(statistics.keys())
        headers.insert(0, prefix)

        print(tabulate(entries, headers=headers, tablefmt='grid', floatfmt=".3f"))
