import torch as th
from tabulate import tabulate
from torch.utils.tensorboard import SummaryWriter

class Logger:
    
    def __init__(self, config):
        self.writer = SummaryWriter(config.LOG_DIR)
        self.header_log_iter = 1000
        
    def log_statistics(self, statistics, epoch, iter_ind):
        for key, value in statistics.items():
            self.writer.add_scalar(key, value, epoch*iter_ind)
        entries = [statistics.values()]
        headers = list(statistics.keys())
        
        print(tabulate(entries, headers=headers, tablefmt='grid'))