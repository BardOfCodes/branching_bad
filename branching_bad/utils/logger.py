import torch as th
from tabulate import tabulate
from torch.utils.tensorboard import SummaryWriter

class Logger:
    
    def __init__(self, config):
        self.writer = SummaryWriter(config.LOG_DIR)
        
    def log_statistics(self, statistics, log_iter):
        for key, value in statistics.items():
            self.writer.add_scalar(key, value, log_iter)
        entries = [statistics.values()]
        headers = list(statistics.keys())
        
        print(tabulate(entries, headers=headers, tablefmt='grid'))