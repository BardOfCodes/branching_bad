import torch as th
from torch.utils.tensorboard import SummaryWriter

class Logger:
    
    def __init__(self, config):
        self.writer = SummaryWriter(config.LOG_DIR)
        
        
    def log_statistics(self, statistics, epoch, iter_ind):
        print_string = "Epoch: {}, Iter: {}".format(epoch, iter_ind)
        for key, value in statistics.items():
            print_string += ", {}: {:.4f}".format(key, value)
            self.writer.add_scalar(key, value, epoch*iter_ind)
        print(print_string)