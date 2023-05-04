import torch as th
from tabulate import tabulate
from torch.utils.tensorboard import SummaryWriter

SIZE_LIMIT = 6

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

        n_entries = len(headers)
        if n_entries > SIZE_LIMIT:
            n_iters = n_entries // SIZE_LIMIT
            for i in range(n_iters+1):
                cur_entries = [entries[0][i * SIZE_LIMIT: (i + 1) * SIZE_LIMIT]]
                cur_headers = headers[i * SIZE_LIMIT: (i + 1) * SIZE_LIMIT]
                if i != 0:
                    cur_entries[0].insert(0, entries[0][0:1])
                    cur_headers = headers[0:1] + cur_headers
                print(tabulate(cur_entries, headers=cur_headers, tablefmt='grid', floatfmt=".3f"))
        else:
            print(tabulate(entries, headers=headers, tablefmt='grid', floatfmt=".3f"))
