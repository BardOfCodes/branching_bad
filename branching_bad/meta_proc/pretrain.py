import torch as th
import torch.nn as nn
import numpy as np
import time
from branching_bad.dataset import DatasetRegistry, format_data
from branching_bad.model import ModelRegistry
from branching_bad.utils.logger import Logger


class Pretrain():
    def __init__(self, config):
        # load dataset
        # load model
        device = config.DEVICE
        self.train_dataset = DatasetRegistry.get_dataset(
            config.TRAIN.DATASET, subset="train", device=device)
        self.val_dataset = DatasetRegistry.get_dataset(
            config.VAL.DATASET, subset="val", device=device)

        self.model = ModelRegistry.create_model(config.MODEL)
        self.optimizer = th.optim.Adam(
            self.model.parameters(), lr=config.TRAIN_SPECS.LR)

        self.logger = Logger(config.LOGGER)
        self.log_interval = config.TRAIN_SPECS.LOG_INTERVAL

        self.start_epoch = 0
        self.end_epoch = config.TRAIN_SPECS.NUM_EPOCHS
        # adjust start_epoch if resuming
        self.dl_specs = config.DATA_LOADER.clone()

        self.cmd_nllloss = th.nn.NLLLoss(reduce=False)
        self.param_nllloss = th.nn.NLLLoss(reduce=False)
        
        

    def start_experiment(self,):
        # Create the dataloaders:
        dl_specs = self.dl_specs
        train_loader = th.utils.data.DataLoader(self.train_dataset, batch_size=dl_specs.BATCH_SIZE, pin_memory=False,
                                            num_workers=dl_specs.NUM_WORKERS, shuffle=False, collate_fn=format_data,
                                            persistent_workers=True)
        
        # val_loader = th.utils.data.DataLoader(self.val_dataset, batch_size=dl_specs.VAL_BATCH_SIZE, pin_memory=False,
        #                                     num_workers=0, shuffle=False, collate_fn=format_data,
        #                                     persistent_workers=True)
        # shift model: 
        self.model = self.model.cuda()
        
        # Train model on the dataset
        for epoch in range(self.start_epoch, self.end_epoch):

            # Train
            end_time = time.time()
            for iter_ind, (canvas, actions, target, n_actions) in enumerate(train_loader):

                # model forward:
                output = self.model.forward_train(canvas, actions, n_actions)
                loss, loss_statistics = self._calculate_loss(output, target)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                statistics = self._calculate_statistics(output, target)

                if iter_ind % self.log_interval == 0:
                    statistics.update(loss_statistics)
                    self.logger.log_statistics(statistics, epoch, iter_ind)
            # Evaluate:
            self._evaluate(epoch)
    
    def _calculate_loss(self, output, target):
        # calculate loss
        # return loss, loss_statistics
        cmd_distr, param_distr = output
        param_distr = param_distr.swapaxes(1, 2)
        cmd_distr_target = target[:, 0]
        param_distr_target = target[:, 1:-1]
        cmd_type_flag = target[:, -1:]
        n_predictions = target.shape[1] - 1
        # cmd_one_hot = th.nn.functional.one_hot(cmd_distr, num_classes=cmd_distr.shape[1])
        cmd_loss = self.cmd_nllloss(cmd_distr, cmd_distr_target)
        avg_cmd_loss = th.mean(cmd_loss, dim=0)
        param_loss_tensor = self.param_nllloss(param_distr, param_distr_target)
        
        # param_loss = th.einsum("ik, i -> i", param_loss_tensor, cmd_type_flag.float())
        param_loss = th.mean(param_loss_tensor * cmd_type_flag.float(), dim=-1)
        
        avg_param_loss = th.mean(param_loss, dim=0)
        
        total_loss = (cmd_loss * 1/n_predictions) + (param_loss * (n_predictions-1)/n_predictions)
        total_loss = th.mean(total_loss)
        statistics = {
            "cmd_loss": avg_cmd_loss.item(),
            "param_loss": avg_param_loss.item(),
            "total_loss": total_loss.item()
        }
        
        return total_loss, statistics
        
        
    def _calculate_statistics(self, output, target):
        statistics = {}
        # accuracy
        # input avg. length
        
        return statistics
        
        
    def _evaluate(self, epoch):
        ...
        # need beam search here.
