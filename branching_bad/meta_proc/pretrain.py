import torch as th
import torch.nn as nn
import numpy as np
import time
import os
from pathlib import Path
from branching_bad.dataset import DatasetRegistry, format_train_data
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

        self.save_dir = config.SAVER.DIR
        self.save_freq = config.SAVER.EPOCH

        self.cmd_neg_ceof = config.LOSS.CMD_NEG_COEF

        self.num_commands = config.DOMAIN.NUM_INIT_CMDS
        self.epoch_iters = config.TRAIN.DATASET.EPOCH_SIZE * \
            config.DATA_LOADER.TRAIN_WORKERS // config.DATA_LOADER.BATCH_SIZE

    def start_experiment(self,):
        # Create the dataloaders:
        dl_specs = self.dl_specs
        train_loader = th.utils.data.DataLoader(self.train_dataset, batch_size=dl_specs.BATCH_SIZE, pin_memory=False,
                                                num_workers=dl_specs.TRAIN_WORKERS, shuffle=False, collate_fn=format_train_data,
                                                persistent_workers=dl_specs.TRAIN_WORKERS > 0)

        val_loader = th.utils.data.DataLoader(self.val_dataset, batch_size=dl_specs.BATCH_SIZE, pin_memory=False,
                                              num_workers=dl_specs.VAL_WORKERS, shuffle=False,
                                              persistent_workers=dl_specs.VAL_WORKERS > 0)
        # shift model:
        self.model = self.model.cuda()

        # Train model on the dataset
        for epoch in range(self.start_epoch, self.end_epoch):

            # Train
            for iter_ind, (canvas, actions, target, n_actions) in enumerate(train_loader):

                # model forward:
                output = self.model.forward_train(canvas, actions, n_actions)
                loss, loss_statistics = self._calculate_loss(output, target)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                if iter_ind % self.log_interval == 0:
                    self.log_statistics(
                        output, target, n_actions, loss_statistics, epoch, iter_ind)
            # Evaluate:
            self._evaluate(epoch, val_loader)
            # Save model checkpoint?
            if epoch % self.save_freq == 0:
                self._save_model(epoch)

    def _save_model(self, prefix="best"):
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        th.save(self.model.state_dict(), os.path.join(
            self.save_dir, f"pretrain_{prefix}.pt"))
        # th.save(self.model.state_dict(), os.path.join(
        #     self.save_dir, f"pretrain.pt"))

    def _calculate_loss(self, output, target):
        # calculate loss
        # return loss, loss_statistics
        cmd_logsf, param_logsf = output
        param_logsf = param_logsf.swapaxes(1, 2)
        cmd_distr_target = target[:, 0]
        param_distr_target = target[:, 1:-1]
        cmd_type_flag = target[:, -1:]
        n_predictions = target.shape[1] - 1
        cmd_logsf = cmd_logsf[:, :self.num_commands]
        
        # cmd_onehot = th.nn.functional.one_hot(
        #     cmd_distr_target, num_classes=self.num_commands)
        # cmd_onehot = cmd_onehot.bool()
        # cmd_loss = th.where(cmd_onehot, -cmd_sim,
        #                     self.cmd_neg_ceof * cmd_sim).sum(-1)
        cmd_loss = self.cmd_nllloss(cmd_logsf, cmd_distr_target)

        param_loss_tensor = self.param_nllloss(param_logsf, param_distr_target)
        param_loss = th.mean(param_loss_tensor * cmd_type_flag.float(), dim=-1)

        total_loss = (cmd_loss * 1/n_predictions) + \
            (param_loss * (n_predictions-1)/n_predictions)
        total_loss = th.mean(total_loss)
        stat_obj = {
            "cmd_obj": cmd_loss,
            "param_obj": param_loss,
            "total_obj": total_loss
        }

        return total_loss, stat_obj

    def log_statistics(self, output, target, n_actions, loss_obj, epoch, iter_ind):
        # accuracy
        # input avg. length
        all_stats = {"Epoch": epoch, "Iter": iter_ind}
        cmd_sim, param_distr = output
        cmd_sim = cmd_sim[:, :self.num_commands]
        param_distr = param_distr.swapaxes(1, 2)
        cmd_distr_target = target[:, 0]
        param_distr_target = target[:, 1:-1]
        cmd_type_flag = target[:, -1:]

        cmd_action = th.argmax(cmd_sim, dim=-1)
        match = (cmd_action == cmd_distr_target).float()
        cmd_acc = th.mean(match)

        param_action = th.argmax(param_distr, dim=1)
        match = (param_action == param_distr_target).float()

        param_acc = th.sum(match * cmd_type_flag)/(th.sum(cmd_type_flag) * 5)
        mean_expr_len = th.mean(n_actions.float())

        statistics = {
            "cmd_acc": cmd_acc.item(),
            "param_acc": param_acc.item(),
            "expr_len": mean_expr_len.item()
        }
        all_stats.update(statistics)
        loss_statistics = {
            "cmd_loss": th.mean(loss_obj["cmd_obj"]).item(),
            "param_loss": th.mean(loss_obj["param_obj"]).item(),
            "total_loss": loss_obj["total_obj"].item()
        }
        all_stats.update(loss_statistics)

        log_iter = epoch * self.epoch_iters + iter_ind
        self.logger.log_statistics(all_stats, log_iter, prefix="train")

    def _evaluate(self, epoch, val_loader):
        ...

    def _new_evaluate(self, epoch, val_loader):
        ...
        # Max Batch size will be BS * Beam Size ^ 2

        metric_obj = MetricObj()
        # Validation
        for iter_ind, canvas in enumerate(val_loader):

            # model forward:
            pred_expressions = batch_beam_decode(self.model, canvas)
            metric_obj._calculate_statistics(pred_expressions, canvas)

        final_score = metric_obj.get_final_score()
        if final_score >= self.best_score:
            self.best_score = final_score
            self._save_model("best")
