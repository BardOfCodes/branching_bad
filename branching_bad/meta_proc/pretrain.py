import torch as th
import torch.nn as nn
import numpy as np
import time
import os
import _pickle as cPickle
from pathlib import Path
from branching_bad.dataset import DatasetRegistry, Collator, val_collate_fn
from branching_bad.domain.compiler import CSG2DCompiler
import branching_bad.domain.state_machine as SM
from branching_bad.model import ModelRegistry
from branching_bad.utils.logger import Logger
from branching_bad.utils.metrics import StatEstimator
from branching_bad.utils.beam_utils import batch_beam_decode


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
        self.length_tax = config.OBJECTIVE.LENGTH_TAX
        self.cmd_entropy_coef = config.OBJECTIVE.CMD_ENTROPY_COEF
        self.param_entropy_coef = config.OBJECTIVE.PARAM_ENTROPY_COEF
        self.weight_decay_coef = config.OBJECTIVE.WEIGHT_DECAY_COEF
        self.score_tolerance = config.OBJECTIVE.SCORE_TOLERANCE
        # USED only for BOOTAD
        self.novelty_score_weight = config.OBJECTIVE.NOVELTY_SCORE_WEIGHT

        self.search_beam_size = config.SEARCH.BEAM_SIZE
        self.search_beam_return = config.SEARCH.BEAM_RETURN
        self.save_dir = config.SAVER.DIR
        self.save_freq = config.SAVER.EPOCH

        # Number of commands: Union, Intersection, Difference, Circle, Rectangle, STOP
        self.num_commands = config.DOMAIN.NUM_INIT_CMDS
        # Number of parameters: trans_x, trans_y, scale_x, scale_y, rot
        self.n_params = config.DOMAIN.NUM_PARAMS
        self.epoch_iters = config.TRAIN.DATASET.EPOCH_SIZE * \
            config.DATA_LOADER.TRAIN_WORKERS // config.DATA_LOADER.TRAIN_BATCH_SIZE

        resolution = config.TRAIN.DATASET.EXECUTOR.RESOLUTION
        self.compiler = CSG2DCompiler(resolution, device)

        self.state_machine_class = getattr(
            SM, config.DOMAIN.STATE_MACHINE_CLASS)

        self.best_score = -np.inf
        self.best_epoch = 0

        self.executor = self.train_dataset.executor
        self.model_translator = self.train_dataset.model_translator

    def start_experiment(self,):
        # Create the dataloaders:
        dl_specs = self.dl_specs
        collator = Collator(self.compiler)

        train_loader = th.utils.data.DataLoader(self.train_dataset, batch_size=dl_specs.TRAIN_BATCH_SIZE, pin_memory=False,
                                                num_workers=dl_specs.TRAIN_WORKERS, shuffle=False, collate_fn=collator,
                                                persistent_workers=dl_specs.TRAIN_WORKERS > 0)

        val_loader = th.utils.data.DataLoader(self.val_dataset, batch_size=dl_specs.VAL_BATCH_SIZE, pin_memory=False,
                                              num_workers=dl_specs.VAL_WORKERS, shuffle=False,
                                              persistent_workers=dl_specs.VAL_WORKERS > 0, collate_fn=val_collate_fn)
        # shift model:
        self.model = self.model.cuda()
        # Train model on the dataset
        self.model.train()
        for epoch in range(self.start_epoch, self.end_epoch):
            for iter_ind, (canvas, actions, action_validity, n_actions) in enumerate(train_loader):

                # model forward:
                output = self.model.forward_train(canvas, actions)
                loss, loss_statistics = self._calculate_loss(
                    output, actions, action_validity)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                if iter_ind % self.log_interval == 0:
                    self.log_statistics(
                        output, actions, action_validity, n_actions, loss_statistics, epoch, iter_ind)
            # Train Epoch over
            self.model.eval()
            stat_estimator = self._evaluate(
                epoch, val_loader, executor=self.executor, model_translator=self.model_translator)
            self.model.train()
            final_metrics = stat_estimator.get_final_metrics()
            # inner saturation check:
            final_score = final_metrics["score"]
            if final_score > self.best_score + self.score_tolerance:
                print("New best score: ", final_score)
                self.best_score = final_score
                self.best_epoch = epoch
                self._save_model(epoch, "best")
            # Save model checkpoint?
            if epoch % self.save_freq == 0:
                self._save_model(epoch)

    def _save_model(self, epoch, prefix="pretrain"):
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        self.model.cpu()
        save_obj = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "score": self.best_score
        }
        file_path = os.path.join(self.save_dir, f"{prefix}_model.pt")
        print(f"saving model at epoch {epoch} at {file_path}")
        cPickle.dump(save_obj, open(file_path, "wb"))
        self.model.cuda()

    def _calculate_loss(self, output, actions, action_validity):
        # calculate loss
        # return loss, loss_statistics
        cmd_logsf, param_logsf = output

        param_logsf = param_logsf.swapaxes(1, 2)

        cmd_distr_target = actions[:, :, 0]
        cmd_distr_target = cmd_distr_target.reshape(-1)
        cmd_validity = action_validity[:, 0]
        cmd_loss = self.cmd_nllloss(cmd_logsf, cmd_distr_target)
        cmd_loss = th.where(cmd_validity, cmd_loss, 0)
        cmd_loss = th.sum(cmd_loss)/th.sum(cmd_validity)

        cmd_p = th.exp(cmd_logsf)
        cmd_entropy = th.sum(-cmd_p * cmd_logsf, dim=-
                             1).sum()/th.sum(cmd_validity)

        param_distr_target = actions[:, :, 1:-1]
        param_distr_target = param_distr_target.reshape(
            -1, self.n_params)
        param_validity = action_validity[:, 1:-1]
        param_loss = self.param_nllloss(param_logsf, param_distr_target)
        param_loss = th.where(param_validity, param_loss, 0)
        # param_loss = th.sum(param_loss, 1)
        param_loss = th.sum(param_loss) / \
            th.sum(param_validity) * self.n_params
        param_p = th.exp(param_logsf)
        param_entropy = th.sum(-param_p * param_logsf,
                               dim=-1).sum()/th.sum(param_validity)

        l2_norms = [th.sum(th.square(w)) for w in self.model.parameters()]
        # divide by 2 to cancel with gradient of square
        l2_norm = sum(l2_norms) / 2
        # total_loss = th.sum(cmd_loss + param_loss)/ (th.sum(cmd_validity) + th.sum(param_validity)/5)
        total_loss = cmd_loss + param_loss + self.cmd_entropy_coef * cmd_entropy + \
            self.param_entropy_coef * param_entropy + self.weight_decay_coef * l2_norm
        stat_obj = {
            "cmd_obj": cmd_loss,
            "param_obj": param_loss,
            "total_obj": total_loss,
            "cmd_entropy": cmd_entropy,
            "param_entropy": param_entropy,
            "l2_norm": l2_norm,
        }

        return total_loss, stat_obj

    def log_statistics(self,  output, actions, action_validity, n_actions, loss_obj, epoch, iter_ind):
        duration_statistics = {"epoch": epoch, "iter": iter_ind, }
        log_iter = epoch * self.epoch_iters + iter_ind
        self.logger.log_statistics(
            duration_statistics, log_iter, prefix="duration")
        self._log_loss_statistics(
            output, actions, action_validity, n_actions, loss_obj, epoch, iter_ind)

    def _log_loss_statistics(self, output, actions, action_validity, n_actions, loss_obj, epoch, iter_ind):
        # accuracy
        # input avg. length
        all_stats = {}
        cmd_sim, param_distr = output
        param_distr = param_distr.swapaxes(1, 2)
        cmd_distr_target = actions[:, :, 0].reshape(-1)
        param_distr_target = actions[:, :, 1:-1]
        param_distr_target = param_distr_target.reshape(
            -1, self.n_params)
        cmd_validity = action_validity[:, 0]
        param_validity = action_validity[:, 1:-1]
        cmd_action = th.argmax(cmd_sim, dim=-1)
        cmd_match = (cmd_action == cmd_distr_target).float()
        cmd_acc = th.sum(th.where(cmd_validity, cmd_match, 0)) / \
            th.sum(cmd_validity)

        param_action = th.argmax(param_distr, dim=1)
        match = (param_action == param_distr_target).float()

        param_acc = th.sum(th.where(param_validity, match, 0)
                           )/th.sum(param_validity)
        mean_expr_len = th.mean(n_actions.float())

        # for novel commands:
        novel_cmds_validity = th.logical_and(
            cmd_distr_target < self.num_commands - 4, cmd_distr_target >= 2)
        novel_acc = th.sum(th.where(novel_cmds_validity,
                           cmd_match, 0))/th.sum(novel_cmds_validity)

        statistics = {
            "cmd_acc": cmd_acc.item(),
            "param_acc": param_acc.item(),
            "novel_acc": novel_acc.item(),
            "n_novel_cmds": th.sum(novel_cmds_validity).item(),
            "expr_len": mean_expr_len.item(),
        }
        all_stats.update(statistics)
        loss_statistics = {
            "cmd_loss": th.mean(loss_obj["cmd_obj"]).item(),
            "cmd_entropy": loss_obj["cmd_entropy"].item(),
            "param_loss": th.mean(loss_obj["param_obj"]).item(),
            "param_entropy": loss_obj["param_entropy"].item(),
            "Weights L2": loss_obj["l2_norm"].item(),
            "total_loss": loss_obj["total_obj"].item(),
        }
        log_iter = epoch * self.epoch_iters + iter_ind
        self.logger.log_statistics(
            all_stats, log_iter, prefix="train-stats")
        self.logger.log_statistics(
            loss_statistics, log_iter, prefix="loss-stats")

    def _evaluate(self, epoch, val_loader, log=True, prefix="val",
                  model_translator=None, executor=None):
        ...
        # Max Batch size will be BS * Beam Size ^ 2

        # TEMPORARY
        st = time.time()
        stat_estimator = StatEstimator(length_weight=self.length_tax,
                                       novelty_score_weight=self.novelty_score_weight,
                                       beam_return_count=self.search_beam_return)

        # Validation
        if model_translator is None:
            model_translator = self.model_translator
        if executor is None:
            executor = self.executor
        stat_estimator.update_novelty_cmds(executor.parser)
        for iter_ind, (canvas, indices) in enumerate(val_loader):
            # model forward:
            with th.no_grad():
                pred_actions = batch_beam_decode(
                    self.model, canvas, self.state_machine_class,
                    n_cmds=self.num_commands, n_params=self.n_params,
                    n_param_cmds=self.num_commands - 4,
                    beam_size=self.search_beam_size)
                # mass convert to expressions
                pred_expressions = model_translator.translate_batch(
                    pred_actions)
                # expressions to executions
                pred_canvases = executor.eval_batch_execute(pred_expressions)
                # select best for each based on recon. metric
                # push batch of canvas to stat_estimator
                stat_estimator.eval_batch_execute(
                    pred_canvases, pred_expressions, canvas, indices)
            print(f"Evaluated {iter_ind} batches", "in time", time.time() - st)

        et = time.time()
        final_metrics = stat_estimator.get_final_metrics()
        stats = {}
        final_metrics["inner_iter"] = epoch
        final_metrics['time'] = et - st
        stats.update(final_metrics)
        stats["best_score"] = self.best_score
        stats["best_epoch"] = self.best_epoch
        log_iter = epoch * self.epoch_iters
        expression_stats = stat_estimator.get_expression_stats()
        if log:
            self.logger.log_statistics(final_metrics, log_iter, prefix=prefix)
            self.logger.log_statistics(expression_stats, log_iter, prefix=prefix + "-expr")
        return stat_estimator

    def load_weights(self, file_name):

        file_path = os.path.join(self.save_dir, f"{file_name}_model.pt")
        print(f"loading model {file_path}")
        save_obj = cPickle.load(open(file_path, "rb"))
        model_weights = save_obj['model']
        self.model.cpu()
        self.model.load_state_dict(model_weights)
        self.model.cuda()
