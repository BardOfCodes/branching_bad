import torch as th
import torch.nn as nn
import numpy as np
import time
import os
import _pickle as cPickle
from pathlib import Path
from branching_bad.dataset import DatasetRegistry, Collator, PLADCollator, val_collate_fn
from branching_bad.domain.compiler import CSG2DCompiler
import branching_bad.domain.state_machine as SM
from branching_bad.model import ModelRegistry
from branching_bad.utils.logger import Logger
from branching_bad.utils.metrics import StatEstimator
from branching_bad.utils.beam_utils import batch_beam_decode
from .pretrain import Pretrain


class PLAD(Pretrain):
    def __init__(self, config):
        # load dataset
        # load model
        super(PLAD, self).__init__(config)
        self.dataset_config = config.PLAD.DATASET
        self.inner_patience = config.PLAD.INNER_PATIENCE
        self.outer_patience = config.PLAD.OUTER_PATIENCE
        self.score_tolerance = config.PLAD.SCORE_TOLERANCE
        self.max_inner_iter = config.PLAD.MAX_INNER_ITER

    def start_experiment(self,):
        # Create the dataloaders:
        dl_specs = self.dl_specs
        collator = PLADCollator(self.compiler)
        self.model = self.model.cuda()
        val_loader = th.utils.data.DataLoader(self.val_dataset, batch_size=dl_specs.VAL_BATCH_SIZE, pin_memory=False,
                                              num_workers=dl_specs.VAL_WORKERS, shuffle=False,
                                              persistent_workers=dl_specs.VAL_WORKERS > 0, collate_fn=val_collate_fn)
        original_train_loader = th.utils.data.DataLoader(self.train_dataset, batch_size=dl_specs.SEARCH_BATCH_SIZE, pin_memory=False,
                                                         num_workers=dl_specs.VAL_WORKERS, shuffle=False, collate_fn=val_collate_fn,
                                                         persistent_workers=dl_specs.VAL_WORKERS > 0)
        outer_loop_saturation = False
        outer_iter = 0
        epoch = 0
        while (not outer_loop_saturation):
            # Search for good programs:

            self.model.eval()
            training_dataset = self.generate_training_dataset(
                epoch, original_train_loader)

            # Create data loaders:
            # NOTE: PLAD dataset loader returns 2x the batch size, hence reducing batch size.
            train_loader = th.utils.data.DataLoader(training_dataset, batch_size=dl_specs.TRAIN_BATCH_SIZE // 2, pin_memory=False,
                                                    num_workers=dl_specs.TRAIN_WORKERS, shuffle=False, collate_fn=collator,
                                                    persistent_workers=dl_specs.TRAIN_WORKERS > 0)
            previous_best = self.best_score
            epoch, _ = self.inner_loop(outer_iter, epoch, train_loader, val_loader)
            new_best = self.best_score
            print("Inner loop increased score from {} to {}".format(
                previous_best, new_best))
            # outer saturation check:
            if outer_iter - self.best_outer_iter >= self.outer_patience:
                print("Reached outer saturation.")
                outer_loop_saturation = True
            else:
                print("Outer loop not saturated yet.")

            outer_iter += 1
        predicted_expressions = training_dataset.predicted_expressions
        return predicted_expressions

    def inner_loop(self, outer_iter, epoch, train_loader, val_loader):
        min_inner_iter = 0
        last_improvment_iter = 0
        self.model.train()
        for inner_iter in range(min_inner_iter, self.max_inner_iter):
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
            self.model.eval()
            stat_estimator = self._evaluate(epoch, val_loader)
            self.model.train()

            final_metrics = stat_estimator.get_final_metrics()
            # inner saturation check:
            final_score = final_metrics["score"]
            if final_score > self.best_score + self.score_tolerance:
                print("New best score: ", final_score)
                self.best_score = final_score
                self.best_epoch = epoch
                self.best_outer_iter = outer_iter
                self._save_model(epoch, "best")
                last_improvment_iter = inner_iter
            if inner_iter - last_improvment_iter >= self.inner_patience:
                # hit saturation.
                print("Reached inner saturation.")
                break
                # Save model checkpoint?
            if epoch % self.save_freq == 0:
                self._save_model(epoch)
            epoch += 1
        return epoch, final_metrics

    def generate_training_dataset(self, epoch, original_train_loader):

        # Do beam search on the training set:
        print("Generating training dataset...")
        stat_estimator = self._evaluate(
            epoch, original_train_loader, prefix="search")
        # Now from expressions and corresponding canvases create the new dataset:

        training_dataset = DatasetRegistry.get_dataset(self.dataset_config,
                                                       device=self.train_dataset.device,
                                                       targets=self.train_dataset.targets,
                                                       expressions=stat_estimator.expressions)
        return training_dataset

    def _evaluate(self, epoch, val_loader, log=True, prefix="val"):
        ...
        # Max Batch size will be BS * Beam Size ^ 2

        # TEMPORARY
        st = time.time()
        stat_estimator = StatEstimator()

        # Validation
        nn_interpreter = self.train_dataset.model_translator
        executor = self.train_dataset.executor
        for iter_ind, (canvas, _, _, _) in enumerate(val_loader):
            # model forward:
            with th.no_grad():
                pred_actions = batch_beam_decode(
                    self.model, canvas, self.state_machine_class)
                # mass convert to expressions
                pred_expressions = nn_interpreter.translate_batch(pred_actions)
                # expressions to executions
                pred_canvases = executor.eval_batch_execute(pred_expressions)
                # select best for each based on recon. metric
                # push batch of canvas to stat_estimator
                stat_estimator.eval_batch_execute(
                    pred_canvases, pred_expressions, canvas)
            print(f"Evaluated {iter_ind} batches", "in time", time.time() - st)

        et = time.time()
        final_metrics = stat_estimator.get_final_metrics()
        final_metrics["best_score"] = self.best_score
        final_metrics['time'] = et - st
        final_metrics["inner_iter"] = epoch
        log_iter = epoch * self.epoch_iters
        if log:
            self.logger.log_statistics(final_metrics, log_iter, prefix=prefix)

        return stat_estimator
