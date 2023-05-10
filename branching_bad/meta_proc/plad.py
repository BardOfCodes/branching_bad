from collections import defaultdict
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
        self.max_inner_iter = config.PLAD.MAX_INNER_ITER
        self.max_outer_iter = config.PLAD.MAX_OUTER_ITER
        self.n_expr_per_entry = config.PLAD.N_EXPR_PER_ENTRY

        self.epoch_iters = config.TRAIN.DATASET.EPOCH_SIZE * \
            config.DATA_LOADER.TRAIN_WORKERS // config.DATA_LOADER.TRAIN_BATCH_SIZE * 2

    def start_experiment(self, train_expressions=None, cutshort=False):
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
        if train_expressions is not None:
            training_dataset = DatasetRegistry.get_dataset(self.dataset_config,
                                                            device=self.train_dataset.device,
                                                            targets=self.train_dataset.targets,
                                                            expression_bank=train_expressions,
                                                            executor=self.executor,
                                                            model_translator=self.model_translator,)
        else:
            training_dataset = None
        while (not outer_loop_saturation):
            # Search for good programs:
            self.model.eval()
            training_dataset = self.generate_training_dataset(
                epoch, original_train_loader, training_dataset)

            # Create data loaders:
            # NOTE: PLAD dataset loader returns 2x the batch size, hence reducing batch size.
            train_loader = th.utils.data.DataLoader(training_dataset, batch_size=dl_specs.TRAIN_BATCH_SIZE // 2, pin_memory=False,
                                                    num_workers=dl_specs.TRAIN_WORKERS, shuffle=False, collate_fn=collator,
                                                    persistent_workers=dl_specs.TRAIN_WORKERS > 0)
            previous_best = self.best_score
            if cutshort:
                return training_dataset.expression_bank
            
            epoch, _ = self.inner_loop(
                outer_iter, epoch, train_loader, val_loader)
            # Load previous best weights?
            # self.load_weights(file_name="best")
            new_best = self.best_score
            print("Inner loop increased score from {} to {}".format(
                previous_best, new_best))
            # outer saturation check:
            outer_condition_1 = outer_iter - self.best_outer_iter >= self.outer_patience
            outer_condition_2 = outer_iter > self.max_outer_iter
            if outer_condition_1 or outer_condition_2:
                print("Reached outer saturation.")
                outer_loop_saturation = True
            else:
                print("Outer loop not saturated yet.")

            outer_iter += 1
        expression_bank = training_dataset.expression_bank
        return expression_bank

    def inner_loop(self, outer_iter, epoch, train_loader, val_loader):
        min_inner_iter = 0
        last_improvment_iter = 0
        self.model.train()
        for inner_iter in range(min_inner_iter, self.max_inner_iter):
            st = time.time()
            for iter_ind, (canvas, actions, action_validity, n_actions) in enumerate(train_loader):
                # model forward:
                output = self.model.forward_train(canvas, actions)
                loss, loss_statistics = self._calculate_loss(
                    output, actions, action_validity)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                if iter_ind % self.log_interval == 0:
                    self.log_statistics(outer_iter, inner_iter, 
                        output, actions, action_validity, n_actions, loss_statistics, epoch, iter_ind)
            self.model.eval()
            stat_estimator = self._evaluate(epoch, val_loader, executor=self.executor,
                                            model_translator=self.model_translator)
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
            else:
                print("New score: ", final_score, " is not better than best score: ", self.best_score)
                
            if inner_iter - last_improvment_iter >= self.inner_patience:
                # hit saturation.
                print("Reached inner saturation.")
                break
                # Save model checkpoint?
            if epoch % self.save_freq == 0:
                self._save_model(epoch, prefix="plad")
            et = time.time()
            print("Epoch Time: ", et - st)
            epoch += 1
        return epoch, final_metrics

    def generate_training_dataset(self, epoch, original_train_loader, previous_training_dataset=None):

        print("Generating training dataset...")
        stat_estimator = self._evaluate(epoch, original_train_loader, prefix="search",
                                        executor=self.executor,
                                        model_translator=self.model_translator)
        if not previous_training_dataset is None:
            new_expression_bank = []
            all_expression_bank = defaultdict(list)
            prev_exprs = previous_training_dataset.expression_bank
            new_exprs = stat_estimator.expression_bank
            for cur_expr in prev_exprs:
                target_index = cur_expr['target_index']
                all_expression_bank[target_index].append(cur_expr)
            for cur_expr in new_exprs:
                target_index = cur_expr['target_index']
                all_expression_bank[target_index].append(cur_expr)
            for target_index in all_expression_bank:
                cur_exprs = all_expression_bank[target_index]
                cur_exprs.sort(key=lambda x: x['score'], reverse=True)
                selected_exprs = cur_exprs[:self.n_expr_per_entry]
                new_expression_bank.extend(selected_exprs)
        else:
            new_expression_bank = stat_estimator.expression_bank

        training_dataset = DatasetRegistry.get_dataset(self.dataset_config,
                                                       device=self.train_dataset.device,
                                                       targets=self.train_dataset.targets,
                                                       expression_bank=new_expression_bank,
                                                       executor=self.executor,
                                                       model_translator=self.model_translator,)
        return training_dataset

    def update_expression_banks(self, new_expression_bank, prev_expression_bank):

        updated_expression_bank = []
        all_expression_bank = defaultdict(list)
        unique_expression_bank = defaultdict(set)
        prev_exprs = prev_expression_bank
        new_exprs = new_expression_bank
        for cur_expr in prev_exprs:
            target_index = cur_expr['target_index']
            expr = " ".join(cur_expr['expression'])
            if expr not in unique_expression_bank[target_index]:
                unique_expression_bank[target_index].add(expr)
                all_expression_bank[target_index].append(cur_expr)
        for cur_expr in new_exprs:
            target_index = cur_expr['target_index']
            expr = " ".join(cur_expr['expression'])
            if expr not in unique_expression_bank[target_index]:
                unique_expression_bank[target_index].add(expr)
                all_expression_bank[target_index].append(cur_expr)
        # now remove duplicates:
        
        for target_index in all_expression_bank:
            cur_exprs = all_expression_bank[target_index]
            # deduplicate
            cur_exprs.sort(key=lambda x: x['score'], reverse=True)
            selected_exprs = cur_exprs[:self.n_expr_per_entry]
            updated_expression_bank.extend(selected_exprs)

        return updated_expression_bank

    def log_statistics(self, outer_iter, inner_iter, output, actions, action_validity, n_actions, loss_obj, epoch, iter_ind):
        duration_statistics = {"Outer": outer_iter, "Inner": inner_iter, "iter": iter_ind}
        log_iter = epoch * self.epoch_iters + iter_ind
        self.logger.log_statistics(
            duration_statistics, log_iter, prefix="duration")
        self._log_loss_statistics( output, actions, action_validity, n_actions, loss_obj, epoch, iter_ind)