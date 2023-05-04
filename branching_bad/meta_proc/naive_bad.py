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
from .plad import PLAD
from .abstraction_crafter.cs_crafter import CSCrafter
from branching_bad.domain.csg2d_executor import MacroExecutor
from branching_bad.domain.nn_interpreter import MacroNNInterpreter
from collections import defaultdict
from branching_bad.utils.metrics import StatEstimator


class NaiveBOOTAD(PLAD):
    def __init__(self, config):
        # load dataset
        # load model
        super(NaiveBOOTAD, self).__init__(config)
        # Do inner loop PLAD
        # DO outer loop Abstraction.
        self.best_dsl_score = -np.inf
        self.abstraction_crafter = CSCrafter(config.ABSTRACTION)

        self.dsl_weight = config.ABSTRACTION.DSL_WEIGHT
        self.task_weight = config.ABSTRACTION.TASK_WEIGHT

        self.dsl_score_tolerance = config.ABSTRACTION.DSL_SCORE_TOLERANCE
        self.dsl_patience = config.ABSTRACTION.DSL_PATIENCE
        self.reload_latest = config.ABSTRACTION.RELOAD_LATEST
        self.novel_cmd_weight = config.OBJECTIVE.NOVEL_CMD_WEIGHT

        self.executor = MacroExecutor(
            config.TRAIN.DATASET.EXECUTOR, device="cuda")
        self.model_translator = MacroNNInterpreter(
            config.TRAIN.DATASET.NN_INTERPRETER)

    def start_experiment(self,):
        # SETUP

        outer_loop_saturation = False
        if self.reload_latest:
            self.era, expression_bank = self.load_with_dsl()
        else:
            self.era = 0
            expression_bank = None
        while (not outer_loop_saturation):

            original_dsl_score = self.best_dsl_score
            if self.era != 0:
                # TODO: Have a clean up step here.
                expression_bank = self.clean_expression_bank(expression_bank)
                new_expression_bank, new_macros = self.craft_abstractions(
                    expression_bank, self.era, self.executor)
                expression_bank = self.update_expression_banks(
                    new_expression_bank, expression_bank)
                # Update all
                # Measure number of expr with a macro in them:
                self.log_dsl_details(expression_bank)

                self.update_all_components(new_macros)
                new_dsl_score = self.get_dsl_scores(expression_bank)
                
                
                print("Abstraction Crafting step increased score from {} to {}".format(
                    original_dsl_score, new_dsl_score))
            else:
                new_dsl_score = original_dsl_score

            # save:
            if self.era % self.save_freq == 0:
                self.save_with_dsl(self.era, expression_bank)
            # Do inner loop PLAD
            cutshort = (self.era == 0)
            expression_bank = super(
                NaiveBOOTAD, self).start_experiment(expression_bank, cutshort=cutshort)

            # Assume the inner loop succeded. Outcome? Expression bank.
            # Perform Merge Splicing on expression bank expressions.
            plad_dsl_score = self.get_dsl_scores(expression_bank)
            print("PLAD step increased score from {} to {}".format(
                new_dsl_score, plad_dsl_score))

            if plad_dsl_score > original_dsl_score + self.dsl_score_tolerance:
                self.best_dsl_era = self.era
                self.best_dsl_score = plad_dsl_score

            # outer saturation check:
            if self.era - self.best_dsl_era >= self.dsl_patience:
                print("Reached DSL saturation.")
                outer_loop_saturation = True
            else:
                print("DSL loop not saturated yet.")
            self.era += 1

        dsl_specs = self.get_dsl_specs()
        return dsl_specs

    def log_dsl_details(self, expression_bank):
        count = 0
        novel_cmds = self.executor.parser.get_novel_cmds()
        for expr in expression_bank:
            expression = expr['expression']
            expression = "".join(expression)
            for n_cmd in novel_cmds:
                if n_cmd in expression:
                    count += 1
                    break
        print("Number of expressions with novel cmds: {}".format(count))

        stat_estimator = StatEstimator(length_weight=self.length_tax,
                                       novelty_score_weight=self.novelty_score_weight,
                                       beam_return_count=self.search_beam_return)
        stat_estimator.expression_bank = expression_bank
        expression_stats = stat_estimator.get_expression_stats()
        self.logger.log_statistics(
            expression_stats, self.era, prefix="post-abstraction-expr")
        
        self.executor.parser.describe_all_macros()

    def save_with_dsl(self, era, expression_bank):

        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        item = [era, self.executor, self.model_translator,
                self.model.cpu().state_dict(), expression_bank]
        # save_file = os.path.join(self.save_dir, "dsl_{}.pkl".format(era))
        save_file = os.path.join(self.save_dir, "dsl_0.pkl")
        cPickle.dump(item, open(save_file, "wb"))
        self.model.cuda()
        self._save_model(era)
        self._save_model(era, "best")
        self._save_model(era, prefix="plad")

    def load_with_dsl(self):
        # find the latest era in save_dir:

        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        save_files = os.listdir(self.save_dir)
        save_files = [x for x in save_files if x.startswith("dsl_")]
        save_files = [int(x.split(".")[0].split("_")[1]) for x in save_files]
        save_files = sorted(save_files)
        if len(save_files) > 0:
            latest_era = save_files[-1]
            save_file = os.path.join(
                self.save_dir, "dsl_{}.pkl".format(latest_era))
            item = cPickle.load(open(save_file, "rb"))
            era, self.executor, self.model_translator, model_weights, expression_bank = item

            self.model.cpu()
            # HACK
            all_cmds = self.executor.get_cmd_list()
            self.model.update_cmds(all_cmds)
            self.model.load_state_dict(model_weights)
            self.model.cuda()
        else:
            era = 0
            expression_bank = None
        return era, expression_bank

    def craft_abstractions(self, expression_bank, era, executor):

        new_expression_bank, new_macros = self.abstraction_crafter.craft_abstractions(
            expression_bank, era, executor)
        return new_expression_bank, new_macros

    def get_dsl_scores(self, expression_bank):
        # Get DSL Size
        dsl_size = self.executor.get_dsl_size()
        # avg_task_score = [x['score'] for x in expression_bank]
        avg_task_score = [x['iou'] + self.abstraction_crafter.length_tax_rate *
                          len(x['expression']) for x in expression_bank]
        avg_task_score = np.nanmean(avg_task_score)

        overall_score = self.dsl_weight * dsl_size + self.task_weight * avg_task_score

        return overall_score

    def clean_expression_bank(self, expression_bank):
        return expression_bank

    def update_all_components(self, new_macros):

        self.executor.update_macros(new_macros)
        self.model_translator.update_macros(new_macros)

        all_cmds = self.executor.get_cmd_list()
        self.model.update_cmds(all_cmds)
        self.num_commands = len(all_cmds)
        self.optimizer  # Not required?

        # update class weightage:
        # all_cmds = self.executor.parser.get_cmd_list()
        # novel_cmds = self.executor.parser.get_novel_cmds()
        # class_weights = []
        # for cmd in all_cmds:
        #     if cmd in novel_cmds:
        #         class_weights.append(self.novel_cmd_weight)
        #     else:
        #         class_weights.append(1.0)
        # class_weights = th.from_numpy(np.array(class_weights)).float().cuda()

        # self.cmd_nllloss = th.nn.NLLLoss(reduce=False, weight=class_weights)

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

    def log_statistics(self, outer_iter, inner_iter, output, actions, action_validity, n_actions, loss_obj, epoch, iter_ind):
        duration_statistics = {
            "Era": self.era, "Outer": outer_iter, "Inner": inner_iter, "iter": iter_ind}
        log_iter = epoch * self.epoch_iters + iter_ind
        self.logger.log_statistics(
            duration_statistics, log_iter, prefix="duration")
        self._log_loss_statistics(
            output, actions, action_validity, n_actions, loss_obj, epoch, iter_ind)
