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
        # Since it depends on the location of SIRI Project
        from .abstraction_crafter.cs_crafter import CSCrafter
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

            # Do inner loop PLAD
            original_dsl_score = self.get_dsl_scores(expression_bank)
            cutshort = (self.era == 0)
            expression_bank = super(
                NaiveBOOTAD, self).start_experiment(expression_bank, cutshort=cutshort)

            # Assume the inner loop succeded. Outcome? Expression bank.
            # Perform Merge Splicing on expression bank expressions.
            plad_dsl_score = self.get_dsl_scores(expression_bank)
            print("PLAD step increased score from {} to {}".format(
                original_dsl_score, plad_dsl_score))

            # TODO: Have a clean up step here.
            new_expression_bank, add_macros = self.craft_abstractions(
                expression_bank, self.era, self.executor)
            expression_bank = self.update_expression_banks(
                new_expression_bank, expression_bank)
            # Find macros to remove:
            expression_bank, remove_macros = self.remove_abstractions(
                expression_bank, self.executor, add_macros)
            # Measure number of expr with a macro in them:
            self.update_all_components(add_macros, remove_macros)
            self.log_dsl_details(expression_bank)
            new_dsl_score = self.get_dsl_scores(expression_bank)

            print("Abstraction Crafting step increased score from {} to {}".format(
                plad_dsl_score, new_dsl_score))

            # save:
            if self.era % self.save_freq == 0:
                self.save_with_dsl(self.era, expression_bank)
            if new_dsl_score > original_dsl_score + self.dsl_score_tolerance:
                self.best_dsl_era = self.era
                self.best_dsl_score = new_dsl_score

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
        # just save few things per era:
        diff_items = [era, self.executor, expression_bank]
        save_file = os.path.join(self.save_dir, f"dsl_info_{era}.pkl")
        cPickle.dump(diff_items, open(save_file, "wb"))
        
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
            self.model.load_state_dict(model_weights)
            self.model.cuda()
            all_cmds = self.executor.get_cmd_list()
            self.num_commands = len(all_cmds)
        else:
            era = 0
            expression_bank = None
        return era, expression_bank

    def remove_abstractions(self, expression_bank, executor, add_macros):

        new_expression_bank, remove_macros = self.abstraction_crafter.remove_abstractions(
            expression_bank, executor, add_macros)
        return new_expression_bank, remove_macros

    def craft_abstractions(self, expression_bank, era, executor):

        new_expression_bank, add_macros = self.abstraction_crafter.craft_abstractions(
            expression_bank, era, executor)
        return new_expression_bank, add_macros

    def get_dsl_scores(self, expression_bank):
        # Get DSL Size
        if expression_bank is None:
            return -np.inf
        dsl_size = self.executor.get_dsl_size()
        # avg_task_score = [x['score'] for x in expression_bank]
        # I think this should be with only the "best"
        avg_task_scores = dict()
        for expr in expression_bank:
            score = expr['iou'] + self.abstraction_crafter.length_tax_rate * \
                len(expr['expression'])
            index = expr['target_index']
            if index not in avg_task_scores.keys():
                avg_task_scores[index] = score
            else:
                avg_task_scores[index] = max(avg_task_scores[index], score)
        for i in range(10000):
            if i not in avg_task_scores.keys():
                avg_task_scores[i] = 0
        avg_task_score = np.nanmean(list(avg_task_scores.values()))

        overall_score = self.dsl_weight * dsl_size + self.task_weight * avg_task_score

        return overall_score

    def update_all_components(self, add_macros, remove_macros=None):

        self.executor.update_macros(add_macros, remove_macros)
        embd_selection = self.model_translator.update_macros(
            add_macros, remove_macros)

        self.model.update_macros(embd_selection, add_macros, remove_macros)
        all_cmds = self.executor.get_cmd_list()
        self.num_commands = len(all_cmds)

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

    def log_statistics(self, outer_iter, inner_iter, output, actions, action_validity, n_actions, loss_obj, epoch, iter_ind):
        duration_statistics = {
            "Era": self.era, "Outer": outer_iter, "Inner": inner_iter, "iter": iter_ind}
        log_iter = epoch * self.epoch_iters + iter_ind
        self.logger.log_statistics(
            duration_statistics, log_iter, prefix="duration")
        self._log_loss_statistics(
            output, actions, action_validity, n_actions, loss_obj, epoch, iter_ind)
