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
from .naive_bad import NaiveBOOTAD
from branching_bad.domain.csg2d_executor import MacroExecutor
from branching_bad.domain.nn_interpreter import MacroNNInterpreter
from collections import defaultdict
from branching_bad.utils.metrics import StatEstimator


class BranchingBAD(NaiveBOOTAD):

    def __init__(self, config):

        super(BranchingBAD, self).__init__(config)
        # Additional parameters:
        self.n_branches = config.N_BRANCHES
        self.dummy_era = 10000
        self.best_scores = [-np.inf for x in range(self.n_branches)]
        self.best_epochs = [0 for x in range(self.n_branches)]

    def start_experiment(self,):
        outer_loop_saturation = False
        if self.reload_latest:
            self.era, expression_bank = self.load_with_dsl(self.era)
        else:
            self.era = 0
            expression_bank = None
        while (not outer_loop_saturation):
            # load the data:
            _, expression_bank = self.load_with_dsl(self.era)
            original_dsl_score = self.get_dsl_scores(expression_bank)

            # create branches if required:
            if self.era % self.n_branches == 0:
                # create branches
                if expression_bank is None:
                    expression_bank = super(
                        NaiveBOOTAD, self).start_experiment(expression_bank, cutshort=True)
                    plad_dsl_score = self.get_dsl_scores(expression_bank)

                self.save_with_dsl(self.dummy_era, expression_bank, dummy=True)
                new_expression_banks, new_macros_set = self.craft_branching_abstractions(
                    expression_bank, self.era, self.executor)
                for ind, expr_branch in enumerate(new_expression_banks):
                    expr_bank = self.update_expression_banks(
                        expr_branch, expression_bank)
                    expr_bank, remove_macros = self.remove_abstractions(
                        expr_bank, self.executor, new_macros_set[ind])
                    self.update_all_components(
                        new_macros_set[ind], remove_macros)
                    self.log_dsl_details(expr_bank)
                    new_dsl_score = self.get_dsl_scores(expr_bank)
                    # Save each:
                    self.save_with_dsl(self.era + ind, expr_bank)
                    _, expression_bank = self.load_with_dsl(
                        self.dummy_era, dummy=True)

                    print("Branch {}:".format(ind))
                    print("Abstraction step increased score from {} to {}".format(
                        plad_dsl_score, new_dsl_score))

            _, expression_bank = self.load_with_dsl(self.era)
            original_dsl_score = self.get_dsl_scores(expression_bank)
            expression_bank = super(
                NaiveBOOTAD, self).start_experiment(expression_bank, cutshort=False)
            plad_dsl_score = self.get_dsl_scores(expression_bank)
            print("PLAD step increased score from {} to {}".format(
                original_dsl_score, plad_dsl_score))

            # saving:
            if self.era % self.n_branches == (self.n_branches - 1):
                # compare all the k branches
                dsl_scores = []
                for ind in range(self.n_branches):
                    self.era, expression_bank = self.load_with_dsl(ind)
                    new_dsl_score = self.get_dsl_scores(expression_bank)
                    dsl_scores.append(new_dsl_score)
                # option 2: merge all the k branches up to top 3 cmds
                best_branch = np.argmax(dsl_scores)
                print("branch {} has the best score".format(best_branch))
                print("score: {}".format(dsl_scores[best_branch]))
                _, expression_bank = self.load_with_dsl(best_branch)
                for ind in range(self.n_branches):
                    self.save_with_dsl(ind, expression_bank)

            # Assume the inner loop succeded. Outcome? Expression bank.
            # Perform Merge Splicing on expression bank expressions.

            if new_dsl_score > original_dsl_score + self.dsl_score_tolerance:
                self.best_dsl_era = self.era
                self.best_dsl_score = new_dsl_score

            # outer saturation check:
            if self.era - self.best_dsl_era >= (self.dsl_patience * self.n_branches):
                print("Reached DSL saturation.")
                outer_loop_saturation = True
            else:
                print("DSL loop not saturated yet.")
            self.era += 1
        dsl_specs = self.get_dsl_specs()
        return dsl_specs

    def save_with_dsl(self, era, expression_bank, dummy=False):

        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        item = [era, self.executor, self.model_translator,
                self.model.cpu().state_dict(), expression_bank]
        # save_file = os.path.join(self.save_dir, "dsl_{}.pkl".format(era))
        if dummy:
            index = era
        else:
            index = era % self.n_branches
        save_file = os.path.join(self.save_dir, f"dsl_{index}.pkl")
        cPickle.dump(item, open(save_file, "wb"))
        self.model.cuda()
        self._save_model(era)
        self._save_model(era, "best")
        self._save_model(era, prefix="plad")

    def load_with_dsl(self, era, dummy=False):
        # find the latest era in save_dir:

        if dummy:
            index = era
        else:
            index = era % self.n_branches
        save_file = os.path.join(self.save_dir, f"dsl_{index}.pkl")
        if os.path.exists(save_file):
            item = cPickle.load(open(save_file, "rb"))
            era, self.executor, self.model_translator, model_weights, expression_bank = item
            self.model.cpu()
            # update the size of command embeddings:
            cmd_embd_size = model_weights['command_tokens.weight'].shape
            self.model.command_tokens = nn.Embedding(
                cmd_embd_size[0], cmd_embd_size[1])
            self.model.load_state_dict(model_weights)
            self.model.cuda()
            all_cmds = self.executor.get_cmd_list()
            self.num_commands = len(all_cmds)
        else:
            expression_bank = None
        return era, expression_bank

    def craft_branching_abstractions(self, expression_bank, era, executor):

        new_expression_bank_set, add_macros_set = self.abstraction_crafter.craft_branching_abstractions(
            expression_bank, era, executor, self.n_branches)
        return new_expression_bank_set, add_macros_set

    def compare_to_best(self, outer_iter, epoch, inner_iter, final_metrics):
        final_score = final_metrics["score"]
        index = self.era%self.n_branches
        best_score = self.best_scores[index]
        if final_score > best_score + self.score_tolerance:
            print("New best score: ", final_score)
            self.best_scores[index] = final_score
            self.best_epochs[index] = epoch
            self.best_outer_iter = outer_iter
            self._save_model(epoch, f"best_{self.era}")
            last_improvment_iter = inner_iter
        else:
            print("New score: ", final_score, " is not better than best score: ", self.best_score)
        return last_improvment_iter