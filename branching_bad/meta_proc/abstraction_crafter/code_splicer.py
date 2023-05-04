from .macro import Macro
from CSG.env.csg2d.languages import GraphicalMCSG2DCompiler, MCSG2DParser, PCSG2DParser
from CSG.bc_trainers.rewrite_engines.code_splice_utils import get_new_command, distill_transform_chains
from CSG.bc_trainers.rewrite_engines.subexpr_cache import MergeSplicerCache
import numpy as np
import torch as th
import time
import random
import faiss
import sys
import _pickle as cPickle

MAX_SIZE = int(100000)


class BootADSplicer(MergeSplicerCache):

    def bbad_setup(self, config, existing_macros=None):
        self.length_tax_rate = config.LENGTH_TAX
        self.merge_max_candidates = config.MACRO_PER_ERA  # config.MERGE_MAX_CANDIDATES
        self.novelty_reward = config.NOVELTY_REWARD
        self.merge_bit_distance = config.MERGE_BIT_DISTANCE
        self.max_new_percentage = config.MAX_NEW_PERCENTAGE
        self.existing_macros = existing_macros

    def generate_cache_and_index(self, best_program_dict, temp_env, executor, era):
        total_expressions = np.sum([len(v) for v in best_program_dict.values()])
        total_required_expressions = int(self.max_new_percentage * total_expressions)
        
        subexpr_cache = self.generate_subexpr_cache(
            best_program_dict, temp_env, executor)
        # cPickle.dump(subexpr_cache, open("step_1.pkl" ,"wb"))
        # subexpr_cache = cPickle.load(open("step_1.pkl" ,"rb"))
        if len(subexpr_cache) > MAX_SIZE:
            subexpr_cache = random.sample(subexpr_cache, MAX_SIZE)
        # Setting some primitives
        print("Number of subexpressions in cache = %d" % len(subexpr_cache))
        th.cuda.empty_cache()
        # cPickle.dump(subexpr_cache, open("step_2.pkl" ,"wb"))
        # subexpr_cache = cPickle.load(open("step_2.pkl" ,"rb"))

        merge_spliced_commands, new_macros = self.merge_cache(
            subexpr_cache, era)

        # measure the performance with the new commands?
        executor.update_macros(new_macros)

        merge_spliced_expression_bank = self.annotate_and_convert(
            merge_spliced_commands, temp_env, executor)
        
        # generate extra samples for each expression.
        # OPTION 1: Try to splice the new macros into other expressions.
        n_new_required = total_required_expressions - len(merge_spliced_expression_bank)
        # Option 2: Brute force macro through all possible positions.
        additional_expressions = self.brute_force_macro_use(best_program_dict, new_macros, temp_env, executor, n_new_required=n_new_required)
        
        # OPTION 2: Generate random samples with the new macros.
        merge_spliced_expression_bank.extend(additional_expressions)

        # get expression_bank from merge_spliced_commands
        return merge_spliced_expression_bank, new_macros


    def brute_force_macro_use(self, best_program_dict, new_macros, temp_env, executor, max_count=2, n_new_required=1000):
        print("Collecting Subexpressions")
        # Check it to the new ones:
        # Replace with temp_env.parser
        # base_parser = temp_env.program_generator.parser
        base_parser = executor.parser
        graph_compiler = GraphicalMCSG2DCompiler(resolution=temp_env.program_generator.compiler.resolution,
                                                 scale=temp_env.program_generator.compiler.scale,
                                                 draw_mode=temp_env.program_generator.compiler.draw.mode)

        graph_compiler.set_to_cuda()
        graph_compiler.set_to_half()
        graph_compiler.reset()

        # base_parser.set_device("cuda")
        # base_parser.set_tensor_type(th.float32)
        #
        new_expression_bank = []
        counter = 0
        add_counter = 0
        st = time.time()
        subexpr_cache = []
        score_deltas = []
        # random sample 10k expressions:
        keys = list(best_program_dict.keys())
        rand_keys = random.sample(
            keys, min(len(keys), self.n_program_to_sample))

        for iteration, key in enumerate(rand_keys):
            if iteration % 500 == 0:
                print("cur iteration %d. Cur Time %f" %
                      (iteration, time.time() - st))
                if len(subexpr_cache) > MAX_SIZE:
                    print("subexpr cache is full", len(
                        subexpr_cache), ". Sampling from it")
                    subexpr_cache = random.sample(subexpr_cache, MAX_SIZE)

            values = best_program_dict[key]
            new_expressions = []
            
            for cur_value in values:
                expression = cur_value['expression']
                with th.no_grad():
                    # with th.cuda.amp.autocast():
                    # Annotate the marco here.
                    command_list, _ = base_parser.parse_for_graph(expression, adjust_param=False)
                    # command_list = base_parser.parse(expression)
                    graph = graph_compiler.command_tree(
                        command_list, None, enable_subexpr_targets=False, add_splicing_info=True)
                graph_nodes = [graph.nodes[i] for i in graph.nodes]
                for node in graph_nodes:
                    # This is to include the leaf commands
                    add = node['type'] in ["B", "M", "D"]
                    # add = self.accept_node(node)
                    if add:
                        # substitute the macro instead of the node.
                        # if the macro matches the node canonical target better.
                         
                        cmd_list = command_list
                        old_command_ids = node['subexpr_info']['command_ids']
                        # node['subexpr_info']['command_ids']
                        cmd_start, cmd_end = old_command_ids
                        old_canonical_commands = node['subexpr_info']['canonical_commands']
                        for macro in new_macros:
                            new_transform_chaim = macro.resolve_param_chain(
                                old_canonical_commands)
                            new_command_list = cmd_list[:cmd_start] + \
                                new_transform_chaim + cmd_list[cmd_end:]
                            expression = base_parser.get_expression(new_command_list)
                            new_expressions.append(expression)
                        counter += 1
            # New compute the new expressions?
            if new_expressions:
                new_expr_len = [len(x) for x in new_expressions]
                # print("max new expression length", max(new_expr_len))
                old_lens = [len(x['expression']) for x in values]
                # print("previous expression length", max(old_lens))
                if max(new_expr_len) > max(old_lens):
                    print("WT?")
                
                previous_reward = cur_value['reward']
                slot_id = cur_value['slot_id']
                target_id = cur_value['target_id']
                target_np, _ = temp_env.program_generator.get_executed_program(
                    slot_id, target_id)
                target = th.from_numpy(target_np).cuda().bool().unsqueeze(0)
                pred_canvases = executor.eval_batch_execute([new_expressions])[0]
                pred_canvases = pred_canvases.bool()
                target = target.expand(pred_canvases.shape[0], -1, -1)
                # max 2 per expression:
                new_rewards = th.logical_and(pred_canvases, target).sum(1).sum(1)/th.logical_or(pred_canvases, target).sum(1).sum(1)
                # bulk execute the new expressions.
                
                value, index = th.topk(new_rewards, k=max_count)
                
                for ind, index in enumerate(index):
                    if value[ind] >= (previous_reward - self.length_tax_rate * len(expression)):
                        cur_expression = new_expressions[index]
                        new_iou = value[ind].item()
                        new_score = new_iou + self.length_tax_rate * len(cur_expression)
                        new_score = new_score
                        expr_obj = dict(expression=cur_expression,
                                        score=new_score, iou=new_iou, target_index=target_id)
                        new_expression_bank.append(expr_obj)
                        score_delta = new_score - previous_reward
                        score_deltas.append(score_delta)
                        add_counter += 1
            if add_counter > n_new_required:
                break
                
        et = time.time()
        print("New program Search", et - st)
        print("Evaluated %d expressions with macros" % counter)
        print("Accepted %d new expressions with macros" % add_counter)
        print("Average score delta", np.mean(score_deltas))
        return new_expression_bank


    def annotate_and_convert(self, merge_spliced_commands, temp_env, executor):

        parser = executor.parser
        expression_bank = []
        deltas = []
        for cmd_obj in merge_spliced_commands:
            slot_id = cmd_obj[0]
            target_id = cmd_obj[1]
            new_cmd_list = cmd_obj[2]
            old_cmd_list = cmd_obj[3]
            old_score = cmd_obj[4]
            old_expression = cmd_obj[5]
            # get targe:
            target_np, _ = temp_env.program_generator.get_executed_program(
                slot_id, target_id)
            target = th.from_numpy(target_np).cuda().bool()
            # convert the expression:
            expression = parser.get_expression(new_cmd_list)

            # TODO: FIX this
            batch = [[expression, ]]
            output = executor.eval_batch_execute(batch)
            output = output[0][0]

            new_iou = th.logical_and(output, target).sum(
            )/th.logical_or(output, target).sum()
            new_iou = new_iou.item()
            new_score = new_iou + self.length_tax_rate * len(expression)
            new_score += self.novelty_reward
            # add to bank with new performance
            expr_obj = dict(expression=expression,
                            score=new_score, iou=new_iou, target_index=target_id)
            # print("previous expression length", len(old_expression))
            # print("New expression length", len(expression))
            if len(expression) > len(old_expression):
                print("WT?")
            expression_bank.append(expr_obj)
            # note delta in performance
            deltas.append(new_score - old_score)
        
        print("Average delta in performance = %f" % np.mean(deltas))
        print("number of new expressions in bank = %d" % len(expression_bank))

        return expression_bank

    def merge_cache(self, subexpr_cache, era, device=th.device("cuda"), dtype=th.float32):
        # Now from this cache create unique:

        avg_length = np.mean([len(x['commands']) for x in subexpr_cache])
        print("Starting merge with  %d sub-expressions with avg. length %f" %
              (len(subexpr_cache), avg_length))

        st = time.time()
        cached_expression_shapes = [
            x['canonical_shape'].reshape(-1) for x in subexpr_cache]
        cached_expression_shapes = th.stack(cached_expression_shapes, 0)
        # cached_expression_shapes.shape
        cached_np = cached_expression_shapes.cpu().data.numpy()
        chached_np_packed = np.packbits(cached_np, axis=-1, bitorder="little")

        self.cache_d = cached_expression_shapes.shape[1]
        merge_nb = cached_expression_shapes.shape[0]
        # Initializing index.
        quantizer = faiss.IndexBinaryFlat(self.cache_d)  # the other index

        index = faiss.IndexBinaryIVF(quantizer, self.cache_d, self.merge_nlist)
        assert not index.is_trained
        index.train(chached_np_packed)
        assert index.is_trained
        index.add(chached_np_packed)
        index.nprobe = self.merge_nprobe
        lims, D, I = index.range_search(
            chached_np_packed, self.merge_bit_distance)
        lims_shifted = np.zeros(lims.shape)
        lims_shifted[1:] = lims[:-1]

        all_indexes = set(list(range(merge_nb)))
        selected_subexprs = []

        # Which abstractions to make?
        # mark the conversions by the delta in reward.
        macro_candidates = []
        all_indexes = set(list(range(merge_nb)))
        selected_subexprs = []
        while (len(all_indexes) > 0):
            cur_ind = next(iter(all_indexes))
            sel_lims = (lims[cur_ind], lims[cur_ind+1])
            selected_ids = I[sel_lims[0]:sel_lims[1]]
            sel_exprs = [subexpr_cache[x] for x in selected_ids]
            min_len = np.inf
            for ind, expr in enumerate(sel_exprs):
                # TODO: Measure the length of the true expression
                cur_len = len(expr['subexpression'])
                if cur_len < min_len:
                    min_len = cur_len
                    min_ind = ind
            # Now for the rest create a new expression with the replacement
            # measure expected increase in performance:
            expr_to_splice = sel_exprs[min_ind]
            macro_candidate = {
                'expr_obj': expr_to_splice,
                'target_exprs': sel_exprs,
                'expected_delta': -self.length_tax_rate * (max(0, len(expr_to_splice['subexpression']) - 1))**0.25 * len(sel_exprs)
                # 'expected_delta': -self.length_tax_rate * len(sel_exprs)
            }
            macro_candidates.append(macro_candidate)

            compressed_expr = {x: y for x, y in expr_to_splice.items()}
            selected_subexprs.append(compressed_expr)
            for ind in selected_ids:
                if ind in all_indexes:
                    all_indexes.remove(ind)
        # Get the best expressions to replace:
        macro_candidates.sort(key=lambda x: x['expected_delta'], reverse=True)
        selected_candidates = macro_candidates[:self.merge_max_candidates]

        # create the new macros:
        merge_spliced_commands = []
        new_macros = []
        for candidate_ind, candidate in enumerate(selected_candidates):
            expr_to_splice = candidate['expr_obj']
            sel_exprs = candidate['target_exprs']
            subexpression = expr_to_splice['subexpression']
            new_macro = Macro(expr_to_splice, era, candidate_ind, subexpression)

            for ind, expr in enumerate(sel_exprs):
                if "all_commands" in expr.keys():
                    # replace the other command with the macro
                    cmd_list = expr['all_commands']
                    old_command_ids = expr['command_ids']
                    # node['subexpr_info']['command_ids']
                    cmd_start, cmd_end = old_command_ids
                    old_reward = expr['reward']
                    old_canonical_commands = expr['canonical_commands']
                    new_transform_chaim = new_macro.resolve_param_chain(
                        old_canonical_commands)
                    new_command_list = cmd_list[:cmd_start] + \
                        new_transform_chaim + cmd_list[cmd_end:]
                    # merge_spliced_commands.append([expr['slot_id'], expr['target_id'],
                    #                                new_command_list, cmd_list, old_reward])
                    merge_spliced_commands.append([expr['slot_id'], expr['target_id'],
                                                   new_command_list, cmd_list, old_reward, expr['expression']])
                else:
                    print("WUT")
            new_macros.append(new_macro)

        et = time.time()
        print("Merge Process Time", et - st)
        return merge_spliced_commands, new_macros

    def generate_subexpr_cache(self, best_program_dict, temp_env, executor):
        print("Collecting Subexpressions")
        # Check it to the new ones:
        # Replace with temp_env.parser
        # base_parser = temp_env.program_generator.parser
        base_parser = executor.parser
        graph_compiler = GraphicalMCSG2DCompiler(resolution=temp_env.program_generator.compiler.resolution,
                                                 scale=temp_env.program_generator.compiler.scale,
                                                 draw_mode=temp_env.program_generator.compiler.draw.mode)

        graph_compiler.set_to_cuda()
        graph_compiler.set_to_half()
        graph_compiler.reset()

        # base_parser.set_device("cuda")
        # base_parser.set_tensor_type(th.float32)

        counter = 0
        st = time.time()
        subexpr_cache = []
        # random sample 10k expressions:
        keys = list(best_program_dict.keys())
        rand_keys = random.sample(
            keys, min(len(keys), self.n_program_to_sample))

        for iteration, key in enumerate(rand_keys):
            value = best_program_dict[key]
            if iteration % 500 == 0:
                print("cur iteration %d. Cur Time %f" %
                      (iteration, time.time() - st))
                if len(subexpr_cache) > MAX_SIZE:
                    print("subexpr cache is full", len(
                        subexpr_cache), ". Sampling from it")
                    subexpr_cache = random.sample(subexpr_cache, MAX_SIZE)

            for cur_value in value:
                expression = cur_value['expression']
                with th.no_grad():
                    # with th.cuda.amp.autocast():
                    # Annotate the marco here.
                    command_list, _ = base_parser.parse_for_graph(expression, adjust_param=False)
                    # command_list = base_parser.parse(expression)
                    graph = graph_compiler.command_tree(
                        command_list, None, enable_subexpr_targets=False, add_splicing_info=True)
                graph_nodes = [graph.nodes[i] for i in graph.nodes]
                for node in graph_nodes:
                    # This is to include the leaf commands
                    add = self.accept_node(node)
                    if add:
                        # start, end = node['subexpr_info']['command_ids']
                        sub_cmds = node['subexpr_info']['commands']
                        subexpression = graph_compiler.cmd_to_expression(sub_cmds)
                        shape = node['subexpr_info']['canonical_shape']
                        node_item = {'canonical_shape': shape,
                                     'commands': node['subexpr_info']['commands'],
                                     'canonical_commands': node['subexpr_info']['canonical_commands'],
                                     'slot_id': cur_value['slot_id'],
                                     'target_id': cur_value['target_id'],
                                     'command_ids': node['subexpr_info']['command_ids'],
                                     'all_commands': command_list,
                                     'reward': cur_value['reward'],
                                     'subexpression': subexpression,
                                     'expression': expression}
                        subexpr_cache.append(node_item)

                        counter += 1
        et = time.time()
        print("Subexpr Discovery Time", et - st)
        print("found %d sub-expressions" % counter)
        return subexpr_cache


def get_new_command(command_list, command_inds, prev_canonical_commands,
                    candidate, use_canonical=True, device=th.device("cuda"), dtype=th.float32):
    # node['subexpr_info']['command_ids']
    command_start, command_end = command_inds
    if use_canonical:
        target_commands = candidate['canonical_commands'] + \
            candidate['commands']
        # [x.copy() for x in node['subexpr_info']['canonical_commands']]
        current_canonical_commands = [x.copy()
                                      for x in prev_canonical_commands]
        current_canonical_commands[1]['param'] *= -1
        current_canonical_commands[0]['param'] = 1 / \
            (current_canonical_commands[0]['param'] + 1e-9)
        # get param from these:
        parameters = current_canonical_commands[::-
                                                1] + candidate['canonical_commands']
        parameters = distill_transform_chains(new_command_list, device, dtype)

        cmd = {'type': "M",
               "symbol": target_commands['symbol'], "param": parameters}
        new_command_list = command_list[:command_start] + \
            cmd + command_list[command_end:]
    else:
        target_commands = candidate['commands']
        new_command_list = command_list[:command_start] + \
            target_commands + command_list[command_end:]
    return new_command_list
