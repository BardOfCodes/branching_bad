from collections import defaultdict
import numpy as np
import torch as th


class StatEstimator:

    def __init__(self, length_weight=-0.01, novelty_score_weight=0.01, beam_return_count=1):

        self.length_weight = length_weight
        self.beam_return_count = beam_return_count
        self.novelty_score_weight = novelty_score_weight

        self.all_scores = []
        self.all_lens = []
        self.all_ious = []
        self.expression_bank = []
        self.novel_cmds = []
        self.novel_cmd_usage = 0

    def get_final_metrics(self):
        metrics = dict()
        metrics["score"] = np.mean(self.all_scores)
        metrics["avg_lengths"] = np.mean(self.all_lens)
        metrics["IoU"] = np.mean(self.all_ious)
        metrics['novel_cmd_usage'] = self.novel_cmd_usage
        return metrics
    
    def get_expression_stats(self):
        count_dict = defaultdict(int)
        
        for expression in self.expression_bank:
            for cmd in expression['expression']:
                cmd_symbol = cmd.split("(")[0]
                count_dict[cmd_symbol] += 1
        # sort the expression dictionary:
        count_dict = {k: v for k, v in sorted(count_dict.items(), key=lambda item: item[1], reverse=True)}
                
        return count_dict
        
        
    def update_novelty_cmds(self, parser):
        self.novel_cmds = parser.get_novel_cmds()
        
    def eval_batch_execute(self, pred_canvases, pred_expressions, targets, indices):

        storage_count = []
        all_preds = []
        all_lengths = []
        all_targets = []
        all_novelty_score = []
        counter = 0
        for i, pred_batch in enumerate(pred_canvases):
            start_counter = counter
            batch_size = pred_batch.shape[0]
            all_preds.append(pred_batch)
            lengths = [len(x) for x in pred_expressions[i]]
            all_lengths.append(np.array(lengths))
            all_targets.append(targets[i:i+1].expand(batch_size, -1, -1))
            counter += batch_size
            end_counter = counter
            storage_count.append((start_counter, end_counter))
            novelty_score = [np.sum([y.split("(")[0] in self.novel_cmds for y in x]) for x in pred_expressions[i]]
            novelty_score = np.array(novelty_score)
            self.novel_cmd_usage += np.sum(novelty_score)
            all_novelty_score.append(novelty_score)

        all_lengths = th.from_numpy(np.concatenate(
            all_lengths, 0)).to(targets.device)
        all_novelty_score = th.from_numpy(np.concatenate(
            all_novelty_score, 0)).to(targets.device)
        all_preds = th.cat(all_preds, 0)
        all_preds = all_preds.reshape(all_preds.shape[0], -1)
        all_targets = th.cat(all_targets, 0)
        all_targets = all_targets.reshape(all_targets.shape[0], -1)

        iou = th.logical_and(all_preds, all_targets).sum(
            1).float() / th.logical_or(all_preds, all_targets).sum(1).float()
        all_scores = iou + self.length_weight * all_lengths + self.novelty_score_weight * all_novelty_score
        # all_scores = all_scores.cpu().numpy()
        iou = iou.cpu().numpy()

        selected_expression_bank = []
        for i, pred_batch in enumerate(pred_canvases):
            select_scores = all_scores[storage_count[i][0]:storage_count[i][1]]
            select_iou = iou[storage_count[i][0]:storage_count[i][1]]

            best_scores, best_ids = th.topk(
                select_scores, k=self.beam_return_count)
            select_scores = select_scores.cpu().numpy()
            best_id = best_ids[0]
            selected_expression = pred_expressions[i][best_id]
            # pred_canvas = pred_canvases[i][best_id]
            self.all_scores.append(select_scores[best_id])
            self.all_ious.append(select_iou[best_id])
            self.all_lens.append(len(selected_expression))
            # now create the expression obj:
            for cur_id in best_ids:
                selected_expression = pred_expressions[i][cur_id]
                cur_score = select_scores[cur_id]
                cur_iou = select_iou[cur_id]
                target_index = indices[i]
                expr_obj = dict(expression=selected_expression,
                                score=cur_score, iou=cur_iou, target_index=target_index)
                selected_expression_bank.append(expr_obj)

        self.expression_bank.extend(selected_expression_bank)

        return selected_expression_bank
