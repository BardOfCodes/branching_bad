import numpy as np
import torch as th


class StatEstimator:

    def __init__(self, length_weight=-0.01):

        self.length_weight = length_weight

        self.all_scores = []
        self.all_lens = []
        self.all_ious = []
        self.expressions = []

    def get_final_metrics(self):
        metrics = dict()
        metrics["score"] = np.mean(self.all_scores)
        metrics["avg_lengths"] = np.mean(self.all_lens)
        metrics["IoU"] = np.mean(self.all_ious)
        return metrics

    def eval_batch_execute(self, pred_canvases, pred_expressions, targets):

        storage_count = []
        all_preds = []
        all_lengths = []
        all_targets = []
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

        all_lengths = th.from_numpy(np.concatenate(
            all_lengths, 0)).to(targets.device)
        all_preds = th.cat(all_preds, 0)
        all_preds = all_preds.reshape(all_preds.shape[0], -1)
        all_targets = th.cat(all_targets, 0)
        all_targets = all_targets.reshape(all_targets.shape[0], -1)

        iou = th.logical_and(all_preds, all_targets).sum(
            1).float() / th.logical_or(all_preds, all_targets).sum(1).float()
        all_scores = iou + self.length_weight * all_lengths

        selected_expressions = []
        for i, pred_batch in enumerate(pred_canvases):
            select_scores = all_scores[storage_count[i][0]:storage_count[i][1]]
            select_iou = iou[storage_count[i][0]:storage_count[i][1]]
            best_id = select_scores.argmax()
            selected_expression = pred_expressions[i][best_id]
            # pred_canvas = pred_canvases[i][best_id]
            self.all_scores.append(select_scores[best_id].item())
            self.all_ious.append(select_iou[best_id].item())
            self.all_lens.append(len(selected_expression))
            selected_expressions.append(selected_expression)

        self.expressions.extend(selected_expressions)

        return selected_expressions
