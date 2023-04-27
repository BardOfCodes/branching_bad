import torch as th
import numpy as np
from collections import defaultdict


def batch_beam_decode(model, canvas, state_machine_class, beam_size=10):
    
    device = th.device('cuda')
    batch_size = canvas.shape[0]
    beam_state_size = canvas.shape[0]
    
    beam_entry_to_batch_id = {i: i for i in range(beam_state_size)}
    beam_entry_to_logprobability = {i: 0. for i in range(beam_state_size)}
    beam_logp = th.zeros(beam_state_size, 1).to(device)
    partial_action_seqs = {i: [] for i in range(beam_state_size)}
    
    n_commands = model.command_tokens.num_embeddings
    state_machine = state_machine_class(batch_size=beam_state_size, 
                                        n_commands=n_commands,
                                        device=device)
    
    finished_action_seqs = defaultdict(list)
    n_non_draw_commands = 4
    non_draw_beam_size = min(beam_size, n_non_draw_commands)
    # get all the cnn features:
    partial_sequences = model.get_init_sequence(canvas)
    while(partial_sequences.shape[0] > 0):
        # perform forward for all partial sequences. 
        output_cmd_logsf, output_param_logsf = model.forward_beam(partial_sequences)
        output_cmd = th.exp(output_cmd_logsf)
        output_param = th.exp(output_param_logsf)
        
        # cmd_entropy = th.sum(-output_cmd * output_cmd_logsf, dim=1)
        # get the top k predictions for each sequence.
        # apply mask on actions
        mask_array = state_machine.get_state_mask()
        output_cmd_masked = output_cmd * mask_array
        # select top k
        # no param commands: 
        no_param_cmd = output_cmd_masked[:, 2:]
        top_k_cmd_probs, top_k_cmd_inds = th.topk(no_param_cmd, k=non_draw_beam_size, dim=1)
        
        # param cmds:
        # 1: Get the top k parameter values:
        top_k_param_vals, top_k_param_inds = th.topk(output_param, k=beam_size, dim=2)
        a = top_k_param_vals[:, 0, :]
        b = top_k_param_vals[:, 1, :]
        c = top_k_param_vals[:, 2, :]
        d = top_k_param_vals[:, 3, :]
        e = top_k_param_vals[:, 4, :]
        new = th.einsum("xa, xb, xc, xd, xe-> xabcde", a, b, c, d, e)
        new = new.reshape(new.shape[0], -1)
        selected_param_probs, inner_top_k = th.topk(new, k=beam_size, dim=1)
        ax_a = (inner_top_k // beam_size ** 4) % beam_size
        ax_b = (inner_top_k // beam_size **3) % beam_size
        ax_c = (inner_top_k // beam_size ** 2) % beam_size
        ax_d = (inner_top_k // beam_size) % beam_size
        ax_e = inner_top_k  % beam_size
        selected_params = th.stack([ax_a, ax_b, ax_c, ax_d, ax_e], dim=2)
        top_k_fliped = top_k_param_inds.swapaxes(1, 2)
        selected_param_inds = th.gather(top_k_fliped, dim=1, index=selected_params)
        # 2: Get the Cmd with parameters:
        # all except bools and stop
        draw_command_probs = output_cmd_masked[:, :2]
        all_probs = th.einsum("xa, xb-> xab", draw_command_probs, selected_param_probs)
        all_probs = all_probs.reshape(all_probs.shape[0], -1)
        final_param_probs, final_param_top_k = th.topk(all_probs, k=beam_size, dim=1)
        ax_a = (final_param_top_k // beam_size) 
        ax_b = final_param_top_k  % beam_size
        final_draw_options = th.stack([ax_a, ax_b], dim=2)
        
        # HACK:
        # final_param_probs = th.gather(draw_command_probs, dim=1, index=ax_a)
        # OR adjust be size:
        final_param_probs = th.exp(th.log(final_param_probs) / 6.0)
        
        # Now gather the top k for each batch id
        # if any seq is finished, add it to "complete programs".
        total_probs = th.cat([top_k_cmd_probs, final_param_probs], dim=1)
        final_probs, final_k = th.topk(total_probs, k=beam_size, dim=1)
        
        
        
        # expensive: could be parallelized.
        counter = 0
        bool_count, canvas_count = state_machine.get_bool_and_canvas_count()
        new_beam_entry_to_batch_id = dict()
        new_beam_entry_to_logprobability = dict()
        new_partial_action_seqs = dict()
        new_bool_count = []
        new_canvas_count = []
        new_partial_sequences = []
        
        
        # Now for each batch entry only allow the top k to continue.
        batch_id_to_logprobs = defaultdict(list)
        final_logps_overall = th.log(final_probs) + beam_logp
        backward_mapper = dict()
        for beam_ind in range(beam_state_size):
            batch_id = beam_entry_to_batch_id[beam_ind]
            batch_id_to_logprobs[batch_id].append(final_logps_overall[beam_ind:beam_ind + 1])
            backward_mapper[(batch_id, len(batch_id_to_logprobs[batch_id]) - 1)] = beam_ind
        
        # for each batch id, 
        for batch_id in batch_id_to_logprobs.keys():
            cur_logps = th.cat(batch_id_to_logprobs[batch_id], dim=1)
            batch_specific_val, batch_specific_k = th.topk(cur_logps, k=beam_size, dim=1)
            batch_specific_k = batch_specific_k[0]
            insider_ind = batch_specific_k // beam_size
            cur_ks = batch_specific_k % beam_size
            for i in range(beam_size):
                back_mapper_ind = insider_ind[i].item()
                beam_ind = backward_mapper[(batch_id, back_mapper_ind)]
                cur_k = cur_ks[i]
        # for beam_ind in range(beam_state_size):
        #     for cur_k in range(beam_size):
                action_type = final_k[beam_ind, cur_k]
                prob = final_probs[beam_ind, cur_k]
                batch_id = beam_entry_to_batch_id[beam_ind]
                prev_logprob = beam_entry_to_logprobability[beam_ind]
                if prob == 0:
                    continue
                prev_partial_seq = partial_sequences[beam_ind].clone()
                prev_bool_count = bool_count[beam_ind]
                prev_canvas_count = canvas_count[beam_ind]
                cur_partial_actions = partial_action_seqs[beam_ind]
                
                if action_type <= non_draw_beam_size-1:
                    # Create new partial seq
                    cmd_type = top_k_cmd_inds[beam_ind, action_type] + 2
                    
                    if cmd_type == 5:
                        # stop action
                        finished_seq = partial_action_seqs[beam_ind]
                        finished_action_seqs[batch_id].append(finished_seq)
                    else:
                        new_beam_entry_to_batch_id[counter] = batch_id
                        new_beam_entry_to_logprobability[counter] = prev_logprob + th.log(prob)
                        new_bool_count.append(prev_bool_count + 1)
                        new_canvas_count.append(prev_canvas_count)
                        extended_seq = model.extend_seq(prev_partial_seq, cmd_type, None)
                        new_partial_sequences.append(extended_seq)
                        new_partial_action_seqs[counter] = cur_partial_actions + [cmd_type.item()]
                        counter += 1
                    
                else:
                    # action with params
                    new_beam_entry_to_batch_id[counter] = batch_id
                    new_beam_entry_to_logprobability[counter] = prev_logprob + th.log(prob)
                    new_bool_count.append(prev_bool_count)
                    new_canvas_count.append(prev_canvas_count + 1)
                    act_type = action_type - (non_draw_beam_size)
                    draw_type = final_draw_options[beam_ind, act_type, 0] #+ 3
                    param_type = final_draw_options[beam_ind, act_type, 1]
                    params = selected_param_inds[beam_ind, param_type]
                    extended_seq = model.extend_seq(prev_partial_seq, draw_type, params)
                    new_partial_sequences.append(extended_seq)
                    new_action = [draw_type.item(), ] + list(params.cpu().numpy())
                    new_partial_action_seqs[counter] = cur_partial_actions + new_action
                    counter += 1
            
        if len(new_bool_count) > 0:
        # from the gathered potential seq. launch k new seq.
            beam_entry_to_batch_id = new_beam_entry_to_batch_id
            beam_entry_to_logprobability = new_beam_entry_to_logprobability
            beam_logp = th.stack(list(beam_entry_to_logprobability.values()), dim=0)
            beam_logp = beam_logp.unsqueeze(1)
            partial_action_seqs = new_partial_action_seqs
            beam_state_size = len(beam_entry_to_batch_id)
            state_machine = state_machine_class(batch_size=beam_state_size, 
                                                n_commands=n_commands,
                                                device=device)
            state_machine.bool_count = th.stack(new_bool_count, 0)
            state_machine.canvas_count = th.stack(new_canvas_count, 0)
            partial_sequences = th.stack(new_partial_sequences, 0)
        else:
            break
    all_actions_seqs = []
    for i in range(batch_size):
        cur_seqs = finished_action_seqs[i]
        if len(cur_seqs) > 0:
            all_actions_seqs.append(cur_seqs)
        else:
            print("WAT")
    # just return all sequences: 
    return all_actions_seqs
    