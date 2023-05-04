
import torch.nn as nn
import torch as th
import torch.nn.functional as F
from .model_registry import ModelRegistry
from .feature_extractor import Vox2DCNN
import einops
from .transformer_components import LearnablePositionalEncoding, AttnLayer


class SEQ_MLP(nn.Module):
    def __init__(self, size_seq, dropout_rate):
        super(SEQ_MLP, self).__init__()
        layers = []
        n_layers = len(size_seq) - 1
        for i in range(n_layers):
            layer = nn.Linear(size_seq[i], size_seq[i+1])
            layers.append(layer)
            if i != (n_layers - 1):
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=dropout_rate))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        output = self.layers(x)
        return output


@ModelRegistry.register("BaseTransformer")
class BaseTransformer(nn.Module):

    def __init__(self, config):
        super(BaseTransformer, self).__init__()
        # Parameters:
        self.set_settings(config)
        max_length = self.visual_seq_len + self.out_seq_len + 1

        self.cnn_extractor = Vox2DCNN(
            self.attn_size, self.dropout, out_len=self.visual_seq_len)

        self.pos_encoding = LearnablePositionalEncoding(
            self.attn_size, self.dropout, max_len=max_length)

        # scale + start and end
        self.param_scale_tokens = nn.Embedding(
            self.real_param_scale, self.attn_size)
        # This will be updated:
        self.command_tokens = nn.Embedding(
            self.command_token_count, self.attn_size)

        self.attn_layers = nn.ModuleList([AttnLayer(
            self.num_heads, self.attn_size, self.dropout) for _ in range(self.num_dec_layers)])

        # convert to MLPS:
        self.singular_token_mapper = SEQ_MLP(
            self.cmd_param_mapper_mlp, self.dropout)

        self.after_attn_process = SEQ_MLP(self.post_attn_mlp, self.dropout)
        self.cmd_vector = SEQ_MLP(self.cmd_mlp, self.dropout)
        self.param_predictor = SEQ_MLP(self.param_mlp, self.dropout)

        self.cmd_logsoft = nn.LogSoftmax(dim=1)  # Module
        self.param_logsoft = nn.LogSoftmax(dim=2)  # Module
        # For Transformer:
        attn_mask = self.generate_attn_mask()
        self.key_mask = None
        start_token = th.LongTensor([[0]])  # .to(self.init_device)
        self.register_buffer("start_token", start_token)
        self.register_buffer("attn_mask", attn_mask)

        self.apply(self.initialize_weights)

    def set_settings(self, config):

        # Parameters:
        self.post_attn_size = config.POST_ATTN_SIZE  # 256
        self.attn_size = config.ATTN_SIZE  # 128 # attn_size
        self.visual_seq_len = config.VISUAL_SEQ_LENGTH  # 8
        self.out_seq_len = config.OUTPUT_SEQ_LENGTH  # 128  + 1# seq_len
        self.num_enc_layers = config.NUM_ENC_LAYERS  # 8 # num_layers
        self.num_dec_layers = config.NUM_DEC_LAYERS  # 8 # num_layers
        self.num_heads = config.NUM_HEADS  # 16# num_heads
        self.dropout = config.DROPOUT
        self.return_all = config.RETURN_ALL  # False
        self.zero_value = th.FloatTensor([0])
        self.beam_mode = False
        #
        self.beam_partial_init = False
        self.x_count = None
        # new
        self.real_param_scale = config.REAL_PARAM_SCALE
        self.command_token_count = config.COMMAND_TOKEN_COUNT
        self.cmd_param_mapper_mlp = config.CMD_PARAM_MAPPER_MLP
        self.post_attn_mlp = config.POST_ATTN_MLP
        self.cmd_mlp = config.CMD_MLP
        self.param_mlp = config.PARAM_MLP
        self.cmd_logsf_scaler = config.CMD_LOGSF_SCALER
        self.reload_on_dsl_update = config.RELOAD_ON_DSL_UPDATE

    def initialize_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)

    def generate_attn_mask(self):
        sz = self.visual_seq_len + self.out_seq_len
        mask = (th.triu(th.ones(sz, sz)) == 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0)).T
        mask[:self.visual_seq_len, :self.visual_seq_len] = 0.
        return mask

    def generate_key_mask(self, num, device):
        if num == self.key_mask.shape[0]:
            return self.key_mask
        else:
            sz = self.visual_seq_len + self.out_seq_len
            self.key_mask = th.zeros(num, sz).bool().to(device)

    def forward_train(self, x_in, actions_in):

        token_embeddings = self.embed_action_sequence(actions_in)
        cnn_features = self.cnn_extractor.forward(x_in)
        out = self.pos_encoding(
            th.cat((cnn_features, token_embeddings), dim=1))

        for attn_layer in self.attn_layers:
            out = attn_layer(out, self.attn_mask, self.key_mask)
        seq_out = out[:, self.visual_seq_len-1:-1, :]

        output = self.stack_all_vectors(seq_out)

        if len(output.shape) == 3:
            output = output.squeeze(1)

        cmd_logsoft, param_logsoft = self.attention_to_cmd_param(output)
        # cmd_distr = th.softmax(cmd_distr, dim = 1)
        return cmd_logsoft, param_logsoft

    def embed_action_sequence(self, actions_in):

        batch_size, max_action_size, _ = actions_in.size()

        cmd_in = actions_in[:, :, 0:1]

        cmd_token_embeddings = self.command_tokens(cmd_in)

        param_in = actions_in[:, :, 1:-1]
        param_token_embeddings = self.param_scale_tokens(param_in)

        all_embeddings = th.cat(
            (cmd_token_embeddings, param_token_embeddings), dim=2)
        all_embeddings = all_embeddings.reshape(
            batch_size, max_action_size, -1)
        token_embeddings = self.singular_token_mapper(all_embeddings)

        action_type = actions_in[:, :, -1:].bool()
        action_type = action_type.expand(-1, -1, self.attn_size)
        token_embeddings = th.where(
            action_type, token_embeddings, cmd_token_embeddings[:, :, 0, :])

        return token_embeddings

    def extend_seq(self, partial_seq, cmd_in, param_in=None):

        cmd_token_embeddings = self.command_tokens(cmd_in)
        cmd_token_embeddings.unsqueeze_(0)
        if not param_in is None:
            param_token_embeddings = self.param_scale_tokens(param_in)
            all_embeddings = th.cat(
                (cmd_token_embeddings, param_token_embeddings), dim=0)
            all_embeddings = all_embeddings.reshape(
                1, -1)
            cmd_token_embeddings = self.singular_token_mapper(all_embeddings)
        # new extend the partial seq:
        cmd_token_embeddings = self.pos_encoding.get_singular_position(
            cmd_token_embeddings, partial_seq.shape[0:1])
        new_partial_seq = th.cat((partial_seq, cmd_token_embeddings), dim=0)
        return new_partial_seq

    def update_cmds(self, cmds):
        
        self.command_token_count = len(cmds)
        new_command_tokens = nn.Embedding(
            self.command_token_count, self.attn_size)
        # Learn them new?
        prev_n_cmds = self.command_tokens.num_embeddings
        new_command_tokens.weight[:prev_n_cmds].data = self.command_tokens.weight.detach().data
        # also reset the cmd predictor?
        if self.reload_on_dsl_update:
            for param in self.cmd_vector.layers:
                print(type(param))
                self.initialize_weights(param)
            for param in self.after_attn_process.layers:
                print(type(param))
                self.initialize_weights(param)
            for param in self.param_predictor.layers:
                print(type(param))
                self.initialize_weights(param)
        
        self.command_tokens = new_command_tokens
        
    def extend_seq_batch(self, partial_seq, actions_in):

        actions_in = actions_in.long()
        batch_size, n_action_tokens = actions_in.size()

        cmd_in = actions_in[:, 0:1]

        cmd_token_embeddings = self.command_tokens(cmd_in)

        param_in = actions_in[:, 1:-1]
        param_token_embeddings = self.param_scale_tokens(param_in)

        all_embeddings = th.cat(
            (cmd_token_embeddings, param_token_embeddings), dim=1)
        all_embeddings = all_embeddings.reshape(
            batch_size, -1)
        token_embeddings = self.singular_token_mapper(all_embeddings)

        action_type = actions_in[:, -1:].bool()
        action_type = action_type.expand(-1, self.attn_size)
        token_embeddings = th.where(
            action_type, token_embeddings, cmd_token_embeddings[:, 0, :])
        token_embeddings = token_embeddings.unsqueeze(1)
        new_partial_seq = th.cat((partial_seq, token_embeddings), dim=1)
        return new_partial_seq

    def get_init_sequence(self, x_in):

        cnn_features = self.cnn_extractor.forward(x_in)
        out = self.pos_encoding(cnn_features)
        return out

    def attention_to_cmd_param(self, output):

        output = self.after_attn_process(output)

        cmd_vectors = self.cmd_vector(output)
        # Convert into distribution over the command tokens
        cmd_vectors_norms = th.norm(cmd_vectors, dim=1)
        cmd_vectors = cmd_vectors / cmd_vectors_norms.unsqueeze(1)

        all_commands = self.command_tokens.weight  # .detach()
        all_commands_norm = th.norm(all_commands, dim=1)
        all_cmd = all_commands / all_commands_norm.unsqueeze(1)

        cmd_sim = th.einsum("bk, mk -> bm", cmd_vectors, all_cmd)
        # cmd_distr = th.softmax(cmd_distr, dim = 1)

        param_output = self.param_predictor(output)
        param_output = param_output.reshape(-1, 5,
                                            self.param_scale_tokens.num_embeddings)

        cmd_sim = self.cmd_logsf_scaler * cmd_sim
        cmd_logsoft = self.cmd_logsoft(cmd_sim)
        # param_distr = th.softmax(param_output, dim = 2)
        param_logsoft = self.param_logsoft(param_output)

        return cmd_logsoft, param_logsoft

    def forward_beam(self, partial_sequence):
        total_len = partial_sequence.shape[1]

        attn_mask = self.attn_mask[:total_len, :total_len]

        for attn_layer in self.attn_layers:
            partial_sequence = attn_layer(partial_sequence, attn_mask, None)
        seq_out = partial_sequence[:, -1:, :]

        output = self.stack_all_vectors(seq_out)

        if len(output.shape) == 3:
            output = output.squeeze(1)
        #
        cmd_logsoft, param_logsoft = self.attention_to_cmd_param(output)
        # cmd_distr = th.softmax(cmd_distr, dim = 1)
        return cmd_logsoft, param_logsoft

    def vector_gather(self, vectors, indices):
        """
        Gathers (batched) vectors according to indices.
        Arguments:
            vectors: Tensor[N, L, D]
            indices: Tensor[N, K] or Tensor[N]
        Returns:
            Tensor[N, K, D] or Tensor[N, D]
        """
        N, L, D = vectors.shape
        squeeze = False
        if indices.ndim == 1:
            squeeze = True
            indices = indices.unsqueeze(-1)
        N2, K = indices.shape
        assert N == N2
        indices = einops.repeat(indices, "N K -> N K D", D=D)
        out = th.gather(vectors, dim=1, index=indices)
        if squeeze:
            out = out.squeeze(1)
        return out

    def stack_vectors(self, vectors, indices):
        output_list = []
        for ind in range(vectors.shape[0]):
            selected = vectors[ind, :indices[ind], :]
            output_list.append(selected)
        output = th.cat(output_list, 0)
        return output

    def stack_all_vectors(self, vectors):
        output = vectors.reshape(-1, vectors.shape[-1])
        return output


@ModelRegistry.register("NestedTransformer")
class NestedTransformer(BaseTransformer):

    def __init__(self, config):
        super(NestedTransformer, self).__init__(config)
        # Parameters:
        # Remove the param predictor
        # del self.param_predictor
        self.n_params = config.N_PARAMS
        self.param_seq_length = self.n_params + 1
        self.param_attn_layers = nn.ModuleList([AttnLayer(
            self.num_heads, self.attn_size, self.dropout) for _ in range(self.num_dec_layers)])
        param_attn_mask = self.generate_param_attn_mask()

        self.param_pos_encoding = LearnablePositionalEncoding(
            self.attn_size, self.dropout, max_len=self.param_seq_length)

        self.cmd_to_param_start = SEQ_MLP(
            [self.attn_size, self.attn_size, self.attn_size], self.dropout)

        self.register_buffer("param_attn_mask", param_attn_mask)
        self.apply(self.initialize_weights)

    def initialize_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)

    def generate_param_attn_mask(self):
        sz = self.param_seq_length
        mask = (th.triu(th.ones(sz, sz)) == 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0)).T
        # mask[:self.visual_seq_len, :self.visual_seq_len] = 0.
        return mask

    def forward_train(self, x_in, actions_in):

        token_embeddings, all_param_embeddings = self.embed_action_sequence(
            actions_in)

        # for normal sequence:
        cnn_features = self.cnn_extractor.forward(x_in)
        out = self.pos_encoding(
            th.cat((cnn_features, token_embeddings), dim=1))

        for attn_layer in self.attn_layers:
            out = attn_layer(out, self.attn_mask, self.key_mask)
        seq_out = out[:, self.visual_seq_len-1:-1, :]

        output = self.stack_all_vectors(seq_out)

        if len(output.shape) == 3:
            output = output.squeeze(1)

        cmd_logsoft = self.attention_to_cmd_param(output)

        # For param prediction attention:

        batch_size, max_action_size, _ = actions_in.size()

        all_param_embeddings = all_param_embeddings.reshape(
            -1, self.param_seq_length, self.attn_size)
        in_token = self.cmd_to_param_start(output.unsqueeze(1))
        all_param_embeddings = th.cat(
            [in_token, all_param_embeddings[:, 1:, :]], 1)
        param_out = self.param_pos_encoding(all_param_embeddings)
        # all_param_embeddings = all_param_embeddings.reshape(
        #     batch_size, max_action_size, self.param_seq_length + self.visual_seq_len, self.attn_size)
        # param_out = all_param_embeddings.reshape(-1, self.param_seq_length, self.attn_size)
        for attn_layer in self.param_attn_layers:
            param_out = attn_layer(
                param_out, self.param_attn_mask, self.key_mask)

        param_out = param_out.reshape(
            batch_size, max_action_size, self.param_seq_length, self.attn_size)
        param_out = param_out[:, :, :-1, :]
        # param_out = param_out.reshape(-1, self.attn_size)

        param_output = self.param_predictor(param_out)
        param_output = param_output.reshape(-1, self.n_params,
                                            self.param_scale_tokens.num_embeddings)
        param_logsoft = self.param_logsoft(param_output)
        # cmd_distr = th.softmax(cmd_distr, dim = 1)
        return cmd_logsoft, param_logsoft

    def embed_action_sequence(self, actions_in):

        batch_size, max_action_size, _ = actions_in.size()

        cmd_in = actions_in[:, :, 0:1]

        cmd_token_embeddings = self.command_tokens(cmd_in)

        param_in = actions_in[:, :, 1:-1]
        param_token_embeddings = self.param_scale_tokens(param_in)

        all_param_embeddings = th.cat(
            (cmd_token_embeddings, param_token_embeddings), dim=2)
        # add location information:
        embd_for_mapping = all_param_embeddings.reshape(
            batch_size, max_action_size, -1)

        token_embeddings = self.singular_token_mapper(embd_for_mapping)
        action_type = actions_in[:, :, -1:].bool()
        action_type = action_type.expand(-1, -1, self.attn_size)
        token_embeddings = th.where(
            action_type, token_embeddings, cmd_token_embeddings[:, :, 0, :])

        return token_embeddings, all_param_embeddings

    def extend_seq(self, partial_seq, cmd_in, param_in=None):

        cmd_token_embeddings = self.command_tokens(cmd_in)
        cmd_token_embeddings.unsqueeze_(0)
        if not param_in is None:
            param_token_embeddings = self.param_scale_tokens(param_in)
            all_embeddings = th.cat(
                (cmd_token_embeddings, param_token_embeddings), dim=0)
            all_embeddings = all_embeddings.reshape(
                1, -1)
            cmd_token_embeddings = self.singular_token_mapper(all_embeddings)
        # new extend the partial seq:
        # add position:
        self.pos_encoding.get_singular_position(
            cmd_token_embeddings, partial_seq.shape[1])
        new_partial_seq = th.cat(
            (partial_seq.clone(), cmd_token_embeddings), dim=0)
        return new_partial_seq

    def attention_to_cmd_param(self, output):

        output = self.after_attn_process(output)

        cmd_vectors = self.cmd_vector(output)
        # Convert into distribution over the command tokens
        cmd_vectors_norms = th.norm(cmd_vectors, dim=1)
        cmd_vectors = cmd_vectors / cmd_vectors_norms.unsqueeze(1)

        all_commands = self.command_tokens.weight.detach()
        all_commands_norm = th.norm(all_commands, dim=1)
        all_cmd = all_commands / all_commands_norm.unsqueeze(1)

        cmd_sim = th.einsum("bk, mk -> bm", cmd_vectors, all_cmd)
        # cmd_distr = th.softmax(cmd_distr, dim = 1)
        # this becomes based on param seq.
        cmd_sim = self.cmd_logsf_scaler * cmd_sim
        cmd_logsoft = self.cmd_logsoft(cmd_sim)
        return cmd_logsoft

    def forward_beam(self, partial_sequence):
        raise NotImplementedError
