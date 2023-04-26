
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
        self.singular_token_mapper = SEQ_MLP(self.cmd_param_mapper_mlp, self.dropout)

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
        else:
            print("WUT")

    def enable_beam_mode(self):
        self.beam_mode = True
        self.beam_partial_init = True

    def disable_beam_mode(self):
        self.beam_mode = False
        self.beam_partial_init = False

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
        # mask[:self.visual_seq_len, :self.visual_seq_len] = 0.
        return mask

    def generate_key_mask(self, num, device):
        if num == self.key_mask.shape[0]:
            return self.key_mask
        else:
            sz = self.visual_seq_len + self.out_seq_len
            self.key_mask = th.zeros(num, sz).bool().to(device)

    def generate_start_token(self, num, max_action_size, device):
        if not num == self.start_token.shape[0]:
            self.start_token = th.LongTensor(
                [[self.command_tokens.num_embeddings - 1]]).to(device).repeat(num, 1)

    def forward_train(self, x_in, actions_in):

        token_embeddings = self.embed_action_sequence(actions_in)
        cnn_features = self.cnn_extractor.forward(x_in)
        out = self.pos_encoding(th.cat((cnn_features, token_embeddings), dim=1))

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
        new_partial_seq = th.cat((partial_seq.clone(), cmd_token_embeddings), dim=0)
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

        all_commands = self.command_tokens.weight.detach()
        all_commands_norm = th.norm(all_commands, dim=1)
        all_cmd = all_commands / all_commands_norm.unsqueeze(1)

        cmd_sim = th.einsum("bk, mk -> bm", cmd_vectors, all_cmd)
        # cmd_distr = th.softmax(cmd_distr, dim = 1)

        param_output = self.param_predictor(output)
        param_output = param_output.reshape(-1, 5,
                                            self.param_scale_tokens.num_embeddings)
        
        cmd_sim = 5 * cmd_sim
        cmd_logsoft = self.cmd_logsoft(cmd_sim)
        # param_distr = th.softmax(param_output, dim = 2)
        param_logsoft = self.param_logsoft(param_output)

        return cmd_logsoft, param_logsoft
    
    
    def forward_beam(self, partial_sequence):
        total_len = partial_sequence.shape[1]
        
        attn_mask = self.attn_mask[:total_len, :total_len]
        for attn_layer in self.attn_layers:
            out = attn_layer(partial_sequence, attn_mask, None)
        seq_out = out[:, -1:, :]

        output = self.stack_all_vectors(seq_out)

        if len(output.shape) == 3:
            output = output.squeeze(1)
        #
        cmd_logsoft, param_logsoft = self.attention_to_cmd_param(output)
        # cmd_distr = th.softmax(cmd_distr, dim = 1)
        return cmd_logsoft, param_logsoft



    def partial_beam_forward(self, x_in, y_in, y_length):
        assert not self.x_count is None, "Need to set x_count"
        if self.beam_partial_init:
            self.beam_partial_init = False
            self.cnn_features = self.cnn_extractor.forward(x_in).detach()

        # Replicate the cnn_features to get
        cnn_features = []
        for ind, count in enumerate(self.x_count):
            cnn_features.append(
                self.cnn_features[ind:ind+1].detach().expand(count, -1, -1))

        cnn_features = th.cat(cnn_features, 0)

        batch_size = y_in.shape[0]
        # Cut size:
        current_seq_len = y_length[0]

        self.generate_start_token(batch_size, y_in.device)
        y_in = th.cat([self.start_token, y_in], 1)
        y_in = y_in[:, :current_seq_len: -1]

        token_embeddings = self.token_embedding(y_in)

        out = self.pos_encoding(
            th.cat((cnn_features, token_embeddings), dim=1))

        # self.generate_key_mask(batch_size, y_in.device)
        # Cut size:
        total_len = self.visual_seq_len + (current_seq_len) + 1
        attn_mask = self.attn_mask[:total_len, :total_len]
        # key_mask = self.key_mask[:, :total_len].detach()
        for attn_layer in self.attn_layers:
            out = attn_layer(out, attn_mask, None)
        seq_out = out[:, self.visual_seq_len:, :]

        if self.return_all:
            raise ValueError("Cant use Return all with this mode")
        else:
            output = seq_out[:, -1]

        if len(output.shape) == 3:
            output = output.squeeze(1)
        output = self.attn_to_output(output)
        return output

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
