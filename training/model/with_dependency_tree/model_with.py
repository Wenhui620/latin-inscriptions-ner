import torch
import torch.nn as nn

class TreeLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.U_i = nn.Linear(input_dim, hidden_dim)
        self.U_f = nn.Linear(input_dim, hidden_dim)
        self.U_o = nn.Linear(input_dim, hidden_dim)
        self.U_u = nn.Linear(input_dim, hidden_dim)
        self.W_i = nn.Linear(hidden_dim, hidden_dim)
        self.W_f = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)
        self.W_u = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, child_h, child_c):
        if len(child_h) == 0:
            i = torch.sigmoid(self.U_i(x))
            o = torch.sigmoid(self.U_o(x))
            u = torch.tanh(self.U_u(x))
            c = i * u
            h = o * torch.tanh(c)
        else:
            child_h_mean = torch.mean(torch.stack(child_h), dim=0)
            i = torch.sigmoid(self.U_i(x) + self.W_i(child_h_mean))
            o = torch.sigmoid(self.U_o(x) + self.W_o(child_h_mean))
            u = torch.tanh(self.U_u(x) + self.W_u(child_h_mean))
            f = [torch.sigmoid(self.U_f(x) + self.W_f(h)) for h in child_h]
            c = i * u + torch.sum(torch.stack([f_k * c_k for f_k, c_k in zip(f, child_c)]), dim=0)
            h = o * torch.tanh(c)
        return h, c

class LatinTreeLSTMModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embed = nn.Embedding(config['word_vocab_size'], config['word_emb_dim'])
        self.pos_embed = nn.Embedding(config['pos_vocab_size'], config['pos_emb_dim'])
        self.deprel_embed = nn.Embedding(config['deprel_vocab_size'], config['deprel_dim'])
        self.char_embed = nn.Embedding(config['char_vocab_size'], config['char_emb_dim'])
        
        self.char_cnn = nn.Sequential(
            nn.Conv1d(config['char_emb_dim'], config['char_emb_dim'], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        
        input_dim = config['word_emb_dim'] + config['pos_emb_dim'] + config['deprel_dim'] + config['char_emb_dim']
        self.tree_lstm = TreeLSTM(input_dim, config['hidden_dim'])
        self.output = nn.Linear(config['hidden_dim'], config['ner_vocab_size'])
        
    def forward(self, trees):
        batch_outputs = []
        max_len = max(len(tree.get_leaf_nodes()) for tree in trees)
        
        for tree in trees:
            leaf_nodes = tree.get_leaf_nodes()
            node_outputs = []
            
            for node in leaf_nodes:
                h, _ = self._process_node(node)
                # 确保输出是2D张量 [1, ner_vocab_size]
                node_output = self.output(h).unsqueeze(0)
                node_outputs.append(node_output)
            
            # 将所有节点输出堆叠成 [seq_len, ner_vocab_size]
            if node_outputs:
                node_outputs = torch.cat(node_outputs, dim=0)
            else:
                node_outputs = torch.zeros((0, self.config['ner_vocab_size']), device=self.output.weight.device)
            
            # 处理填充
            if len(leaf_nodes) < max_len:
                padding = torch.zeros(
                    (max_len - len(leaf_nodes), 
                    self.config['ner_vocab_size']),
                    device=self.output.weight.device
                )
                full_output = torch.cat([node_outputs, padding], dim=0)
            else:
                full_output = node_outputs
            
            batch_outputs.append(full_output.unsqueeze(0))  # 添加批次维度
        
        # 最终形状: [batch_size, max_len, ner_vocab_size]
        return torch.cat(batch_outputs, dim=0)
    
    def _process_node(self, node):
        word_emb = self.word_embed(torch.tensor([node.word_index], device=self.word_embed.weight.device))
        pos_emb = self.pos_embed(torch.tensor([node.pos_index], device=self.pos_embed.weight.device))
        deprel_emb = self.deprel_embed(torch.tensor([node.deprel_index], device=self.deprel_embed.weight.device))
        
        char_input = torch.tensor(node.word_split, device=self.char_embed.weight.device).long()
        char_emb = self.char_embed(char_input).unsqueeze(0)
        char_emb = char_emb.permute(0, 2, 1)
        char_feat = self.char_cnn(char_emb).squeeze()
        
        x = torch.cat([word_emb, pos_emb, deprel_emb, char_feat.unsqueeze(0)], dim=-1)
        
        child_states = [self._process_node(child) for child in node.child_list]
        if child_states:
            child_h, child_c = zip(*child_states)
        else:
            child_h, child_c = [], []
        
        h, c = self.tree_lstm(x.squeeze(0), child_h, child_c)
        node.h, node.c = h.detach(), c.detach()
        return h, c