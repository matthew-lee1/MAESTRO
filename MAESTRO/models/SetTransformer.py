# https://github.com/juho-lee/set_transformer #
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from pytorch_lightning.utilities import rank_zero_only

class MAB(nn.Module):
    """
    Multi-Head Attention Block (MAB) 🎯

    This class implements a multi-head attention mechanism with optional layer normalization and ReLU activation.
    """
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        """
        Initialize the MAB 🛠️

        Parameters:
        - dim_Q: Dimension of query ➡️
        - dim_K: Dimension of key ➡️
        - dim_V: Dimension of value ➡️
        - num_heads: Number of attention heads 🧠
        - ln: Boolean indicating whether to apply layer normalization 🔄
        """
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)
        self.relu = nn.ReLU()

    def forward(self, Q, K):
        """
        Forward pass for the MAB 🔄

        Parameters:
        - Q: Query matrix ➡️
        - K: Key matrix ➡️

        Returns:
        - O: Output of the attention mechanism 🎯
        """
        Q = self.fc_q(Q)
        K, V = (self.fc_k(K), self.fc_v(K))
        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)
        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + self.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    """
    Self-Attention Block (SAB) 🔄

    This class implements a self-attention mechanism using the MAB.
    """

    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        """
        Initialize the SAB 🛠️

        Parameters:
        - dim_in: Input dimension ➡️
        - dim_out: Output dimension ➡️
        - num_heads: Number of attention heads 🧠
        - ln: Boolean indicating whether to apply layer normalization 🔄
        """
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        """
        Forward pass for the SAB 🔄

        Parameters:
        - X: Input matrix ➡️

        Returns:
        - Output of the self-attention mechanism 🎯
        """
        o, a = self.mab(X, X)
        o = o + X
        return (o, a)

class ISAB(nn.Module):
    """
    Induced Set-Attention Block (ISAB) 🔄

    This class implements an induced self-attention mechanism using the MAB.
    """

    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        """
        Initialize the ISAB 🛠️

        Parameters:
        - dim_in: Input dimension ➡️
        - dim_out: Output dimension ➡️
        - num_heads: Number of attention heads 🧠
        - num_inds: Number of inducing points 🔢
        - ln: Boolean indicating whether to apply layer normalization 🔄
        """
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        """
        Forward pass for the ISAB 🔄

        Parameters:
        - X: Input matrix ➡️

        Returns:
        - Output of the induced self-attention mechanism 🎯
        """
        H, h_A = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        o, o_a = self.mab1(X, H)
        o = o + X
        return (o, h_A, o_a)

class PMA(nn.Module):
    """
    Pooling by Multihead Attention (PMA) 🔄

    This class implements a pooling mechanism using multi-head attention.
    """

    def __init__(self, dim, dim_latent, num_heads, num_seeds, ln=False):
        """
        Initialize the PMA 🛠️

        Parameters:
        - dim: Dimension of input and output ➡️
        - num_heads: Number of attention heads 🧠
        - num_seeds: Number of seed vectors for pooling 🌱
        - ln: Boolean indicating whether to apply layer normalization 🔄
        """
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim_latent))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim_latent, dim, dim_latent, num_heads, ln=ln)

    def forward(self, X):
        """
        Forward pass for the PMA 🔄

        Parameters:
        - X: Input matrix ➡️

        Returns:
        - Output of the pooling mechanism 🎯
        """
        o, a = self.mab(self.S.repeat(X.size(0), 1, 1), X)
        return (o, a)

class SetTransformer(nn.Module):
    """
    SetTransformer for Classification.

    This class implements a Set Transformer model.
    It includes an encoder, pooling layer, and a classification head with softmax activation.
    """

    def __init__(self, dim_input, num_classes=18, num_inds=256, dim_hidden=2048, dim_latent=1024, num_heads=8, num_seeds=1, ln=True):
        super(SetTransformer, self).__init__()
        self.dim_hidden = dim_hidden
        self.dim_latent = dim_latent
        self.dim_input = dim_input
        self.num_seeds = num_seeds
        self.num_classes = num_classes
        self.enc1 = nn.Linear(dim_input, dim_hidden)
        self.enc2 = ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln)
        self.enc3 = ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln)
        self.pma = PMA(dim_hidden, dim_latent, num_heads, num_seeds, ln=ln)
        self.classifier = nn.Linear(dim_latent, num_classes)

    def forward_encoder(self, X):
        x = self.enc1(X)
        x, _, _ = self.enc2(x)
        x, _, _ = self.enc3(x)
        x, _ = self.pma(x)
        return x

    def forward(self, X):
        x = self.forward_encoder(X)
        logits = self.classifier(x.squeeze(1))
        probabilities = F.softmax(logits, dim=-1)
        return (logits, probabilities)