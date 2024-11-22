import os
import gc
import time
import math
import umap
import numpy as np
import matplotlib.pyplot as plt
import colormaps as cmaps

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from pytorch_lightning.utilities import rank_zero_only
from geomloss import SamplesLoss
from sklearn.decomposition import PCA

class MAB(nn.Module):
    """
    Multi-Head Attention Block (MAB) 🎯

    This class implements a multi-head attention mechanism with optional layer normalization and SwiGLU activation.
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
        self.fc_q = nn.Linear(dim_Q, dim_V, dtype=torch.bfloat16)
        self.fc_k = nn.Linear(dim_K, dim_V, dtype=torch.bfloat16)
        self.fc_v = nn.Linear(dim_K, dim_V, dtype=torch.bfloat16)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V, dtype=torch.bfloat16)
            self.ln1 = nn.LayerNorm(dim_V, dtype=torch.bfloat16)
        self.swig = SwiGLU(dim_V, dim_V)

    def forward(self, Q, K):
        """
        Forward pass for the MAB 🔄

        Parameters:
        - Q: Query matrix ➡️
        - K: Key matrix ➡️

        Returns:
        - O: Output of the attention mechanism 🎯
        - A: Attention scores
        """
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + self.swig(O)
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O, A

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
        - o: Output of the self-attention mechanism 🎯
        - a: Attention scores
        """
        o, a = self.mab(X, X)
        o = o + X
        return o, a
        
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
        - o: Output of the induced self-attention mechanism 🎯
        - h_a: Attention scores for the induced set
        - o_a: Attention scores for the original set
        """
        H, h_A = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        o, o_a = self.mab1(X, H) 
        o = o + X
        return o, h_A, o_a

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
        - o: Output of the pooling mechanism 🎯
        - a: Attention scores
        """
        o, a = self.mab(self.S.repeat(X.size(0), 1, 1), X)
        return o, a

class SwiGLU(nn.Module):
    """
    SwiGLU Activation 🔄

    This class implements the SwiGLU activation function.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, bias=True):
        """
        Initialize the SwiGLU 🛠️

        Parameters:
        - in_features: Input feature dimension ➡️
        - hidden_features: Hidden feature dimension (optional) ➡️
        - out_features: Output feature dimension (optional) ➡️
        - bias: Boolean indicating whether to include a bias term 🔄
        """
        super(SwiGLU, self).__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias, dtype=torch.bfloat16)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias, dtype=torch.bfloat16)

    def forward(self, x):
        """
        Forward pass for the SwiGLU 🔄

        Parameters:
        - x: Input tensor ➡️

        Returns:
        - Output tensor after applying the SwiGLU activation 🎯
        """
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)

class SetTransformer(nn.Module):
    """
    SetTransformer.

    This class implements a Set Transformer model for set-to-set learning. It includes an encoder, pooling layer, and decoder.
    """
    def __init__(self, dim_input, dim_output, num_inds=256, dim_hidden=2048, dim_latent=1024, num_heads=8, num_seeds=1, num_outputs=5000, ln=True):
        """
        Initialize the SetTransformer.

        Parameters:
        - dim_input: Dimension of input features.
        - dim_output: Dimension of output features.
        - num_inds: Number of inducing points for ISAB.
        - dim_hidden: Dimension of hidden layers.
        - dim_latent: Dimension of latent embedding. 
        - num_heads: Number of attention heads.
        - num_seeds: Number of seed vectors for PMA.
        - num_outputs: Number of output vectors for UpSAB.
        - ln: Boolean indicating whether to apply layer normalization.
        """
        super(SetTransformer, self).__init__()
        self.dim_hidden = dim_hidden
        self.dim_latent = dim_latent
        self.dim_input = dim_input
        self.num_seeds = num_seeds
        self.num_outputs = num_outputs

        # Encoder
        self.enc1 = nn.Linear(dim_input, dim_hidden)
        self.enc2 = ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln)
        self.enc3 = ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln)
        self.enc4 = ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln)
        
        # Pooling & Projection
        self.pma = PMA(dim_hidden, dim_latent, num_heads, num_seeds, ln=ln)
        self.project = nn.Linear(dim_latent, dim_latent)

        # Decoder
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim_latent)).to(torch.bfloat16)
        self.dec1 = PMA(dim_latent, dim_hidden, num_heads, num_outputs, ln=ln)
        self.dec2 = SAB(dim_hidden, dim_hidden, num_heads, ln=ln)
        self.dec3 = SAB(dim_hidden, dim_hidden, num_heads, ln=ln)
        self.dec4 = SAB(dim_hidden, dim_hidden, num_heads, ln=ln)
        self.dec5 = nn.Linear(dim_hidden, dim_output)
    
    def forward_encoder(self, X):
        """
        Forward pass for the encoder.

        Parameters:
        - X: Input tensor.

        Returns:
        - x: Latent representation after encoding.
        - projection: Projected latent representation (probabilities for self-distillation)
        - pool_attn: Attention scores for the pooling mechanism
        """

        x = X
        x = self.enc1(x)
        
        x, _, _ = self.enc2(x)
        x, _, _ = self.enc3(x)
        x, _, _ = self.enc4(x)
        x, pool_attn = self.pma(x)

        projection = F.relu(self.project(x))
        
        return x, projection, pool_attn
    
    def forward_decoder(self, x, mask):
        """
        Forward pass for the decoder.

        Parameters:
        - x: Latent representation.
        - mask: Mask tensor indicating which positions are masked.

        Returns:
        - Decoded output tensor.
        """
        
        N, _, D = x.shape
        x_expanded = torch.zeros(N, mask.shape[1], D).to(x.device)
        x_expanded = x_expanded.to(torch.bfloat16)
        mask_tokens = self.mask_token.expand(N, torch.sum(mask==1), D).to(x.device)

        for i in range(N):
            unmasked_indices = torch.where(mask[i] == 0)[0]
            masked_indices = torch.where(mask[i] == 1)[0]

            if len(unmasked_indices) > 0:
                x_expanded[i, unmasked_indices, :] = x[i, :len(unmasked_indices), :]
            if len(masked_indices) > 0:
                x_expanded[i, masked_indices, :] = mask_tokens[i, :len(masked_indices), :]

        # Apply decoder
        x_expanded = x_expanded
        x_decoded, _ = self.dec1(x_expanded)
        x_decoded, _ = self.dec2(x_decoded)
        x_decoded, _ = self.dec3(x_decoded)
        x_decoded, _ = self.dec4(x_decoded)
        x_decoded = self.dec5(x_decoded)

        return x_decoded

    def forward(self, X, mask):
        """
        Forward pass for the SetTransformer.

        Parameters:
        - X: Input tensor.
        - mask: Mask tensor indicating which positions are masked.

        Returns:
        - pred: Prediction tensor.
        - projection: Projected latent representation.
        - latent: Latent representation after encoding.
        """
        
        latent, projection, attn = self.forward_encoder(X)
        pred = self.forward_decoder(latent, mask)
        return pred, projection, latent, attn

class MAESTRO(nn.Module):
    """
    MAESTRO 🎼

    This class implements the MAESTRO model, which includes a student and teacher model based on SetTransformer.
    It uses masking, self-distillation, and reconstruction loss techniques for training.
    """
    def __init__(self, dim_input, dim_output, num_inds, dim_hidden, dim_latent, num_heads, num_outputs, ln, number_cells_subset, student_temperature, teacher_temperature):
        """
        Initialize the MAESTRO 🛠️

        Parameters:
        - dim_input: Dimension of input features ➡️
        - dim_output: Dimension of output features ➡️
        - num_inds: Number of inducing points for ISAB 🔢
        - dim_hidden: Dimension of hidden layers 🌐
        - num_heads: Number of attention heads 🧠
        - num_outputs: Number of output vectors 🔢
        - ln: Boolean indicating whether to apply layer normalization 🔄
        - number_cells_subset: Number of cells per sample 🔢
        - student_temperature: Temperature for softening probabilities (student) 🌡️
        - teacher_temperature: Temperature for softening probabilities (teacher) 🌡️
        """
        super(MAESTRO, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.num_inds = num_inds
        self.dim_hidden = dim_hidden
        self.dim_latent = dim_latent
        self.num_heads = num_heads
        self.num_outputs = num_outputs
        self.ln = ln
        self.number_cells_subset = number_cells_subset
        self.sinkhorn_loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.005, scaling=0.95, verbose=True)
        self.student_temperature = student_temperature
        self.teacher_temperature = teacher_temperature
        self.centerlatent = torch.ones((1, dim_latent))
        self.mask_token = nn.Parameter(torch.zeros(1, dim_input)).to(torch.bfloat16)

        self.student = SetTransformer(dim_input=dim_input, 
                                      dim_output=dim_output, 
                                      num_inds=num_inds, 
                                      dim_hidden=dim_hidden, 
                                      dim_latent=dim_latent,
                                      num_heads=num_heads, 
                                      num_outputs=num_outputs, 
                                      ln=ln,)
        
        self.teacher = SetTransformer(dim_input=dim_input, 
                                      dim_output=dim_output, 
                                      num_inds=num_inds, 
                                      dim_hidden=dim_hidden, 
                                      dim_latent=dim_latent,
                                      num_heads=num_heads, 
                                      num_outputs=num_outputs, 
                                      ln=ln,)
        
        self._init_student()
        self.student = self.student.to(torch.bfloat16)
        self._init_teacher()
    
    def _init_student(self):
        for m in self.student.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _init_teacher(self):
        for param_q, param_k in zip(self.student.parameters(), self.teacher.parameters()):
            param_k.data.copy_(param_q.data)  # Initialize
            param_k.requires_grad = False  # Not to be updated by gradient
    
    def random_masking(self, x, mask_ratio):
        """
        Apply random masking to the input tensor 🎭

        Parameters:
        - x: Input tensor ➡️
        - mask_ratio: Ratio of tokens to be masked ❓

        Returns:
        - x_masked_full: Masked input tensor 🎭
        - mask: Mask tensor indicating which positions are masked ❓
        - ids_restore: Indices to restore the original order 🔢
        """
        N, L, D = x.shape  # batch, length (num rows), dim
        device = x.device
        
        if mask_ratio == 0.0:
            perm = torch.randperm(L, device=device).expand(N, -1)
            x = x[torch.arange(N).unsqueeze(-1), perm.long()]
            mask = torch.zeros([N, L], device=device)
            return x, mask, torch.arange(L).expand(N, -1)
        
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        mask_token = self.mask_token.to(x.device)
        x_masked_full = x.clone().to(torch.bfloat16)  # Start with a clone of x
                
        for i in range(N):
            mask_token_expanded = mask_token.expand((L - len_keep, D))
            x_masked_full[i, mask[i].bool(), :] = mask_token_expanded  # Only overwrite the masked elements
        
        return x_masked_full, mask, ids_restore

    def non_random_block_masking(self, x, mask_ratio):
        """
        Apply non-random block masking to the input tensor based on similarity to a random row 🎭

        Parameters:
        - x: Input tensor ➡️
        - mask_ratio: Ratio of tokens to be masked ❓

        Returns:
        - x_masked_full: Masked input tensor 🎭
        - mask: Mask tensor indicating which positions are masked ❓
        - ids_restore: Indices to restore the original order 🔢
        """
        N, L, D = x.shape  # batch, length (num rows), dim
        device = x.device

        if mask_ratio == 0.0:
            perm = torch.randperm(L, device=device).expand(N, -1)
            x = x[torch.arange(N).unsqueeze(-1), perm.long()]
            mask = torch.zeros([N, L], device=device)
            return x, mask, torch.arange(L).expand(N, -1)

        len_keep = int(L * (1 - mask_ratio))

        # Pick a random row
        random_row_idx = torch.randint(0, L, (1,)).item()
        random_row = x[:, random_row_idx, :].unsqueeze(1)  # Shape: (N, 1, D)

        # Calculate cosine similarity between each row and the random row
        similarities = F.cosine_similarity(x, random_row, dim=-1)

        # Sort by similarity
        ids_sort = torch.argsort(similarities, dim=1, descending=True)
        ids_restore = torch.argsort(ids_sort, dim=1)

        # Apply block masking
        num_mask = int(L * mask_ratio)
        mask = torch.zeros([N, L], device=x.device)

        current_masked = 0
        block_size = num_mask // (L // num_mask)
        block_size = max(1, block_size)

        for start in range(0, L, 2 * block_size):
            end = min(start + block_size, L)
            mask[:, start:end] = 1
            current_masked += (end - start)
            if current_masked >= num_mask:
                break

        # Ensure the mask has exactly num_mask masked positions
        while mask.sum() > num_mask:
            excess = mask.sum() - num_mask
            for i in range(N):
                if excess > 0 and mask[i].sum() > 0:
                    mask_idx = (mask[i] == 1).nonzero(as_tuple=True)[0]
                    mask[i, mask_idx[0]] = 0
                    excess -= 1

        mask = torch.gather(mask, dim=1, index=ids_restore)  # Reorder mask according to the original order

        mask_token = self.mask_token.to(x.device)
        x_masked_full = x.clone().to(torch.bfloat16)

        for i in range(N):
            mask_count = int(mask[i].sum().item())
            mask_token_expanded = mask_token.expand(mask_count, D)
            x_masked_full[i, mask[i].bool(), :] = mask_token_expanded  # Only overwrite the masked elements

        return x_masked_full, mask, ids_restore

    def calculate_kl_divergence(self, student_logits, teacher_logits):
        """
        Calculate the KL divergence between student and teacher logits 🔄

        Parameters:
        - student_logits: Logits from the student model 🎓
        - teacher_logits: Logits from the teacher model 👨‍🏫

        Returns:
        - kl_div: KL divergence value 🔄
        """
        student_log_probs = F.log_softmax(student_logits / self.student_temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits, dim=-1) + 1e-9 # added for numerical stability
        kl_div = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        return kl_div

    def apply_centering_and_sharpening(self, teacher_output, teacher_center):
        """
        Apply centering and sharpening to the teacher output 🎯

        Parameters:
        - teacher_output: Output from the teacher model 👨‍🏫
        - teacher_center: Centering vector for the teacher output 🧩

        Returns:
        - sharpened_output: Sharpened and centered output 🎯
        """
        centered_output = teacher_output - teacher_center.unsqueeze(0)
        sharpened_output = torch.softmax(centered_output / self.teacher_temperature, dim=-1)
        return sharpened_output

    def forward_loss_sinkhorn(self, X, pred, mask):
        """
        Calculate the Sinkhorn loss 🔄

        Parameters:
        - X: Original input tensor ➡️
        - pred: Predicted output tensor 📈
        - mask: Mask tensor indicating which positions are masked ❓

        Returns:
        - loss: Sinkhorn loss value 🔄
        - X_masked: Masked original values 🎭
        - pred_masked: Masked predicted values 📈
        """
        X_masked = X[mask.bool()]
        pred_masked = pred[mask.bool()]  
        loss = self.sinkhorn_loss(X_masked, pred_masked)
        return loss, X_masked, pred_masked
 
    @torch.no_grad()
    def _update_teacher(self, beta=0.999):
        """
        Update the teacher model parameters with momentum while ensuring cleanup
        """
        for (name_q, param_q), (name_k, param_k) in zip(self.student.named_parameters(), self.teacher.named_parameters()):
            param_q_data = param_q.data.to(param_k.device)
            param_k.data = (beta * param_k.data + (1. - beta) * param_q_data).detach()
        return

    @torch.no_grad()
    def update_center(self, teacher_output, teacher_center, momentum=0.9):
        """
        Update the centering vector for the teacher output 🛠️

        Parameters:
        - teacher_output: Output from the teacher model 👨‍🏫
        - teacher_center: Centering vector for the teacher output 🧩
        - momentum: Momentum factor 🔄
        """
        current_mean = teacher_output.mean(dim=0)
        teacher_center.data = teacher_center.data.to(current_mean)
        teacher_center *= momentum
        teacher_center += (1 - momentum) * current_mean

    def forward(self, X, fps):
        """
        Forward pass for the MAESTRO model 🎼

        Parameters:
        - X: Input tensor ➡️
        - fps: Farthes point sampling tensor ➡️

        Returns:
        - loss: Combined loss value 🔄
        - avg_sinkhorn_loss: Average Sinkhorn loss value 🔄
        - distillation_loss: Distillation loss value 🔄
        - mask_only_fps: Mask tensor indicating which positions are masked ❓
        - pred_only_fps: Predicted output tensor 📈
        """

        def encode_global(input_tensor, mask_ratio):
            masked_input, mask, ids_restore = self.non_random_block_masking(input_tensor, mask_ratio)
            student_pred, student_projection, _, _ = self.student(masked_input, mask)
            return mask, ids_restore, student_pred, student_projection

        def encode_local(input_tensor, mask_ratios):
            local_projections = []
            for ratio in mask_ratios:
                with torch.set_grad_enabled(True):
                    masked_input, _, _ = self.non_random_block_masking(input_tensor, ratio)
                    _, student_projection, _ = self.student.forward_encoder(masked_input)
                    local_projections.append(student_projection)
                    del masked_input
            return local_projections

        def compute_losses(input_tensor, student_pred, mask):
            sinkhorn_loss, mask_only, pred_only = self.forward_loss_sinkhorn(input_tensor, student_pred, mask)
            return sinkhorn_loss, mask_only, pred_only
        
        if self.number_cells_subset > X.shape[1]:
            X_sub = X[:, np.random.choice(X.shape[1], self.number_cells_subset, replace=True), :]
        else:
            X_sub = X[:, np.random.choice(X.shape[1], self.number_cells_subset, replace=False), :]

        # Student Encode & Reconstruct 
        mask_global_a, ids_restore_a, student_pred_ga, student_projection_ga = encode_global(X_sub, 0.5)
        fps_mask_global_a, fps_ids_restore_a, student_fps_pred_ga, student_fps_projection_ga = encode_global(fps, 0.5)

        # Student Encode Only 
        mask_ratios = [0.2, 0.4, 0.6, 0.8]
        student_projections = []
        student_projections.extend(encode_local(X_sub, mask_ratios))
        student_projections.extend(encode_local(fps, mask_ratios))

        # Teacher Encode Only
        with torch.no_grad():
            X_full, _, _ = self.non_random_block_masking(X, 0.0)
            _, teacher_projection_ga, _ = self.teacher.forward_encoder(X_full)
            teacher_projection_ga = teacher_projection_ga.detach()

        # Teacher Updates
        self.update_center(teacher_projection_ga, self.centerlatent)
        teacher_a_outputs_projection = self.apply_centering_and_sharpening(teacher_projection_ga, self.centerlatent)

        # Calculate Reconstruction Losses
        sinkhorn_loss_x, _, _ = compute_losses(X_sub, student_pred_ga, mask_global_a)
        sinkhorn_loss_fps, mask_only_fps, pred_only_fps = compute_losses(fps, student_fps_pred_ga, fps_mask_global_a)
        avg_sinkhorn_loss = (sinkhorn_loss_x + sinkhorn_loss_fps) / 2
       
        # Calculate Self-Distillation Loss
        distillation_loss = i = 0
        for student_projection in [student_projection_ga, student_fps_projection_ga] + student_projections:
            distillation_loss += self.calculate_kl_divergence(student_projection, teacher_a_outputs_projection)
            i += 1
        distillation_loss /= i

        # Final Loss Calculation
        loss = avg_sinkhorn_loss + distillation_loss

        return loss, avg_sinkhorn_loss, distillation_loss, mask_only_fps, pred_only_fps

class MAESTROLightning(L.LightningModule):
    def __init__(self, dim_input, dim_output, num_inds, dim_hidden, dim_latent, num_heads, num_outputs, ln, number_cells_subset, initial_lr, min_lr, epochs, student_temperature, teacher_temperature, output_path):
        super().__init__()
        self.save_hyperparameters()
        self.model = MAESTRO(self.hparams.dim_input, 
                             self.hparams.dim_output, 
                             self.hparams.num_inds, 
                             self.hparams.dim_hidden, 
                             self.hparams.dim_latent,
                             self.hparams.num_heads, 
                             self.hparams.num_outputs, 
                             self.hparams.ln, 
                             self.hparams.number_cells_subset, 
                             self.hparams.student_temperature, 
                             self.hparams.teacher_temperature,
                             self.hparams.output_path)
        self.epoch_start_time = 0
        self.epoch_loss = []
        self.epoch_sinkhorn = []
        self.epoch_distillation = [] 
        self.best_loss = 1e8
        self.model_start_time = None
        
    def forward(self, X, fps, sc):
        return self.model(X, fps, sc)
    
    def training_step(self, batch, batch_idx):
        data_tensor, fps_tensor, sample_name = batc
        loss, sinkhorn_loss, distillation_loss, _, _  = self.model(data_tensor, fps_tensor)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=1)
        self.log('sinkhorn_loss', sinkhorn_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=1)
        self.log('distillation_loss', distillation_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=1)
        self.epoch_loss.append(loss.detach())
        self.epoch_sinkhorn.append(sinkhorn_loss.detach())
        self.epoch_distillation.append(distillation_loss.detach())
    
        if self.current_epoch % 10 == 0:
            mask_only = mask_only.cpu().detach()
            pred_only = pred_only.cpu().detach()
            self.plot_pca(mask_only, pred_only, self.current_epoch, sample_name[0], self.hparams.output_path)
            self.plot_umap(mask_only, pred_only, self.current_epoch, sample_name[0], self.hparams.output_path)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.initial_lr, weight_decay=1e-3)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.epochs, eta_min=self.hparams.min_lr),
            'interval': 'epoch',  
            'frequency': 1,
            'strict': True,
        }
        return [optimizer], [scheduler]
    
    @rank_zero_only
    def on_train_start(self):
        print(f'🎼 MAESTRO 🎼')
        print(f'🏋🏻‍♂️  Beginning training at 🗓️  {time.strftime("%a, %d %b %Y %H:%M:%S")}')
        checkpoint = {
            'dim_input':self.hparams.dim_input,
            'dim_output':self.hparams.dim_output,
            'num_inds':self.hparams.num_inds,
            'dim_hidden':self.hparams.dim_hidden,
            'dim_latent':self.hparams.dim_latent,
            'num_heads':self.hparams.num_heads,
            'num_outputs':self.hparams.num_outputs,
            'ln':self.hparams.ln,
            'number_cells_subset':self.hparams.number_cells_subset, 
            'initial_lr':self.hparams.initial_lr, 
            'min_lr':self.hparams.min_lr,
            'epochs':self.hparams.epochs,
            'student_temperature':self.hparams.student_temperature,
            'teacher_temperature':self.hparams.teacher_temperature,
            'output_path':self.hparams.output_path,
        }
        torch.save(checkpoint, f'{self.hparams.output_path}/config.pth')
        self.model_start_time = time.time()

    @rank_zero_only
    def on_train_end(self):
        model_end_time = time.time()
        model_duration = self.model_start_time - model_end_time
        print(f'🕰️ Time to train entire model was {model_duration:.2f} seconds 🕰️')
        print(f'Training finished at 🗓️ {time.strftime("%a, %d %b %Y %H:%M:%S")}')
    
    @rank_zero_only
    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()

    @rank_zero_only
    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.epoch_loss).mean().item()
        avg_sinkhorn_loss = torch.stack(self.epoch_sinkhorn).mean().item()
        avg_distillation_loss = torch.stack(self.epoch_distillation).mean().item()
        epoch_duration = (time.time() - self.epoch_start_time) / 60

        self.epoch_loss.clear()
        self.epoch_sinkhorn.clear()
        self.epoch_distillation.clear()

        print(f"🎶 Epoch {self.current_epoch} | ⏱️  Duration: {epoch_duration:.2f} min | 💰 Loss: {avg_loss} | ⚡️⬇️  Sinkhorn: {avg_sinkhorn_loss} | ⚗️ Distillation: {avg_distillation_loss} 🎶")

        torch.cuda.empty_cache()
        gc.collect()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
            torch.cuda.empty_cache()

    #@rank_zero_only
    def plot_pca(self, mask_only, pred_only, epoch, sample_name, output_dir):

        mask_only_np = mask_only.to(dtype=torch.float32).numpy()
        pred_only_np = pred_only.to(dtype=torch.float32).numpy()

        pca = PCA(n_components=2, random_state=253)
        mask_only_pca = pca.fit_transform(mask_only_np)
        pred_only_pca = pca.transform(pred_only_np)

        plt.figure(figsize=(10, 10))
        plt.scatter(mask_only_pca[:, 0], mask_only_pca[:, 1], c=mask_only_pca[:, 0], cmap=cmaps.l_orangesat1, label='Mask Only', alpha=0.5, marker='o')
        plt.scatter(pred_only_pca[:, 0], pred_only_pca[:, 1], c=pred_only_pca[:, 0], cmap=cmaps.turqw1, label='Pred Only', alpha=0.5, marker='^')
        plt.title(f'PCA Projection - {sample_name}', fontsize=25)
        plt.xlabel('PC1', fontsize=20)
        plt.ylabel('PC2', fontsize=20)
        plt.legend(prop={'size': 15})
        os.makedirs(f'{output_dir}/pca/epoch_{epoch}', exist_ok=True)
        plt.savefig(f'{output_dir}/pca/epoch_{epoch}/{sample_name}_PCA.png')
        plt.close()

    def plot_umap(self, mask_only, pred_only, epoch, sample_name, output_dir):
        mask_only_np = mask_only.to(dtype=torch.float32).cpu().numpy()
        pred_only_np = pred_only.to(dtype=torch.float32).cpu().numpy()

        # Combine the data for UMAP
        combined_data = np.vstack((mask_only_np, pred_only_np))

        # Perform UMAP
        reducer = umap.UMAP(n_neighbors=25, min_dist=0.4, n_components=2)
        embedding = reducer.fit_transform(combined_data)

        # Split the embedding back into mask_only and pred_only
        mask_only_umap = embedding[:len(mask_only_np)]
        pred_only_umap = embedding[len(mask_only_np):]

        plt.figure(figsize=(10, 10))
        plt.scatter(mask_only_umap[:, 0], mask_only_umap[:, 1], c=mask_only_umap[:, 0], cmap=cmaps.l_orangesat1, label='Mask Only', alpha=0.5, marker='o')
        plt.scatter(pred_only_umap[:, 0], pred_only_umap[:, 1], c=pred_only_umap[:, 0], cmap=cmaps.turqw1, label='Pred Only', alpha=0.5, marker='^')
        plt.title(f'UMAP Projection - {sample_name}', fontsize=25)
        plt.xlabel('UMAP1', fontsize=20)
        plt.ylabel('UMAP2', fontsize=20)
        plt.legend(prop={'size': 15})
        os.makedirs(f'{output_dir}/umap/epoch_{epoch}', exist_ok=True)
        plt.savefig(f'{output_dir}/umap/epoch_{epoch}/{sample_name}_UMAP.png')
        plt.close()