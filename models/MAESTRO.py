####################################################################################################
# 🎶 MAESTRO - MAsked Encoding Set TRansformer w/ self-DistillatiOn 🎶
# Author: Matthew E. Lee
# Advisors: E. John Wherry & Dokyoon Kim
# Contact: matthew.lee1@pennmedicine.upenn.edu
# MAESTRO.py
####################################################################################################

import os
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'OFF'
import time
import math
import umap
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import colormaps as cmaps

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from pytorch_lightning.utilities import rank_zero_only
from geomloss import SamplesLoss
from sklearn.decomposition import PCA
from entmax import sparsemax, entmax_bisect
from torch.utils.checkpoint import checkpoint

import warnings
warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True")
warnings.filterwarnings('ignore', category=UserWarning, module='torch.distributed')


import pykeops

def ruler_masking(X, mask_ratio):
    B, N, D = X.shape
    device = X.device
    N_mask = max(1, int(N * mask_ratio))

    Xf = X.float()
    X_centered = Xf - Xf.mean(dim=1, keepdim=True)

    _, _, Vh = torch.linalg.svd(X_centered, full_matrices=False)
    pc1 = Vh[:, 0, :]

    proj_1d = torch.einsum('bnd,bd->bn', X_centered, pc1)
    sorted_indices = proj_1d.argsort(dim=1)

    if torch.rand(1).item() > 0.5:
        mask_indices = sorted_indices[:, -N_mask:]
    else:
        mask_indices = sorted_indices[:, :N_mask]

    mask = torch.zeros(B, N, device=device, dtype=torch.bool)
    mask.scatter_(1, mask_indices, True)
    return mask

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False, softmax_type='regular', use_checkpoint=True, ent=1.15):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.dim_split = dim_V // num_heads
        self.softmax_type = softmax_type
        self.use_checkpoint = use_checkpoint
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.swig = SwiGLU(dim_V, dim_V)
        self.ent = ent

    def forward(self, Q, K, softmax_type=None):
        if softmax_type is None:
            softmax_type = self.softmax_type

        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        Q_ = torch.cat(Q.split(self.dim_split, 2), 0)
        K_ = torch.cat(K.split(self.dim_split, 2), 0)
        V_ = torch.cat(V.split(self.dim_split, 2), 0)

        S = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_split)

        if softmax_type == 'regular':
            A = torch.softmax(S, dim=2)
        elif softmax_type == 'sparse':
            A = entmax_bisect(S, alpha=self.ent, dim=2)
        else:
            raise ValueError(f"Unknown softmax_type: {softmax_type}, expected 'regular' or 'sparse'.")

        O = torch.cat((A.bmm(V_)).split(Q.size(0), 0), 2)
        O = Q + O
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + self.swig(O)
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O, A
        
class SAB(nn.Module):
    """
    Self-Attention Block (SAB) 🔄

    This class implements a self-attention mechanism using the MAB.
    """
    def __init__(self, dim_in, dim_out, num_heads, ln=False, softmax_type='regular'):
        """
        Initialize the SAB 🛠️

        Parameters:
        - dim_in: Input dimension ➡️
        - dim_out: Output dimension ➡️
        - num_heads: Number of attention heads 🧠
        - ln: Boolean indicating whether to apply layer normalization 🔄
        """
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln, softmax_type=softmax_type)

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
        return o, a
 
class IPAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=True, softmax_type='sparse'):
        super(IPAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, int(num_inds), dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.softmax_type = softmax_type
    
    def forward(self, X, use_self_attention=False):
        """
        Forward pass for the IPAB
        
        Parameters:
        - X: Input matrix [batch, seq_len, dim_in]
        - use_self_attention: If True, use X as keys/values (like SAB at inference).
                             If False, use learned inducing points I.
        
        Returns:
        - o: Output tensor
        - o_a: Attention weights
        """
        if use_self_attention:
            H = X
            h_a = None
        else:
            K = self.I.repeat(X.size(0), 1, 1)
            H, h_a = self.mab0(K, X, self.softmax_type)
        
        o, o_a = self.mab1(X, H, self.softmax_type)
        return o, o_a, h_a

class PMA(nn.Module):
    def __init__(self, dim, dim_latent, num_heads, num_seeds, ln=False, softmax_type='regular'):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim_latent))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim_latent, dim, dim_latent, num_heads, ln=ln, softmax_type=softmax_type) 
        self.softmax_type=softmax_type

    def forward(self, X):
        o, a = self.mab(self.S.repeat(X.size(0), 1, 1), X, self.softmax_type)
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
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

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
    def __init__(self, dim_input, dim_output, num_inds=16, dim_hidden=128, dim_latent=256, num_heads=1, num_seeds=1, num_outputs=30000, ln=True):
        super(SetTransformer, self).__init__()
        self.dim_hidden = dim_hidden
        self.dim_latent = dim_latent
        self.dim_input = dim_input
        self.num_seeds = num_seeds
        self.num_outputs = num_outputs

        # Encoder
        self.enc1 = nn.Linear(dim_input, dim_hidden)
        self.enc2 = IPAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln, softmax_type='sparse')
        self.enc3 = IPAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln, softmax_type='sparse')
        self.enc4 = IPAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln, softmax_type='sparse')
        self.enc5 = IPAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln, softmax_type='sparse')
        self.enc6 = IPAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln, softmax_type='sparse')
        
        # Pooling & Projection
        self.pma = PMA(dim_hidden, dim_latent, num_heads, num_seeds, ln=ln, softmax_type='sparse')
        self.project = nn.Linear(dim_latent, dim_latent)

        # Decoder
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim_latent))
        self.dec1a = PMA(dim_latent, dim_hidden, num_heads, num_outputs, ln=ln, softmax_type='regular')
        self.dec2 = SAB(dim_hidden, dim_hidden, num_heads, ln=ln, softmax_type='regular')
        self.dec3 = SAB(dim_hidden, dim_hidden, num_heads, ln=ln, softmax_type='regular')
        self.dec4 = SwiGLU(dim_hidden, dim_hidden, dim_output)

    def forward_encoder(self, X, use_self_attention=False):
        x = self.enc1(X)
        x, attn1, attn1_i = self.enc2(x, use_self_attention=use_self_attention)
        x, attn2, attn2_i = self.enc3(x, use_self_attention=use_self_attention)
        x, attn3, attn3_i = self.enc4(x, use_self_attention=use_self_attention)
        x, attn4, attn4_i = self.enc5(x, use_self_attention=use_self_attention)
        x, attn5, attn5_i = self.enc6(x, use_self_attention=use_self_attention)
        return x, attn1, attn2, attn3, attn1_i, attn2_i, attn3_i 
    
    def forward_pooling(self, X):
        x, pool_attn = self.pma(X)
        projection = F.softmax(self.project(x), dim=-1)
        return x, projection, pool_attn
    
    def forward_decoder(self, x, mask):
        N, total_length = mask.shape
        num_unmasked = (~mask).sum(dim=1)[0]
        x_repeated = x.repeat(1, num_unmasked, 1)
        mask_tokens = self.mask_token.expand(N, total_length, x.shape[-1])
        x_full = mask_tokens.clone()
        x_full[~mask] = x_repeated.reshape(-1, x.shape[-1])

        # Process through decoder layers
        x_decoded, _ = self.dec1a(x_full)
        x_decoded, _ = self.dec2(x_decoded)
        x_decoded, _ = self.dec3(x_decoded)
        x_decoded = self.dec4(x_decoded)
        return x_decoded

    def forward(self, X, mask):
        latent, attn1, attn2, *_, _, _, _ = self.forward_encoder(X)
        latent, projection, pool_attn = self.forward_pooling(latent)
        pred = self.forward_decoder(latent, mask)
        return pred, projection, latent, attn1, attn2, pool_attn

class MAESTRO(nn.Module):
    def __init__(self, dim_input, dim_output, num_inds, dim_hidden, dim_latent, num_heads, num_outputs, ln, number_cells_subset, student_temperature, teacher_temperature, sinkhorn_start=0):
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
        self.sinkhorn_start = sinkhorn_start
        self.energy_loss = SamplesLoss(loss="energy", p=2, verbose=False)
        self.sinkhorn_loss = SamplesLoss(loss="sinkhorn", p=2, verbose=False)
        self.use_sinkhorn = (sinkhorn_start == 0)
        self.student_temperature = student_temperature
        self.initial_teacher_temperature = teacher_temperature
        self.current_teacher_temperature = teacher_temperature
        self.temperature_step = 0.01  # Amount to increase per epoch
        self.centerlatent = torch.ones((1, dim_latent))
        self.cell_token = nn.Parameter(torch.zeros(1, dim_input))

        self.student = SetTransformer(dim_input=dim_input, 
                                      dim_output=dim_output, 
                                      num_inds=num_inds, 
                                      dim_hidden=dim_hidden, 
                                      dim_latent=dim_latent,
                                      num_heads=num_heads, 
                                      num_outputs=num_outputs, 
                                      ln=ln)
        
        self.teacher = SetTransformer(dim_input=dim_input, 
                                      dim_output=dim_output, 
                                      num_inds=num_inds, 
                                      dim_hidden=dim_hidden, 
                                      dim_latent=dim_latent,
                                      num_heads=num_heads, 
                                      num_outputs=num_outputs, 
                                      ln=ln)
        
        self._init_student()
        self._init_teacher()
    
    def _init_student(self):
        for m in self.student.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    with torch.no_grad():
                        nn.init.trunc_normal_(m.weight, std=.05)
                if m.bias is not None:
                    with torch.no_grad():
                        m.bias.data.fill_(0.0)
                if isinstance(m, nn.LayerNorm):
                    with torch.no_grad():
                        m.weight.data.fill_(1.0)

    def _init_teacher(self):
        for param_q, param_k in zip(self.student.parameters(), self.teacher.parameters()):
            param_k.data.copy_(param_q.data)  # Initialize
            param_k.requires_grad = False  # Not to be updated by gradient
      
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
        sharpened_output = torch.softmax(centered_output / self.current_teacher_temperature, dim=-1)
        return sharpened_output

    @torch.no_grad()
    def _update_teacher(self, beta=0.995):
        """
        Update the teacher model parameters with momentum 🛠️

        Parameters:
        - beta: Momentum factor 🔄
        """
        for (name_q, param_q), (name_k, param_k) in zip(self.student.named_parameters(), self.teacher.named_parameters()):
            param_q_data = param_q.data.to(param_k.device)  # Ensure param_q is on the same device as param_k
            param_k.data = (beta * param_k.data + (1. - beta) * param_q_data)
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
    
    def update_temperature(self):
        """
        Update the teacher temperature, linearly increasing it until it matches the student temperature
        """
        if self.current_teacher_temperature < self.student_temperature:
            self.current_teacher_temperature = min(
                self.current_teacher_temperature + self.temperature_step,
                self.student_temperature
            )

    def forward(self, X):
        B, N_total, D = X.shape
        N_sub = min(self.number_cells_subset, N_total)
        device = X.device

        # ── Cell subset ───────────────────────────────────────────────────────
        if N_total > N_sub:
            indices = torch.stack([torch.randperm(N_total, device=device)[:N_sub] for _ in range(B)])
        else:
            indices = torch.stack([torch.randint(0, N_total, (N_sub,), device=device) for _ in range(B)])
        X_sub = X[torch.arange(B, device=device).unsqueeze(1), indices]

        # ── Masking ───────────────────────────────────────────────────────────
        mask_rate = random.choice([0.0, 0.2, 0.4, 0.6, 0.8])
        mask = ruler_masking(X_sub, mask_rate)
        X_masked = X_sub[~mask].view(B, -1, D)

        # ── Student forward ───────────────────────────────────────────────────
        student_pred, student_projection, *_ = self.student(X_masked, mask)

        # ── Teacher forward ───────────────────────────────────────────────────
        with torch.no_grad():
            teacher_latent, *_ = self.teacher.forward_encoder(X)
            _, teacher_projection, _ = self.teacher.forward_pooling(teacher_latent)

        # ── Teacher updates ───────────────────────────────────────────────────
        self.update_center(teacher_projection, self.centerlatent)
        teacher_projection_sharpened = self.apply_centering_and_sharpening(teacher_projection, self.centerlatent)

        # ── Reconstruction loss (energy warmup → sinkhorn) ────────────────────
        loss_fn = self.sinkhorn_loss if self.use_sinkhorn else self.energy_loss
        sinkhorn_loss = loss_fn(X_sub.to(torch.float32), student_pred.to(torch.float32)).mean()

        # ── Self-distillation loss ────────────────────────────────────────────
        distillation_loss = self.calculate_kl_divergence(student_projection, teacher_projection_sharpened)

        # ── Total loss ────────────────────────────────────────────────────────
        loss = sinkhorn_loss + distillation_loss

        return loss, sinkhorn_loss, distillation_loss, X_masked, student_pred

class MAESTROLightning(L.LightningModule):
    def __init__(self, dim_input, dim_output, num_inds, dim_hidden, dim_latent, num_heads, num_outputs, ln, number_cells_subset, initial_lr, min_lr, epochs, student_temperature, teacher_temperature, output_path, sinkhorn_start=0):
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
                             self.hparams.sinkhorn_start)
        self.epoch_start_time = 0
        self.epoch_loss = []
        self.epoch_sinkhorn = []
        self.epoch_distillation = []
        self.best_loss = 1000000
        self.model_start_time = None
     
    def forward(self, X):
        return self.model(X)
    
    def training_step(self, batch, batch_idx):
        data_tensor, *_ = batch
        sinkhorn_start = self.hparams.sinkhorn_start
        self.model.use_sinkhorn = (sinkhorn_start == 0 or self.current_epoch >= sinkhorn_start)
        loss, sinkhorn_loss, distillation_loss, mask_only, pred_only = self.model(data_tensor)

        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=data_tensor.shape[0])
        self.epoch_loss.append(loss.detach())
        self.epoch_sinkhorn.append(sinkhorn_loss.detach())
        self.epoch_distillation.append(distillation_loss.detach())

        #torch.cuda.empty_cache()
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
            'sinkhorn_start':self.hparams.sinkhorn_start,
            'output_path':self.hparams.output_path,
        }
        torch.save(checkpoint, f'{self.hparams.output_path}/config.pth')
        self.model_start_time = time.time()

    @rank_zero_only
    def on_train_end(self):
        model_end_time = time.time()
        model_duration = model_end_time - self.model_start_time
        print(f'🕰️ Time to train entire model was {model_duration:.2f} seconds 🕰️')
        print(f'Training finished at 🗓️ {time.strftime("%a, %d %b %Y %H:%M:%S")}')
    
    @rank_zero_only
    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()

    @rank_zero_only
    def on_train_epoch_end(self):
        self.model.update_temperature()
        
        avg_loss = torch.stack(self.epoch_loss).mean().item()
        avg_sinkhorn_loss = torch.stack(self.epoch_sinkhorn).mean().item()
        avg_distillation_loss = torch.stack(self.epoch_distillation).mean().item()
        epoch_duration = (time.time() - self.epoch_start_time) / 60

        self.epoch_loss.clear()
        self.epoch_sinkhorn.clear()
        self.epoch_distillation.clear()

        loss_name = "sinkhorn" if self.model.use_sinkhorn else "energy"
        print(f"🎶 Epoch {self.current_epoch} [{loss_name}] | ⏱️  Duration: {epoch_duration:.2f} min | 💰 Loss: {avg_loss} | ⚡️ Recon: {avg_sinkhorn_loss:.4f} | ⚗️ Distillation: {avg_distillation_loss:.3e} 🎶\n")

        if self.current_epoch % 10 == 0:
            try:
                self._visualize_reconstructions(self.current_epoch)
            except Exception as e:
                print(f"⚠️ Visualization failed at epoch {self.current_epoch}: {e}")

    @rank_zero_only
    @torch.no_grad()
    def _visualize_reconstructions(self, epoch, n_samples=3, n_cells_viz=10000, mask_ratio=0.5):
        warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
        warnings.filterwarnings("ignore", category=UserWarning, module="umap")

        device = self.device
        model = self.model
        model.eval()

        cell_type_colors = {
            'T cell CD4 Naive': '#009874', 'T cell CD4 Mem': '#FF6F61',
            'T cell CD4 EMRA': '#5A5B9F', 'T cell CD8 Naive': '#BF1932',
            'T cell CD8 Mem': '#F5DF4D', 'T cell gd': '#D94F70',
            'T cell DN': '#6667AB', 'B cell': '#92A8D1',
            'Plasmablast': '#DECDBE', 'Monocyte Classical': '#88B04B',
            'Monocyte Nonclassical': '#9B1B30', 'mDC': '#F0C05A',
            'pDC': '#53B0AE', 'Neutrophil': '#0F4C81',
            'Eosinophil': '#F7CAC9', 'Basophil': '#5F4B8B',
            'NK cell': '#E2583E',
        }
        default_color = '#AAAAAA'
        status_colors = {'unmasked': '#FFB703', 'target': '#E63946', 'predicted': '#0077B6'}

        dataloader = self.trainer.train_dataloader
        all_batches, all_cell_types, all_names = [], [], []
        for batch in dataloader:
            data_tensor, cell_types_batch, sample_names = batch
            for i in range(data_tensor.shape[0]):
                all_batches.append(data_tensor[i])
                all_cell_types.append(cell_types_batch[i])
                all_names.append(sample_names[i])
            if len(all_batches) >= n_samples * 3:
                break

        if len(all_batches) < n_samples:
            model.train()
            return

        sample_indices = random.sample(range(len(all_batches)), n_samples)
        fig, axes = plt.subplots(n_samples, 5, figsize=(20, 4 * n_samples))

        col_titles = ['Cell Types', 'All (by status)', 'Unmasked', 'Target (masked)', 'Predicted']

        for row, sample_idx in enumerate(sample_indices):
            input_data = all_batches[sample_idx].to(device).unsqueeze(0)
            sample_cell_types = all_cell_types[sample_idx]
            sample_name = all_names[sample_idx]

            N_total = input_data.shape[1]
            N_sub = min(n_cells_viz, N_total)
            if N_total > N_sub:
                sub_idx = torch.randperm(N_total, device=device)[:N_sub].unsqueeze(0)
                input_data = input_data.gather(1, sub_idx.unsqueeze(-1).expand(-1, -1, input_data.shape[-1]))
                if isinstance(sample_cell_types, torch.Tensor):
                    sample_cell_types = sample_cell_types[sub_idx.squeeze(0).cpu()]

            mask = ruler_masking(input_data, mask_ratio)

            B, N, D = input_data.shape
            N_unmasked = (~mask).sum(dim=1)[0].item()
            N_masked = mask.sum(dim=1)[0].item()

            if N_masked == 0 or N_unmasked == 0:
                for col in range(5):
                    axes[row, col].set_xticks([])
                    axes[row, col].set_yticks([])
                continue

            model_dtype = next(model.parameters()).dtype
            unmasked = input_data[~mask].view(B, N_unmasked, D).to(model_dtype)
            masked_target = input_data[mask].view(B, N_masked, D)

            encoded, *_ = model.student.forward_encoder(unmasked)
            latent, _, _ = model.student.forward_pooling(encoded)
            pred_full = model.student.forward_decoder(latent, mask)
            # pred_full: [1, num_outputs, dim_output] — PMA produces fixed-size set output,
            # not one prediction per input cell, so we subsample N_masked points for visualization.
            pred_all = pred_full[0]  # [num_outputs, dim_output]
            perm = torch.randperm(pred_all.shape[0], device=pred_all.device)[:N_masked]
            pred = pred_all[perm].unsqueeze(0)  # [1, N_masked, dim_output]

            unmasked_np = unmasked.squeeze(0).float().cpu().numpy()
            target_np = masked_target.squeeze(0).float().cpu().numpy()
            pred_np = pred.squeeze(0).float().cpu().numpy()

            del unmasked, masked_target, encoded, latent, pred_full, pred
            torch.cuda.empty_cache()

            combined = np.vstack([unmasked_np, target_np, pred_np])
            n_u, n_t, n_p = len(unmasked_np), len(target_np), len(pred_np)

            mask_cpu = mask.squeeze(0).cpu()
            unmasked_indices = (~mask_cpu).nonzero(as_tuple=True)[0]
            masked_indices = mask_cpu.nonzero(as_tuple=True)[0]

            dataset = dataloader.dataset
            if isinstance(sample_cell_types, torch.Tensor):
                if hasattr(dataset, 'get_cell_type_name'):
                    unmasked_ct = [dataset.get_cell_type_name(sample_cell_types[i].item()) for i in unmasked_indices]
                    masked_ct = [dataset.get_cell_type_name(sample_cell_types[i].item()) for i in masked_indices]
                else:
                    unmasked_ct = [f'Unknown_{sample_cell_types[i].item()}' for i in unmasked_indices]
                    masked_ct = [f'Unknown_{sample_cell_types[i].item()}' for i in masked_indices]
            else:
                unmasked_ct = [sample_cell_types[i] for i in unmasked_indices]
                masked_ct = [sample_cell_types[i] for i in masked_indices]

            try:
                reducer = umap.UMAP(n_neighbors=30, min_dist=0.3)
                emb = reducer.fit_transform(combined)
            except Exception as e:
                print(f"⚠️ UMAP failed for {sample_name}: {e}")
                for col in range(5):
                    axes[row, col].set_xticks([])
                    axes[row, col].set_yticks([])
                continue

            emb_u = emb[:n_u]
            emb_t = emb[n_u:n_u + n_t]
            emb_p = emb[n_u + n_t:]

            x_min, x_max = emb[:, 0].min() - 0.5, emb[:, 0].max() + 0.5
            y_min, y_max = emb[:, 1].min() - 0.5, emb[:, 1].max() + 0.5

            ax = axes[row, 0]
            all_ct = unmasked_ct + masked_ct + ['predicted'] * n_p
            ct_colors = [cell_type_colors.get(ct, default_color) for ct in all_ct]
            ax.scatter(emb[:, 0], emb[:, 1], c=ct_colors, s=1.5, alpha=0.5)
            ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_ylabel(f'{sample_name[:25]}', fontsize=8)
            if row == 0: ax.set_title(col_titles[0], fontsize=9)

            ax = axes[row, 1]
            ax.scatter(emb_u[:, 0], emb_u[:, 1], c=status_colors['unmasked'], s=1.5, alpha=0.4, label='unmasked')
            ax.scatter(emb_t[:, 0], emb_t[:, 1], c=status_colors['target'], s=1.5, alpha=0.4, label='target')
            ax.scatter(emb_p[:, 0], emb_p[:, 1], c=status_colors['predicted'], s=1.5, alpha=0.4, label='predicted')
            ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(col_titles[1], fontsize=9)
                ax.legend(markerscale=5, fontsize=6, loc='upper right')

            ax = axes[row, 2]
            ax.scatter(emb_u[:, 0], emb_u[:, 1], c=status_colors['unmasked'], s=1.5, alpha=0.5)
            ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0: ax.set_title(col_titles[2], fontsize=9)

            ax = axes[row, 3]
            ax.scatter(emb_t[:, 0], emb_t[:, 1], c=status_colors['target'], s=1.5, alpha=0.5)
            ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0: ax.set_title(col_titles[3], fontsize=9)

            ax = axes[row, 4]
            ax.scatter(emb_p[:, 0], emb_p[:, 1], c=status_colors['predicted'], s=1.5, alpha=0.5)
            ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0: ax.set_title(col_titles[4], fontsize=9)

        legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=c, markersize=6, label=ct)
                           for ct, c in cell_type_colors.items()]
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                               markerfacecolor=default_color, markersize=6, label='Unknown'))
        fig.legend(handles=legend_elements, loc='center right',
                   bbox_to_anchor=(1.08, 0.5), fontsize=7, frameon=False)

        fig_dir = os.path.join(self.hparams.output_path, 'reconstruction_viz')
        os.makedirs(fig_dir, exist_ok=True)
        fig_path = os.path.join(fig_dir, f'epoch_{epoch:04d}.pdf')

        plt.suptitle(f'Reconstruction — Epoch {epoch} (mask={mask_ratio})', fontsize=14)
        plt.tight_layout(rect=[0, 0, 0.93, 0.97])
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"📊 Saved reconstruction visualization to {fig_path}")
        model.train()
