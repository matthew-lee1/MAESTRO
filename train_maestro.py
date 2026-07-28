####################################################################################################
# 🎶 MAESTRO - MAsked Encoding Set TRansformer w/ self-DistillatiOn 🎶
# Author: Matthew E. Lee
# Advisors: E. John Wherry & Dokyoon Kim
# Contact: matthew.lee1@pennmedicine.upenn.edu
# train_maestro.py
####################################################################################################

import os
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch.distributed')
warnings.filterwarnings('ignore', message='.*Please use the new API settings.*')
warnings.filterwarnings('ignore', message='.*you have set wrong precision.*')
warnings.filterwarnings('ignore', message='.*CUDA device.*Tensor Cores.*')
warnings.filterwarnings('ignore', message='.*Tensor Cores.*')

import argparse
import torch
import torch.distributed as dist
import torch.utils.data as data
from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.strategies import DeepSpeedStrategy
from lightning.pytorch import callbacks
from lightning.pytorch.loggers import CSVLogger

from data.cytof_dataset import CyTOFDataset
from models.MAESTRO import MAESTROLightning
from configs.config import DeepSpeedConfig, UpdateTeacher, SinkhornCheckpoint


LOCAL_RANK = int(os.environ.get('LOCAL_RANK', 0))

def parse_args():
    parser = argparse.ArgumentParser(description='🎶MAESTRO🎶')
    parser.add_argument('--project', type=str, help='Project name', default=None)
    parser.add_argument('--devices', type=str, help='GPU devices', default='0')
    parser.add_argument('--accelerator', type=str, default='cuda', choices=['cuda', 'cpu', 'mps'],
                        help="Accelerator. 'cuda' uses DeepSpeed + bf16 (the manuscript setup); "
                             "'cpu'/'mps' run single-device fp32 without DeepSpeed, for the demo.")
    parser.add_argument('--data_dirs', type=str, nargs='+', required=True, help='Data directories (used for training)')
    parser.add_argument('--marker_dirs', type=str, nargs='+', default=None, help='Marker-only directories (included in shared-marker intersection but not used for training)')
    parser.add_argument('--subset_size', type=int, default=100000, help='Cells loaded per sample by the dataloader (the teacher sees this many)')
    parser.add_argument('--number_cells_subset', type=int, default=40000, help='Cells per sample')
    parser.add_argument('--dim_input', type=int, default=30, help='Input dimension per cell')
    parser.add_argument('--num_inds', type=int, default=16, help='IPAB inducing points')
    parser.add_argument('--dim_hidden', type=int, default=384, help='Hidden dimension')
    parser.add_argument('--dim_latent', type=int, default=256, help='Latent dimension')
    parser.add_argument('--num_heads', type=int, default=4, help='Attention heads')
    parser.add_argument('--ln', type=bool, default=True, help='Layer normalization')
    parser.add_argument('--initial_lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-12, help='Minimum learning rate')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--sinkhorn_start', type=int, default=25, help='Convert loss function to sampleloss Sinkhorn')
    parser.add_argument('--num_outputs', type=int, default=40000, help='Number of output tokens')
    parser.add_argument('--student_temperature', type=float, default=0.11, help='Student softmax temperature')
    parser.add_argument('--teacher_temperature', type=float, default=0.04, help='Teacher softmax temperature')
    parser.add_argument('--center_momentum', type=float, default=0.9, help='EMA momentum for teacher centering')
    parser.add_argument('--teacher_beta', type=float, default=0.99, help='EMA momentum for teacher weights')
    parser.add_argument('--mode', type=str, default='Train', help='Train or Validate')
    parser.add_argument('--cell_type_removal', type=str, nargs='+', default=None, help='Cell types to filter')
    parser.add_argument('--ckpt_resume', type=str, default='None', help='Checkpoint path to resume')

    return parser.parse_args()

def main():
    args = parse_args()

    output_path = os.path.join('output', args.project)
    os.makedirs(output_path, exist_ok=True)

    L.seed_everything(206, workers=True)

    dataset = CyTOFDataset(args.data_dirs, subset_size=args.subset_size, marker_dirs=args.marker_dirs, cell_type_removal=args.cell_type_removal)

    dim_input = len(dataset.shared_markers)

    if LOCAL_RANK == 0:
        print(f'Project: {args.project}')
        print(f'Training {len(dataset)} Samples 📏')
        print(f'dim_input inferred from shared markers: {dim_input}')

    model = MAESTROLightning(
        dim_input=dim_input,
        dim_output=dim_input,
        num_inds=args.num_inds,
        dim_hidden=args.dim_hidden,
        dim_latent=args.dim_latent,
        num_heads=args.num_heads,
        ln=args.ln,
        number_cells_subset=args.number_cells_subset,
        initial_lr=args.initial_lr,
        min_lr=args.min_lr,
        epochs=args.epochs,
        output_path=output_path,
        student_temperature=args.student_temperature,
        teacher_temperature=args.teacher_temperature,
        num_outputs=args.num_outputs,
        sinkhorn_start=args.sinkhorn_start,
        teacher_beta=args.teacher_beta,
        center_momentum=args.center_momentum,
    )
    
    every_10_callback = callbacks.ModelCheckpoint(
        dirpath=output_path,
        filename='{epoch:03d}',
        every_n_epochs=10,
        save_top_k=-1,
        save_last=False,
        save_weights_only=False,
        verbose=True,
        save_on_train_epoch_end=True,
    )

    checkpoint_callback = SinkhornCheckpoint(
        sinkhorn_start=args.sinkhorn_start,
        dirpath=output_path,
        filename='best-{epoch:03d}',
        monitor='train_loss_epoch',
        mode='min',
        save_top_k=1,
        save_last=True,
        save_weights_only=False,
        verbose=True,
        save_on_train_epoch_end=True,
    )

    deepspeed_config = DeepSpeedConfig()
    batch_size = deepspeed_config["train_micro_batch_size_per_gpu"]

    # DeepSpeed + bf16 is the manuscript configuration and requires CUDA. On cpu/mps we
    # fall back to a plain single-device fp32 trainer so the demo runs on a laptop.
    use_deepspeed = (args.accelerator == 'cuda')
    if use_deepspeed:
        trainer_devices = args.devices
        trainer_strategy = DeepSpeedStrategy(config=deepspeed_config)
        trainer_precision = 'bf16-mixed'
    else:
        trainer_devices = 1
        trainer_strategy = 'auto'
        trainer_precision = '32-true'


    if args.mode == 'Train':
        train_dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True,
            num_workers=8, pin_memory=True, prefetch_factor=2,
        )
        trainer = L.Trainer(
            devices=trainer_devices,
            accelerator=args.accelerator,
            strategy=trainer_strategy,
            precision=trainer_precision,
            max_epochs=args.epochs,
            min_epochs=min(300, args.epochs),
            enable_model_summary=False,
            enable_progress_bar=False,
            callbacks=[UpdateTeacher(), every_10_callback, checkpoint_callback],
            log_every_n_steps=1,
            logger=CSVLogger(save_dir='logs/', name=args.project),
        )
        if use_deepspeed:
            trainer.strategy.config['zero_force_ds_cpu_optimizer'] = False

        if args.ckpt_resume != 'None':
            trainer.fit(model=model, train_dataloaders=train_dataloader, ckpt_path=args.ckpt_resume)
        else:
            trainer.fit(model=model, train_dataloaders=train_dataloader)

    elif args.mode == 'Validate':
        train_set_size = int(len(dataset) * 0.9)
        valid_set_size = len(dataset) - train_set_size

        seed = torch.Generator().manual_seed(206)
        train_set, valid_set = data.random_split(dataset, [train_set_size, valid_set_size], generator=seed)

        train_dataloader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True, drop_last=True,
            num_workers=8, pin_memory=True, prefetch_factor=2,
        )
        valid_dataloader = DataLoader(
            valid_set, batch_size=batch_size, shuffle=False, drop_last=True,
            num_workers=8, pin_memory=True, prefetch_factor=2,
        )

        trainer = L.Trainer(
            devices=trainer_devices,
            accelerator=args.accelerator,
            strategy=trainer_strategy,
            precision=trainer_precision,
            max_epochs=args.epochs,
            min_epochs=min(300, args.epochs),
            enable_model_summary=False,
            enable_progress_bar=False,
            callbacks=[UpdateTeacher(), every_10_callback, checkpoint_callback],
            log_every_n_steps=1,
            logger=CSVLogger(save_dir='logs/'),
        )
        if use_deepspeed:
            trainer.strategy.config['zero_force_ds_cpu_optimizer'] = False
        trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

if __name__ == "__main__":
    main()