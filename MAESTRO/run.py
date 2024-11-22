import os
import argparse

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.strategies import DeepSpeedStrategy
from lightning.pytorch import callbacks

from data.CyTOFDataset import CyTOFDataset
from models.MAESTRO import MAESTROLightning
from configs.config import DeepSpeedConfig, UpdateTeacher

def parse_args():
    parser = argparse.ArgumentParser(description='MAESTRO')
    ### Project and device settings ##########################################################################################################
    parser.add_argument('--project', type=str, help='Project name for the run', default=None)
    parser.add_argument('--devices', type=str, help='Number of GPUs used', default='0')
    ### Data #################################################################################################################################
    parser.add_argument('--data_dir', type=str, default=None, help='Directory with the data')
    ### Model and training configurations ####################################################################################################
    parser.add_argument('--number_cells_subset', type=int, default=10000, help='Number of cells per sample')
    parser.add_argument('--dim_input', type=int, default=30, help='Features per input (dimension of one set element)')
    parser.add_argument('--dim_output', type=int, default=30, help='Features per output (dimension of one set element)')
    parser.add_argument('--num_inds', type=int, default=2500, help='Number of token elements')
    parser.add_argument('--dim_hidden', type=int, default=2048, help='Embedding size for the teacher')
    parser.add_argument('--dim_latent', type=int, default=1024, help='Embedding size for the latent')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num_outputs', type=int, default=10000, help='Number of output elements, typically equals number_cells_per_sample')
    parser.add_argument('--ln', type=bool, default=True, help='Use Layer Normalization')
    parser.add_argument('--initial_lr', type=float, default=1e-5, help='Initial learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-9, help='Minimum learning rate')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train')
    parser.add_argument('--student_temperature', type=float, default=0.1, help='Temperature for softening probabilities (student)')
    parser.add_argument('--teacher_temperature', type=float, default=0.07, help='Temperature for softening probabilities (teacher)')
    parser.add_argument('--mode', type=str, default='Train', help='Options = Train, Validate, Test')
    return parser.parse_args()

def main():
    args = parse_args()

    output_path = os.path.join('/home/lematthe/MAESTRO/output/training/', args.project)
    os.makedirs(output_path, exist_ok=True)
    print(f'Project: {args.project}')

    L.seed_everything(206, workers=True)
    torch.set_float32_matmul_precision('high')
    
    dataset = CyTOFDataset(args.data_dir)
    print(f'Training {dataset.__len__()} Samples 📏')
    model = MAESTROLightning(dim_input=args.dim_input,
                dim_output=args.dim_output,
                num_inds=args.num_inds,
                dim_hidden=args.dim_hidden,
                dim_latent=args.dim_latent,
                num_heads=args.num_heads,
                num_outputs=args.num_outputs,
                ln=args.ln,
                number_cells_subset=args.number_cells_subset, 
                initial_lr = args.initial_lr, 
                min_lr = args.min_lr,
                epochs=args.epochs,
                student_temperature=args.student_temperature,
                teacher_temperature=args.teacher_temperature,
                output_path=output_path)

    checkpoint_callback = callbacks.ModelCheckpoint(output_path, monitor='train_loss', save_top_k=2, save_last=True, mode='min', save_weights_only=False, every_n_epochs=1, verbose=True)
    deepspeed_config = DeepSpeedConfig()
    
    if args.mode == 'Train':
        train_dataloader = DataLoader(dataset, batch_size=deepspeed_config["train_micro_batch_size_per_gpu"], shuffle=True, drop_last=True, num_workers=12, pin_memory=True, prefetch_factor=2)
        trainer = L.Trainer(devices=args.devices,
                            accelerator='cuda',
                            strategy=DeepSpeedStrategy(config=deepspeed_config),
                            precision='bf16-true',
                            max_epochs=args.epochs,
                            min_epochs=300,
                            enable_model_summary=True,
                            enable_progress_bar=False,
                            callbacks=[UpdateTeacher(), checkpoint_callback],
                            log_every_n_steps=1,
                            )
        trainer.strategy.config['zero_force_ds_cpu_optimizer'] = False
        trainer.fit(model=model, train_dataloaders=train_dataloader)

    elif args.mode == 'Validate':
        train_set_size = int(len(dataset) * 0.8)
        valid_set_size = len(dataset) - train_set_size
        seed = torch.Generator().manual_seed(206)
        train_set, valid_set = data.random_split(dataset, [train_set_size, valid_set_size], generator=seed)
        train_dataloader = DataLoader(train_set, batch_size=1, shuffle=True, drop_last=True)
        valid_dataloader = DataLoader(valid_set, batch_size=1, shuffle=False, drop_last=True)
        trainer = L.Trainer(devices=args.devices,
                            accelerator='cuda',
                            strategy=DeepSpeedStrategy(config=deepspeed_config),
                            precision='bf16-true',
                            max_epochs=args.epochs,
                            min_epochs=300,
                            enable_model_summary=True,
                            enable_progress_bar=False,
                            callbacks=[UpdateTeacher(), checkpoint_callback],
                            log_every_n_steps=1,
                            )
        trainer.strategy.config['zero_force_ds_cpu_optimizer'] = False
        trainer.fit(model=model, train_dataloaders=train_dataloader, valid_dataloaders=valid_dataloader)

if __name__ == "__main__":
    main()