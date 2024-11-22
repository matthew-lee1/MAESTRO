from lightning.pytorch.callbacks import Callback

def DeepSpeedConfig():
    deepspeed_config = {
    "zero_allow_untested_optimizer": True,
    "zero_optimization": {
        "stage": 3,  # Enable Stage 2 ZeRO (0/1/2/3 Disabled/Optimizer/Gradient/Weights state partitioning)
        "offload_optimizer": {"device" : "cpu",
                              "pin_memory" : True},
        "offload_param" : {"device" : "cpu",
                           "pin_memory" : True},       
        "contiguous_gradients": True,  # Reduce gradient fragmentation.
        "overlap_comm": True,  # Overlap reduce/backward operation of gradients for speed.
    },
    "bf16" : {
        "enabled" : True,
    },
    "gradient_clipping" : 5.0,
    "train_batch_size" : 4,
    "train_micro_batch_size_per_gpu" : 1,
}
    return deepspeed_config   

class UpdateTeacher(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs,  batch, batch_idx):
        pl_module.model._update_teacher()
