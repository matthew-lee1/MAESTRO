####################################################################################################
# 🎶 MAESTRO - MAsked Encoding Set TRandfOrmer 🎶
# Author: Matthew E. Lee
# Advisors: E. John Wherry & Dokyoon Kim
# Contact: matthew.lee1@pennmedicine.upenn.edu
####################################################################################################
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint


def DeepSpeedConfig():
    deepspeed_config = {
        "zero_allow_untested_optimizer": True,
        "zero_optimization": {
            "stage": 1,
            "contiguous_gradients": True,
            "overlap_comm": True,
        },
        "bf16": {
            "enabled": True,
        },
        "gradient_clipping": 5.0,
        "train_batch_size": 12,
        "train_micro_batch_size_per_gpu": 3,
    }
    return deepspeed_config

class UpdateTeacher(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs,  batch, batch_idx):
        pl_module.model._update_teacher()

class SinkhornCheckpoint(ModelCheckpoint):
    """Best-model checkpoint that ignores epochs before sinkhorn_start.

    During the energy-loss warmup the monitored metric is on a different scale,
    so any "best" score recorded there would prevent sinkhorn-phase models from
    ever being saved.  This subclass simply skips the save logic while
    current_epoch < sinkhorn_start, and resets best_model_score the first time
    the sinkhorn phase begins so the comparison starts fresh.
    """
    def __init__(self, sinkhorn_start: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.sinkhorn_start = sinkhorn_start
        self._switched = False

    def on_train_epoch_end(self, trainer, pl_module):
        if self.sinkhorn_start > 0 and trainer.current_epoch < self.sinkhorn_start:
            return  # skip entirely during warmup

        if self.sinkhorn_start > 0 and not self._switched:
            # First sinkhorn epoch — reset so warmup scores don't pollute the best
            self.best_model_score = torch.tensor(float('inf'))
            self.best_model_path = ''
            self._switched = True

        super().on_train_epoch_end(trainer, pl_module)


class GradientClipCallback(Callback):
    def __init__(self, clip_values={0: 1.0, 1: 2.0, 2: 3.0, 3: 5.0}):
        super().__init__()
        self.clip_values = clip_values
        
    def on_train_epoch_start(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        # Get clip value for current epoch, defaulting to 5.0 for later epochs
        new_clip_value = self.clip_values.get(current_epoch, 5.0)
        
        # Update DeepSpeed engine's gradient clipping
        if hasattr(trainer, 'strategy') and hasattr(trainer.strategy, 'model_engine'):
            trainer.strategy.model_engine.gradient_clipping = new_clip_value
