import os
import logging
from collections import namedtuple
from typing import Optional, Union, List
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, EarlyStopping
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBar, RichProgressBarTheme


from boudams.tagger import BoudamsTagger


INVALID = "<INVALID>"
ACCEPTABLE_MONITOR_METRICS = {"accuracy", "f1", "precision", "recall", "loss"}
Score = namedtuple("Score", ["loss", "accuracy", "precision", "recall", "fscore", "scorer"])

logger = logging.getLogger(__name__)


class SaveModelCallback(Callback):
    def on_validation_end(self, trainer: "Trainer", pl_module: BoudamsTagger) -> None:
        if not trainer.sanity_checking:
            #logger.info('Saving to {}_{}'.format(trainer.model_name, trainer.current_epoch))
            trainer.lightning_module.dump(f'{trainer.model_name}_{trainer.current_epoch}')


class Trainer(pl.Trainer):
    def __init__(
            self,
            model_name: Optional[str] = "model.boudams_model",
            monitor: str = "accuracy",
            patience: int = 5,
            min_delta: float = 0.01,
            callbacks: Optional[Union[List[Callback], Callback]] = None,
            *args,
            **kwargs
    ):
        self.model_name = model_name
        kwargs['logger'] = False
        kwargs['enable_checkpointing'] = False
        kwargs['callbacks'] = callbacks or []
        if not isinstance(kwargs['callbacks'], list):
            kwargs['callbacks'] = [kwargs['callbacks']]

        if monitor not in ACCEPTABLE_MONITOR_METRICS:
            raise ValueError(f"The monitor parameter can only be one of: {', '.join(ACCEPTABLE_MONITOR_METRICS)}")
        elif monitor not in {"loss", "f1"}:
            monitor = f"val_{monitor[:3]}"
        else:
            monitor = f"val_{monitor}"

        kwargs["callbacks"].extend([
            RichProgressBar(leave=True),
            SaveModelCallback(),
            EarlyStopping(monitor=monitor, min_delta=min_delta, patience=patience, verbose=False,
                          mode="min" if monitor == "val_loss" else "max")
        ])

        super().__init__(*args, **kwargs)
