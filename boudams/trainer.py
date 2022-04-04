import os
import logging
from collections import namedtuple
from typing import Optional, Union, List
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, EarlyStopping
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks import RichModelSummary


from boudams.tagger import BoudamsTagger


INVALID = "<INVALID>"
DEBUG = bool(os.getenv("DEBUG"))
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
        kwargs["callbacks"].extend([SaveModelCallback()])

        super().__init__(*args, **kwargs)
