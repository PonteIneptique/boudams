import torch
import torch.nn as nn

from typing import TYPE_CHECKING, List, Optional
if TYPE_CHECKING:
    from ..trainer import Scorer
    from ..encoder import LabelEncoder

pprint_2d = lambda x: [line for line in x.t().tolist() if not print(line)]
pprint_1d = lambda x: print(x.tolist())


class BaseSeq2SeqModel(nn.Module):
    """
    This contains base functionality to avoid some mystical conditions here and there

    """
    remove_first = False
    masked_only = False

    use_init: bool = True
    use_eos: bool = True

    def predict(self, src, src_len, label_encoder: "LabelEncoder",
                override_src: Optional[List[str]] = None) -> List[List[str]]:
        """ Predicts value for a given tensor

        :param src: tensor(batch size x sentence_length)
        :param src_len: tensor(batch size)
        :param label_encoder: Encoder
        :return: tensor(batch_size x sentence_length)
        """
        out, _ = self(src.t(), src_len, None, teacher_forcing_ratio=0)
        # Out is (sentence_length, batch_size) so we transpose it
        logits = torch.argmax(out, 2)[1:].t()  # Remove SOS token
        return label_encoder.reverse_batch(logits)

    def init_weights(self):
        pass

    def gradient(
        self,
        src, src_len, trg=None,
        scorer: "Scorer" = None, criterion=None,
        evaluate: bool = False
    ):
        """ Performs a gradient on a batch

        :param src: tensor(sentence length x batch_size)
        :param src_len: tensor(batch size)
        :param trg: Optional[tensor(sentence length x batch size)]
        :param scorer: Scorer to register batches
        :param criterion: Loss
        :param evaluate: Whether or not we evaluate things
        :return: tensor(output length x batch size x decoder dimension)
        """

        src, trg = src.t(), trg.t()

        kwargs = {}
        if evaluate:
            kwargs = dict(teacher_forcing_ratio=0)

        output, attention = self(src, src_len, trg, **kwargs)

        scorer.register_batch(
            torch.argmax(output, 2).t(),
            trg.t()
        )

        # trg = [trg sent len, batch size]
        # output = [trg sent len, batch size, output dim]

        # trg = [(trg sent len - 1) * batch size]
        # output = [(trg sent len - 1) * batch size, output dim]

        # About contiguous : https://stackoverflow.com/questions/48915810/pytorch-contiguous
        # Basically, elements of the tensor are spread over memory and to make it VERY simple, it's a bit like
        #   deepcopy.
        loss = criterion(
            output.contiguous()[1:].view(-1, output.shape[-1]),
            trg.contiguous()[1:].view(-1)
        )

        return loss
