import torch
import torch.nn as nn

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..trainer import Scorer

pprint_2d = lambda x: [line for line in x.t().tolist() if not print(line)]
pprint_1d = lambda x: print(x.tolist())


class BaseSeq2SeqModel(nn.Module):
    """
    This contains base functionality to avoid some mystical conditions here and there

    """
    remove_first = False

    use_init: bool = True
    use_eos: bool = True
    batch_first = False

    def predict(self, src, src_len) -> torch.Tensor:
        """ Predicts value for a given tensor

        :param src: tensor(sentence len x batch size)
        :param src_len: tensor(batch size)
        :return:  tensor(output len x batch size)
        """
        return self(src, src_len, None, teacher_forcing_ratio=0)[0]

    def gradient(
        self,
        src, src_len, trg=None,
        scorer: "Scorer" = None, criterion=None,
        evaluate: bool = False
    ):
        """ Performs a gradient on a batch

        :param src: tensor(sentence len x batch size)
        :param src_len: tensor(batch size)
        :param trg: Optional[tensor(output length x batch_size)]
        :param scorer: Scorer to register batches
        :param criterion: Loss
        :param evaluate: Whether or not we evaluate things
        :return: tensor(output length x batch size x decoder dimension)
        """

        kwargs = {}
        if evaluate:
            kwargs = dict(teacher_forcing_ratio=0)

        output, attention = self(src, src_len, trg, **kwargs)

        # We register the current batch
        #  For this to work, we get ONLY the best score of output which mean we need to argmax
        #   at the second layer (base 0 I believe)
        # We basically get the best match at the output dim layer : the best character.

        # The prediction and ground truth batches NECESSARLY starts by "0" where
        #    0 is the SOS token. In order to have a score independant from hardcoded ints,
        #    we remove the first element of each sentence

        scorer.register_batch(
            torch.argmax(output, 2)[1:],
            trg[1:],
            remove_first=self.remove_first
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

    @staticmethod
    def argmax(out: torch.Tensor):
        return torch.argmax(out, 2)[1:].t()
