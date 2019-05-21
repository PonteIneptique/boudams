import torch


pprint_2d = lambda x: [line for line in x.t().tolist() if not print(line)]
pprint_1d = lambda x: print(x.tolist())


class BaseSeq2SeqModel:
    """
    This contains base functionality to avoid some mystical conditions here and there

    """
    remove_first = False

    use_init: bool = True
    use_eos: bool = True

    def compute(
        self,
        src, src_len, trg=None,
        scorer=None, criterion=None,
        evaluate: bool = False
    ):

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
