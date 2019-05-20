import torch


class BaseSeq2SeqModel:
    """
    This contains base functionality to avoid some mystical conditions here and there

    """
    @staticmethod
    def _reshape_input(src: torch.Tensor, trg: torch.Tensor):
        if trg is None:
            return src
        return src, trg

    @staticmethod
    def _reshape_out_for_loss(out: torch.Tensor, trg: torch.Tensor):
        # Remove the first because it's AUTOMATICALLY <SOS> for non-CONV models

        # Contiguous is not normally used in the original code.
        # ToDo: Understand why somehow it become something that needed to be added
        return out[1:].contiguous().view(-1, out.shape[-1]), \
               trg[1:].contiguous().view(-1)

    @staticmethod
    def _reshape_output_for_scorer(out: torch.Tensor, trg: torch.Tensor = None):
        # Remove the score from every prediction, keep the best one
        if trg is None:
            return torch.argmax(out, 2)[1:]
        return torch.argmax(out, 2)[1:], trg[1:]
