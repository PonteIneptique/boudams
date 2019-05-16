import torch


class BaseSeq2SeqModel:
    """
    This contains base functionality to avoid some mystical conditions here and there

    """
    @staticmethod
    def _reshape_input(src: torch.Tensor, trg: torch.Tensor):
        return src, trg

    @staticmethod
    def _reshape_out_for_loss(out: torch.Tensor, trg: torch.Tensor):
        return out[1:].view(-1, out.shape[-1]), \
               trg[1:].view(-1)

    @staticmethod
    def _reshape_output_for_scorer(out: torch.Tensor):
        # Remove the score from every prediction, keep the best one
        return torch.argmax(out, 2)
