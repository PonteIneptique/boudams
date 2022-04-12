from typing import Optional, Callable, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils


__all__ = [
    "ModelWrapper",
    "PosEmbedding",
    "Conv", "SequentialConv",
    "Dropout",
    "BiLSTM", "BiGru",
    "Linear"
]


class ModelWrapper(nn.Module):
    def __init__(self, input_dim: int, use_positional: bool = False):
        super(ModelWrapper, self).__init__()
        self._input_dim: int = input_dim
        self._output_dim: int = 0
        self._nn: nn.Module = nn.Module()
        self._use_positional: bool = use_positional

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def _forward(
            self,
            inp: torch.Tensor,
            inp_length: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        raise NotImplementedError()

    def forward(self,
                inp: torch.Tensor,
                inp_length: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self._forward(inp, inp_length), inp_length

    def init_weights(self):
        return


class Dropout(ModelWrapper):
    def __init__(
        self,
        input_dim,
        rate: float
    ):
        super(Dropout, self).__init__(input_dim=input_dim)
        self._rate = rate
        self._output_dim = input_dim
        self._nn = nn.Dropout(self._rate)

    def _forward(self, inp: torch.Tensor, inp_length: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self._nn(inp)


class Linear(ModelWrapper):
    def __init__(
        self,
        input_dim,
        output_dim: int
    ):
        super(Linear, self).__init__(input_dim=input_dim, use_positional=False)
        self._nn = nn.Linear(in_features=input_dim, out_features=output_dim)
        # Directions*Hidden_Dim
        self._output_dim = output_dim

    def _forward(self, inp: torch.Tensor, inp_length: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self._nn(inp)


class Conv(ModelWrapper):
    """
    LICENSE: https://github.com/allenai/allennlp/blob/master/LICENSE
    """
    def __init__(
            self,
            input_dim,
            out_filters: int,
            filter_size: int,
            padding_size: Optional[int] = 0,
            activation: Optional[str] = "g"
    ):
        super(Conv, self).__init__(input_dim=input_dim, use_positional=False)
        self._padding_size = padding_size
        self._nn = nn.Conv1d(
            in_channels=input_dim,
            out_channels=out_filters if activation != "g" else 2*out_filters,
            kernel_size=(filter_size, ),
            padding=padding_size
        )
        self._output_dim = out_filters

        if activation == "g":
            self.activation: Callable[[torch.Tensor], torch.Tensor] = lambda conv_output: F.glu(conv_output, dim=1)

    def _forward(self, inp: torch.Tensor, inp_length: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Input is (batch_size, sequence_length, dimension)
        #   The convolution layers expect input of shape `(batch_size, in_channels, sequence_length)
        #   We permute last two dimensions

        # conv_input = [batch size, encoding dim, src sent len]
        conv_input = inp.permute(0, 2, 1)

        # If we have glu, out_dim is 2*out_filters then out_filters at activation time
        # Otherwise, always out_filters
        # conved = [batch size, 2*out_filters, src sent len]
        conved = self._nn(conv_input)
        return self.activation(conved).permute(0, 2, 1)


class PosEmbedding(ModelWrapper):
    """
    LICENSE: https://github.com/allenai/allennlp/blob/master/LICENSE
    """
    def __init__(
            self,
            input_dim,
            maximum_sentence_size: int,
            padding_size: Optional[int] = 0,
            activation: Optional[str] = None
    ):
        super(PosEmbedding, self).__init__(input_dim=input_dim, use_positional=False)
        self._padding_size = padding_size
        self._output_dim = self._input_dim

        self._nn: nn.Embedding = nn.Embedding(maximum_sentence_size, input_dim)
        self.activation: Optional[nn.Linear] = None
        if activation:
            self.activation = nn.Linear(input_dim, input_dim)

    def _forward(self, inp: torch.Tensor, inp_length: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Input is (batch_size, sequence_length, dimension)
        #   The convolution layers expect input of shape `(batch_size, in_channels, sequence_length)
        #   We permute last two dimensions

        pos = torch.arange(0, inp.shape[1], device=inp.device).unsqueeze(0).repeat(inp.shape[0], 1)

        inp = inp + self._nn(pos)
        if self.activation is not None:
            return self.activation(inp)
        return inp


class BiLSTM(ModelWrapper):
    def __init__(
            self,
            input_dim,
            hidden_dim: int,
            padding_idx = 0,
            layers: int = 1
    ):
        super(BiLSTM, self).__init__(input_dim=input_dim, use_positional=False)
        self._nn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=layers,
            bidirectional=True,
            batch_first=True
        )
        # Directions*Hidden_Dim
        self._output_dim = 2*hidden_dim
        self._padding_idx: int = padding_idx

    def _forward(self, inp: torch.Tensor, inp_length: Optional[torch.Tensor] = None) -> torch.Tensor:

        # inp = [src sent len, batch size, emb dim]
        # packed_outputs = [src sent len, batch size, hid dim * n directions]
        inp = rnn_utils.pack_padded_sequence(inp, lengths=inp_length.cpu(), batch_first=True)
        output, _ = self._nn(inp)
        return rnn_utils.pad_packed_sequence(
            output,
            padding_value=self._padding_idx,
            batch_first=True
        )[0]

    def init_weights(self):
        for name, param in self.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)


class BiGru(ModelWrapper):
    def __init__(
        self,
        input_dim,
        hidden_dim: int,
        layers: int = 1,
        padding_idx = 0
    ):
        super(BiGru, self).__init__(input_dim=input_dim, use_positional=False)
        self._nn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=layers,
            bidirectional=True,
            batch_first=True
        )
        # Directions*Hidden_Dim
        self._output_dim = 2 * hidden_dim
        self._padding_idx: int = padding_idx

    def _forward(self, inp: torch.Tensor, inp_length: Optional[torch.Tensor] = None) -> torch.Tensor:

        # inp = [src sent len, batch size, emb dim]
        # packed_outputs = [src sent len, batch size, hid dim * n directions]

        inp = rnn_utils.pack_padded_sequence(inp, lengths=inp_length.cpu(), batch_first=True)
        output, _ = self._nn(inp)
        return rnn_utils.pad_packed_sequence(
            output,
            padding_value=self._padding_idx,
            batch_first=True
        )[0]

    def init_weights(self):
        for name, param in self.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)


class SequentialConv(ModelWrapper):
    # https://github.com/bentrevett/pytorch-seq2seq/blob/master/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb
    def __init__(
        self,
        input_dim,
        filter_dim,
        n_layers,
        filter_size,
        use_sum: Optional[str] = None,
        dropout: Optional[float] = None
    ):
        super().__init__(input_dim=input_dim)

        assert filter_size % 2 == 1, "Filter size must be odd!"

        self._output_dim = input_dim
        self._filter_size = filter_size

        self._scale = torch.sqrt(torch.FloatTensor([0.5]))
        self._inp_to_filter = nn.Linear(input_dim, filter_dim)
        self._filter_to_inp = nn.Linear(filter_dim, input_dim)

        self._nns = nn.ModuleList([
            nn.Conv1d(in_channels=filter_dim, out_channels=2 * filter_dim,
                      kernel_size=filter_size, padding=(filter_size - 1) // 2)
            for _ in range(n_layers)
        ])

        self._dropout: Optional[nn.Dropout] = None
        if dropout:
            self._dropout = nn.Dropout(dropout)

        self._use_sum: Optional[str] = use_sum

    def dropout(self, x):
        if self._dropout is not None:
            return self._dropout(x)
        return x

    def _forward(self, inp: torch.Tensor, inp_length: Optional[torch.Tensor] = None) -> torch.Tensor:
        # pass embedded through linear layer to go through emb dim -> hid dim
        # conv_input = [batch size, src sent len, hid dim]
        conv_input = self._inp_to_filter(inp)

        # permute for convolutional layer
        conv_input = conv_input.permute(0, 2, 1)

        # conv_input = [batch size, hid dim, src sent len]
        self._scale = self._scale.type_as(conv_input)

        conved: Optional[torch.Tensor] = None

        for i, conv in enumerate(self._nns):
            # pass through convolutional layer
            conved = conv(self.dropout(conv_input))

            # conved = [batch size, 2*hid dim, src sent len]

            # pass through GLU activation function
            conved = F.glu(conved, dim=1)

            # conved = [batch size, hid dim, src sent len]

            # apply residual connection
            conved = (conved + conv_input) * self._scale

            # conved = [batch size, hid dim, src sent len]

            # set conv_input to conved for next lo`op iteration
            conv_input = conved

        # permute and convert back to emb dim
        conved = self._filter_to_inp(conved.permute(0, 2, 1))

        if self._use_sum:
            # conved = [batch size, src sent len, emb dim]

            # elementwise sum output (conved) and input (embedded) to be used for attention
            combined = (conved + inp) * self._scale

            # combined = [batch size, src sent len, emb dim]
            return combined
        return conved
