import torch
import torch.cuda
import torch.nn as nn
import torch.optim as optim

import random
import time


from .utils import timeSince, showPlot
from .indexes import SOS_token, EOS_token, Dictionary
from .model import EncoderRNN, AttnDecoderRNN
from .dataset import Dataset


#
teacher_forcing_ratio = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 150


class Seq2SeqTokenizer:
    def __init__(self, hidden_size: int = 256, max_length=MAX_LENGTH, device: str=DEVICE):
        self.char_dict: Dictionary = Dictionary("characters")
        self.word_dict: Dictionary = Dictionary("words")
        self.max_length: int = max_length
        self.encoder: EncoderRNN = None
        self.decoder: AttnDecoderRNN = None
        self.device = DEVICE
        self.hidden_size: int = hidden_size

    @classmethod
    def from_data(cls,
                  data_paths, hidden_size: int = 256, dropout: float = 0.1, device: str = DEVICE) -> "Seq2SeqTokenizer":
        """

        :param data_paths:
        :param hidden_size:
        :param dropout:
        :param device:
        :return:
        """
        obj = cls(hidden_size)
        obj.char_dict, obj.word_dict, obj.max_length = Dictionary.load_dataset(data_paths)
        obj.max_length += 1  # EOS

        print("[DATASET] Using max length %s " % obj.max_length)
        print("[DATASET] Input characters %s " % obj.char_dict.n_words)
        print("[DATASET] Ouput characters %s " % obj.word_dict.n_words)

        obj.encoder = EncoderRNN(obj.char_dict.n_words, obj.hidden_size).to(device)
        obj.decoder = AttnDecoderRNN(
            obj.hidden_size, obj.word_dict.n_words,
            dropout_p=dropout, max_length=obj.max_length
        ).to(device)

        return obj

    def _train_sentence(self, input_tensor, target_tensor, encoder_optimizer, decoder_optimizer, criterion):
        encoder_hidden = self.encoder.initHidden()

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_size, device=self.device)

        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(
                input_tensor[ei], encoder_hidden
            )
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=self.device)

        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                loss += criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss += criterion(decoder_output, target_tensor[di])
                if decoder_input.item() == EOS_token:
                    break

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        return loss.item() / target_length

    def train(self,
              dataset: Dataset, n_epochs: int = 10, print_every=100, plot_every=100, learning_rate=0.01, plot=True):
        """

        :param dataset: The dataset that will be itered over
        :param n_epochs: Number of time the dataset should be seen
        :param print_every: Number of sentence that should be seen before printing loss
        :param plot_every: Number of sentence that should be seen before ploting loss
        :param learning_rate: Learning rate to apply
        :param plot: Whether to plot or not
        :return:
        """
        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.SGD(self.decoder.parameters(), lr=learning_rate)
        criterion = nn.NLLLoss()

        for epoch in range(1, n_epochs + 1):
            print("[Epoch %s] %s Sentences" % (epoch, dataset.iterable))

            for index_sentence, sentence in enumerate(dataset):
                index_sentence += 1  # Avoid dividing by 0
                input_tensor, target_tensor = self.tensorsFromPair(sentence)

                loss = self._train_sentence(
                    input_tensor, target_tensor,
                    encoder_optimizer, decoder_optimizer,
                    criterion
                )
                print_loss_total += loss
                plot_loss_total += loss

                if index_sentence % print_every == 0:
                    print_loss_avg = print_loss_total / print_every
                    print_loss_total = 0
                    print('%s (%d %d%%) %.4f' % (
                        timeSince(start, index_sentence / dataset.iterable),
                        index_sentence, index_sentence / dataset.iterable * 100, print_loss_avg)
                    )
                    out_chars, out_tensor = self.evaluate(sentence[0])
                    print("> " + "".join(out_chars))
                    print("= " + "".join(sentence[0]))

                if index_sentence % plot_every == 0:
                    plot_loss_avg = plot_loss_total / plot_every
                    plot_losses.append(plot_loss_avg)
                    plot_loss_total = 0

        if plot:
            showPlot(plot_losses)

    def tensorsFromPair(self, pair):
        input_tensor = self.char_dict.tensorFromSentence(pair[0], device=self.device)
        target_tensor = self.word_dict.tensorFromSentence(pair[1], device=self.device)
        return input_tensor, target_tensor

    def evaluate(self, sentence):
        with torch.no_grad():
            input_tensor = self.char_dict.tensorFromSentence(sentence, device=self.device)
            input_length = input_tensor.size()[0]
            encoder_hidden = self.encoder.initHidden()

            encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_size, device=self.device)

            for ei in range(input_length):
                encoder_output, encoder_hidden = self.encoder(input_tensor[ei],
                                                         encoder_hidden)
                encoder_outputs[ei] += encoder_output[0, 0]

            decoder_input = torch.tensor([[SOS_token]], device=self.device)  # SOS

            decoder_hidden = encoder_hidden

            decoded_words = []
            decoder_attentions = torch.zeros(self.max_length, self.max_length)

            for di in range(self.max_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_attentions[di] = decoder_attention.data
                topv, topi = decoder_output.data.topk(1)
                if topi.item() == EOS_token:
                    decoded_words.append('<EOS>')
                    break
                else:
                    decoded_words.append(self.word_dict.index2word[topi.item()])

                decoder_input = topi.squeeze().detach()

            return decoded_words, decoder_attentions[:di + 1]

    def evaluateRandomly(self, pairs, n=10):
        for i in range(n):
            pair = random.choice(pairs)
            print('>', pair[0])
            print('=', pair[1])
            output_words, attentions = self.evaluate(pair[0])
            output_sentence = ' '.join(output_words)
            print('<', output_sentence)
            print('')
