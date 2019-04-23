from boudams.tagger import Seq2SeqTokenizer
from boudams.dataset import SOS_TOKEN
import math

vocabulary, train_dataset, dev_dataset, test_dataset = Seq2SeqTokenizer.get_dataset_and_vocabularies(
    "data/train/train.tab", "data/dev/dev.tab", "data/test/test.tab"
)


tagger = Seq2SeqTokenizer(vocabulary, n_layers=4, hidden_size=512, device="cuda")
tagger.train(train_dataset, dev_dataset, n_epochs=50)

#    src = batch.src l.198 evaluate
# AttributeError: 'BucketIterator' object has no attribute 'src'
# test_loss = tagger.test(test_dataset)
# print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
