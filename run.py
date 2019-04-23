from boudams.tagger import Seq2SeqTokenizer
from boudams.dataset import SOS_TOKEN
import math

vocabulary, train_dataset, dev_dataset, test_dataset = Seq2SeqTokenizer.get_dataset_and_vocabularies(
    "data/train/train.tab", "data/dev/dev.tab", "data/test/test.tab"
)


for system in ["bi-gru"]:  # "gru",
    tagger = Seq2SeqTokenizer(vocabulary, n_layers=2,
                              hidden_size=256, emb_enc_dim=128, emb_dec_dim=128,
                              device="cuda", system=system)
    tagger.train(train_dataset, dev_dataset, n_epochs=100, fpath="models/"+system+".tar", batch_size=128)

#    src = batch.src l.198 evaluate
# AttributeError: 'BucketIterator' object has no attribute 'src'
# test_loss = tagger.test(test_dataset)
# print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
