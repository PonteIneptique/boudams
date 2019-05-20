import logging
import datetime

from boudams.tagger import Seq2SeqTokenizer
from boudams.trainer import Trainer
from boudams.encoder import LabelEncoder, DatasetIterator

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)




TEST = False
RANDOM = True
DEVICE = "cuda"
MAXIMUM_LENGTH = 150

if TEST is True:
    train_path, dev_path, test_path = "data/small/train.tsv", "data/small/dev.tsv", "data/small/test.tsv"
else:
    train_path, dev_path, test_path = "data/fro/train.tsv", "data/fro/dev.tsv", "data/fro/test.tsv"

vocabulary = LabelEncoder(maximum_length=MAXIMUM_LENGTH)
vocabulary.build(train_path, dev_path, test_path, debug=True)

# Get the datasets
train_dataset: DatasetIterator = vocabulary.get_dataset(train_path, random=False)
dev_dataset: DatasetIterator = vocabulary.get_dataset(dev_path, random=False)
test_dataset: DatasetIterator = vocabulary.get_dataset(test_path, random=False)

from pprint import pprint
#pprint(vocabulary.vocab.freqs)
print("-- Dataset informations --")
print("Number of training examples: {}".format(len(train_dataset)))
print("Number of dev examples: {}".format(len(dev_dataset)))
print("Number of testing examples: {}".format(len(test_dataset)))
print("--------------------------")


def examples(obj):
    example_sentences = [
        ('vosvenitesdevantmoiqantgevosdisquevosenaillissiezousece', 'vos venites devant moi qant ge vos dis que vos en aillissiez ou se ce'),
        ('nonlicuersmepartiroitelventrecarjaienvostotemisel', 'non li cuers me partiroit el ventre car j ai en vos tote mise l'),
    ]
    treated = obj.annotate([x[0] for x in example_sentences])

    for (inp, exp), out in zip(example_sentences, treated):
        logger.debug("----")
        logger.debug("Inp " + inp)
        logger.debug("Exp " + exp)
        logger.debug("Out " + out)
        logger.debug("----")


for settings, system, batch_size, train_dict in [
    (dict(hidden_size=512, emb_enc_dim=256, emb_dec_dim=256,
          enc_n_layers=6, dec_n_layers=6,
          enc_dropout=0.25, dec_dropout=0.25), "conv", 128,
        dict(lr_grace_periode=2, lr_patience=2, lr=0.0005)),
    (dict(hidden_size=128, emb_enc_dim=128, emb_dec_dim=128), "gru", 256,
        dict(lr_grace_periode=2, lr_patience=2)),
    (dict(hidden_size=256, emb_enc_dim=256, emb_dec_dim=256, enc_n_layers=2, dec_n_layers=2), "lstm", 256,
        dict(lr_grace_periode=2, lr_patience=2, lr=0.01)),
    (dict(hidden_size=256, emb_enc_dim=128, emb_dec_dim=128), "bi-gru", 32,
         dict(lr_grace_periode=2, lr_patience=2, lr=0.01))
]:
    device = DEVICE
    tagger = Seq2SeqTokenizer(vocabulary, device=device, system=system, out_max_sentence_length=MAXIMUM_LENGTH
                              , **settings)
    trainer = Trainer(tagger, device=device)
    print(tagger.model)
    print()
    trainer.run(
        train_dataset, dev_dataset, n_epochs=10,
        fpath="models/"+system+str(datetime.datetime.today()).replace(" ", "--").split(".")[0]+".tar",
        batch_size=batch_size,
        debug=examples,
        **train_dict
    )

#    src = batch.src l.198 evaluate
# AttributeError: 'BucketIterator' object has no attribute 'src'
# test_loss = tagger.test(test_dataset)
# print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
