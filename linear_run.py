import logging
import datetime

from boudams.tagger import Seq2SeqTokenizer
from boudams.trainer import Trainer
from boudams.encoder import LabelEncoder, DatasetIterator

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

EPOCHS = 100
TEST = "fro"
RANDOM = True
DEVICE = "cuda"
MAXIMUM_LENGTH = 100
LOAD_VOCABULARY = False
LRS = (0.0001, )
# Masked should not work given the fact that out_token_embedding is gonna be screwed
MASKED = True

if TEST is "seints":
    train_path, dev_path, test_path = "data/seints/train.tsv", "data/seints/dev.tsv", "data/seints/test.tsv"
elif TEST is True:
    train_path, dev_path, test_path = "data/small/train.tsv", "data/small/dev.tsv", "data/small/test.tsv"
else:
    train_path, dev_path, test_path = "data/fro/train.tsv", "data/fro/dev.tsv", "data/fro/test.tsv"

if LOAD_VOCABULARY:
    import json
    with open("voc-2.json") as f:
        vocabulary = LabelEncoder.load(json.load(f))
else:
    vocabulary = LabelEncoder(maximum_length=MAXIMUM_LENGTH, masked=MASKED)
    vocabulary.build(train_path, dev_path, test_path, debug=True)
    print(vocabulary.dump())
    with open("voc-2.json", "w") as f:
        f.write(vocabulary.dump())

logging.info(vocabulary.stoi)

# Get the datasets
train_dataset: DatasetIterator = vocabulary.get_dataset(train_path, randomized=RANDOM)
dev_dataset: DatasetIterator = vocabulary.get_dataset(dev_path, randomized=RANDOM)
test_dataset: DatasetIterator = vocabulary.get_dataset(test_path, randomized=RANDOM)


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
        logger.info("----")
        logger.info("Inp " + inp)
        logger.info("Exp " + exp)
        logger.info("Out " + out)
        logger.info("----")


linear = (
    dict(
        hidden_size=512, emb_enc_dim=256, emb_dec_dim=256,
        enc_n_layers=10, dec_n_layers=10,
        enc_dropout=0.25, dec_dropout=0.25),
    "linear-conv", 32,
    dict(lr_grace_periode=2, lr_patience=2, lr=0.0001)
)


for settings, system, batch_size, train_dict in [
    linear
]:
    for lr in LRS:
        device = DEVICE
        tagger = Seq2SeqTokenizer(vocabulary, device=device, system=system, out_max_sentence_length=MAXIMUM_LENGTH
                                  , **settings)
        trainer = Trainer(tagger, device=device)
        print(tagger.model)
        print()
        train_dict["lr"] = lr
        trainer.run(
            train_dataset, dev_dataset, n_epochs=EPOCHS,
            fpath="models/"+system+str(datetime.datetime.today()).replace(" ", "--").split(".")[0]+"-"+str(lr)+".tar",
            batch_size=batch_size,
            debug=examples,
            **train_dict
        )

#    src = batch.src l.198 evaluate
# AttributeError: 'BucketIterator' object has no attribute 'src'
# test_loss = tagger.test(test_dataset)
# print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
