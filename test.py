from boudams.tagger import Seq2SeqTokenizer
from boudams.trainer import Trainer

import glob
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


TEST = True
BATCH_SIZE = 32
DEVICE = "cuda"


if TEST is True:
    train_path, dev_path, test_path = "data/small/train.tsv", "data/small/dev.tsv", "data/small/test.tsv"
else:
    train_path, dev_path, test_path = "data/fro/train.tsv", "data/fro/dev.tsv", "data/fro/test.tsv"


for model in glob.glob("/home/thibault/dev/boudams/models/conv2019-05-21--07:29:22.tar"):
    tokenizer = Seq2SeqTokenizer.load(model, device=DEVICE)
    print("Model : " + tokenizer.system.upper() + " from  " + model)
    test_data = tokenizer.vocabulary.get_dataset(test_path, random=False)
    trainer = Trainer(tokenizer, device=DEVICE)
    trainer.test(test_data, batch_size=BATCH_SIZE)
