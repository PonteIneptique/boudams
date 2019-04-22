from boudams.dataset import Dataset
from boudams.tagger import Seq2SeqTokenizer
import glob


tagger = Seq2SeqTokenizer.from_data(glob.glob("data/**/*.tab"))  # Loading the global dictionary
tagger.train(Dataset("data/train/train.tab"))