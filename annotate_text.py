input_text = """pouture.⁊laprouanostresiresqeparla
desertedelaglorieuseuirgeqifumar
tẏrieporsamour.deluirailceusdelpais
delgrantfeu.⁊delgranttorment.⁊ali
soithoneurs⁊gloireqisiaide⁊deliure
ceusquienluisefient.⁊qileseruentes
"""

print(input_text)

from boudams.tagger import Seq2SeqTokenizer
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

tokenizer = Seq2SeqTokenizer.load("/home/thibault/dev/boudams/models/linear-conv2019-05-24--14:08:58-0.0001.tar", device="cpu")
print(" ".join(tokenizer.annotate_text(input_text.replace(" ", ""))))