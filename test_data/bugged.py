from boudams.encoder import LabelEncoder
import logging


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

label = LabelEncoder(masked=True)
label.build("bugged.txt", debug=True)

dataset = label.get_dataset("bugged.txt")
batch_generator = dataset.get_epoch(batch_size=32)()
batch = next(batch_generator)
print(batch)