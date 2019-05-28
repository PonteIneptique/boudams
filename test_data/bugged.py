from boudams.encoder import LabelEncoder
import logging


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

label = LabelEncoder(masked=True, lower=False, remove_diacriticals=True)
label.build("bugged.txt", debug=True)

dataset = label.get_dataset("bugged.txt")
batch_generator = dataset.get_epoch(batch_size=64)()
batch = next(batch_generator)
print(batch)