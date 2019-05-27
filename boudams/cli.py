import click
import re
import tqdm
import json


from boudams.tagger import Seq2SeqTokenizer


@click.group()
def cli():
    """ Boudams is a tokenizer built on deep learning. """


@cli.command("template")
@click.argument("filename", type=click.File(mode="w"))
def template(filename):
    """ Generates a template training model in file [FILENAME]"""
    template = {
        "max_sentence_size": 100,
        "encoder": {  # Configuration of the encoder
          'emb_enc_dim': 256,
          'enc_n_layers': 10,
          'enc_kernel_size': 3,
          'enc_dropout': 0.25
        },
        "model": 'linear-conv',
        "batch_size": 32,
        "learner": {
           'lr_grace_periode': 2,
           'lr_patience': 2,
           'lr': 0.0001
        },
        "datasets": {
            "test": "./test.tsv",
            "train": "./train.tsv",
            "dev": "./dev.tsv"
        }
    }
    json.dump(template, filename, indent=4, separators=(',', ': '))


@cli.command("train")
def train():
    """ Train a model """
    # ToDo: Have JSON configs here read into the train module


@cli.command("tag")
@click.argument("model", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("filename", nargs=-1, type=click.File("r"))
#@click.option("device", default="cpu", help="Device to use for the network (cuda, cpu, etc.)")
def tag(model, filename, device="cpu", batch_size=64):
    """ Tag all [FILENAME] using [MODEL]"""
    print("Loading the model.")
    model = Seq2SeqTokenizer.load(model, device=device)
    print("Model loaded.")
    remove_line = True
    spaces = re.compile("\s+")
    for file in tqdm.tqdm(filename):
        out_name = file.name.replace(".txt", ".tokenized.txt")
        content = file.read()  # Could definitely be done a better way...
        if remove_line:
            content = spaces.sub("", content)
        with open(out_name, "w") as out_io:
            for tokenized_string in model.annotate_text(content, batch_size=batch_size):
                out_io.write(tokenized_string+" ")
        # print("--- File " + file.name + " has been tokenized")


if __name__ == "__main__":
    cli()
