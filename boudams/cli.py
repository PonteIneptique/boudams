import click
import re
import tqdm


from boudams.tagger import Seq2SeqTokenizer


@click.group()
def cli():
    """ Boudams is a tokenizer built on deep learning. """


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
