from tokenizers import BertWordPieceTokenizer
import random
from pathlib import Path

# Path of the files
PATH = "/home/recognai/disk/oscar_files"
# Size of the final vocab
VOCAB_SIZE = 50000
# Fraction of the corpus used for the vocab creation
CORPUS_FRAC = 0.5
# Output folder
OUTPUT = f"/home/recognai/disk/vocab_{VOCAB_SIZE}_{CORPUS_FRAC}"


path = Path(PATH)

# create output folder
output_folder = Path(OUTPUT)
output_folder.mkdir()
        
# Select files for training
files = list(map(str, path.glob("file*.txt")))
files = files[:int(len(files)*CORPUS_FRAC)]

# save list of files
with (output_folder / "file_list.txt").open("w") as f:
    f.write("\n".join(files))

# train tokenizer
tokenizer = BertWordPieceTokenizer(lowercase=False, strip_accents=False)
tokenizer.train(
    files, 
    vocab_size=VOCAB_SIZE, 
    min_frequency=2, 
    limit_alphabet=1000
)

# save vocab.txt
tokenizer.save_model(str(output_folder))


# # check fraction of continuation tokens
# j = 0
# for i in tokenizer.get_vocab():
#     if i.startswith("##"):
#         j += 1
# j/tokenizer.get_vocab_size()
