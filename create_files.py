from datasets import load_dataset
from pathlib import Path

# samples per file, 100 000 results in ~300 MB files
SAMPLES_PER_FILE = 100000

# create folder
path = Path("/home/recognai/disk/oscar_files")
path.mkdir()

ds = load_dataset("oscar", "unshuffled_deduplicated_es", split="train")
# shuffling makes the process really slow ...
#ds = ds.shuffle(seed=43)

# define write to file function
batch_nr = 0
def write_file(batch):
    global batch_nr
    with (path / f"file_{batch_nr}.txt").open("w") as file:
        file.write("\n\n".join(batch["text"]))
    batch_nr += 1
    
# process batch-wsie
ds.map(write_file, batched=True, batch_size=SAMPLES_PER_FILE)


