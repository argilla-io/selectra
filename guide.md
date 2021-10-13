# Guide to pre-train SELECTRA with gcloud

In the following we provide detailed steps to reproduce our models with gcloud.

## 0. Gcloud stuff

- in the gcloud console, create a project called 'selectra'
- download the gcloud sdk (https://cloud.google.com/sdk/docs/quickstart#linux) to your machine
- make a `gcloud init` with our google cloud account, choose the 'selectra' project and the europe-west4-a zone
- most of the commands below are best executed within a screen session, check out this link for a little introduction of screen: https://www.howtogeek.com/662422/how-to-use-linuxs-screen-command/

## 1. Download corpus

- Create a basic VM on google cloud with an Ubuntu 20 image, call it 'selectra-1'
- You have to give this instance full access to all Cloud APIs (VM instance details -> Cloud API access scopes -> Allow full access to all Cloud APIs), otherwise the training scripts cannot access the storage bucket and the tpu
- Add a 1TB (standard storage) disk to the selectra-1 instance (see https://cloud.google.com/compute/docs/disks/add-persistent-disk)
- Now you can connect to the VM from your machine via `gcloud compute ssh recognai@selectra-1`
- On the VM, first make a `gcloud init` with your gcloud account, choose the selectra project and the europe-west4-a zone
- mount the external 1TB disk under '/home/recognai/disk' (`sudo mount /dev/<HD-id> /home/recognai/disk`)
- install git (`sudo apt update; sudo apt install git`)
- install miniconda (`wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh; bash Miniconda3-latest-Linux-x86_64.sh`)
- clone selectra repo (`git clone https://github.com/recognai/selectra.git`) 
- create and activate conda env (`cd selectra; conda env create; conda activate selectra`)
- set env var for HF cache (`export HF_HOME=/home/recognai/disk/huggingface`), put in the bashrc. This is important since the oscar data set is huge and only fits on the external disk.
- execute the download script in a screen session (`python download_oscar.py`), this will take some time (hours).

## 2. Create vocab/tfrecords

- Separate files of our corpus are needed for the vocab creation and the creation of the tf records. If you want to use directly the dataset, make sure you do not shuffle it, since this makes the vocab creation process super slow, but we need the files anyhow later on for the tfrecords creation process.
- execute the `create_vocab.py` script (`python create_vocab.py`). Make sure you execute it on a machine with at least 32GB RAM and 8 CPUs, upgrade your VM if necessary.
- execute the `create_tfrecords.sh` script (`bash create_tfrecords.sh`, use a machine with at least 8 CPUs)
- create a storage bucket (see https://cloud.google.com/storage/docs/creating-buckets), if you already know the region of your TPU you can set it here for the lowest latency
- copy the vocab and the tfrecords to the bucket (`gsutil cp ~/disk/vocab_50000_0.5/vocab.txt gs://selectra-1/vocab_50000_0.5/vocab.txt`, `gsutil -m cp -r ~/disk/tfrecords gs://selectra-1/tfrecords`)

## 3. Training

- create a TPU node in the Google console 
- Going back to your storage bucket in the Google console, you have to provide permissions to this TPU node to save the checkpoints/model (Permissions -> Permissions Add -> Add email address of the TPU service, something like service-446706891675@cloud-tpu.iam.gserviceaccount.com -> Give it the role: Cloud Storage / Storage Admin)
- execute the pre-training script (`bash run_pretraining_small.sh`)
- optional, start tensorboard: `tensorboard --logdir gs://selectra-1/models/selectra_small`; now you can do a port forwarding in your machine using gcloud: open a new terminal and type in `gcloud compute ssh recognai@selectra-1 -- -NL 6006:localhost:6006`. Now you can go to a browser and open the webpage localhost:6006

## 4. Fine-tuning

- simply run the fine-tuning script (`bash run_finetuning_small.sh`)
- results are saved in `gs://selectra-1/models/selectra_small/results`

