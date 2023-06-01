

import torch

import os
import glob

import pytorch_lightning as pl

from models.seq2seq import Seq2SeqModule
from models.reward import RewardModule
from models.vae import VqVaeModule

import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ROOT_DIR = os.getenv('ROOT_DIR', './')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', './results')
LOGGING_DIR = os.getenv('LOGGING_DIR', './logs')
MAX_N_FILES = int(os.getenv('MAX_N_FILES', -1))

D_MODEL = int(os.getenv('D_MODEL', 512))
D_LATENT = int(os.getenv('D_LATENT', 1024))

CHECKPOINT = os.getenv('CHECKPOINT', None)
VAE_CHECKPOINT = os.getenv('VAE_CHECKPOINT', None)

BATCH_SIZE = int(os.getenv('BATCH_SIZE', 128))
TARGET_BATCH_SIZE = int(os.getenv('TARGET_BATCH_SIZE', 512))

EPOCHS = int(os.getenv('EPOCHS', '16'))
WARMUP_STEPS = int(float(os.getenv('WARMUP_STEPS', 4000)))
MAX_STEPS = int(float(os.getenv('MAX_STEPS', 1e20)))
MAX_TRAINING_STEPS = int(float(os.getenv('MAX_TRAINING_STEPS', 100_000)))
LEARNING_RATE = float(os.getenv('LEARNING_RATE', 1e-4))
LR_SCHEDULE = os.getenv('LR_SCHEDULE', 'const')
CONTEXT_SIZE = int(os.getenv('CONTEXT_SIZE', 256))

ACCUMULATE_GRADS = max(1, TARGET_BATCH_SIZE//BATCH_SIZE)

N_WORKERS = min(os.cpu_count(), float(os.getenv('N_WORKERS', 'inf')))
if device.type == 'cuda':
  N_WORKERS = min(N_WORKERS, 8*torch.cuda.device_count())
N_WORKERS = int(N_WORKERS)


def main():
  MAX_CONTEXT = min(1024, CONTEXT_SIZE)
  vae_module = None

  ### Load data ###

  csv_files = glob.glob(os.path.join('./musicgpt-user-study/*.csv'), recursive=True)

  dfs = []
  for csv in csv_files:
    dfs.append(pd.read_csv(csv))
  
  df = pd.concat(dfs, axis=0, ignore_index=True)
  # Make sure we're looking at the processed MIDI clips
  df = df.applymap(lambda value: str(value).replace('prompt', 'prompt-processed'))
  # Shuffle rows
  df.sample(frac=1, random_state=0)
  feedback_data = df.to_numpy()

  ### Create and train model ###

  # load model from checkpoint if available
  if CHECKPOINT is None:
    print("No checkpoint given for baseline figaro model.")
    exit()

  else:
    seq2seq_kwargs = {
      'encoder_layers': 4,
      'decoder_layers': 6,
      'num_attention_heads': 8,
      'intermediate_size': 2048,
      'd_model': D_MODEL,
      'context_size': MAX_CONTEXT,
      'lr': LEARNING_RATE,
      'warmup_steps': WARMUP_STEPS,
      'max_steps': MAX_STEPS,
    }
    dec_kwargs = { **seq2seq_kwargs }
    dec_kwargs['encoder_layers'] = 0

    model = RewardModule(backbone_checkpoint=CHECKPOINT,
                         d_model=D_MODEL,
                         context_size=MAX_CONTEXT,
                         lr=LEARNING_RATE,
                         warmup_steps=WARMUP_STEPS,
                         max_steps=MAX_STEPS)

  datamodule = model.get_datamodule(
    feedback_data=feedback_data,
    midi_root_dir=ROOT_DIR,
    vae_module=vae_module,
    batch_size=BATCH_SIZE, 
    num_workers=N_WORKERS, 
    pin_memory=True
  )

  # Callback used to retrieve a checkpoint during training
  checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
    monitor='valid_loss',
    dirpath=os.path.join(OUTPUT_DIR, 'reward'),
    filename='{step}-{valid_loss:.2f}',
    save_last=True,
    save_top_k=2,
    every_n_train_steps=1000,
  )

  lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')

  # trainer = pl.Trainer(
  #   gpus=0 if device.type == 'cpu' else torch.cuda.device_count(),
  #   accelerator='dp',
  #   profiler='simple',
  #   callbacks=[checkpoint_callback, lr_monitor],
  #   max_epochs=EPOCHS,
  #   max_steps=MAX_TRAINING_STEPS,
  #   log_every_n_steps=max(100, min(25*ACCUMULATE_GRADS, 200)),
  #   val_check_interval=max(500, min(300*ACCUMULATE_GRADS, 1000)),
  #   limit_val_batches=64,
  #   auto_scale_batch_size=False,
  #   auto_lr_find=False,
  #   accumulate_grad_batches=ACCUMULATE_GRADS,
  #   stochastic_weight_avg=True,
  #   gradient_clip_val=1.0, 
  #   terminate_on_nan=True
  # )

  # Debugging trainer
  trainer = pl.Trainer(
    gpus=0 if device.type == 'cpu' else torch.cuda.device_count(),
    limit_train_batches=1.0
  )

  trainer.fit(model, datamodule)

if __name__ == '__main__':
  main()