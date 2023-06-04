import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import math
from models.seq2seq import Seq2SeqModule
from datasets import MidiDataModule, FeedbackDataModule
from vocab import RemiVocab, DescriptionVocab
from constants import PAD_TOKEN, EOS_TOKEN, BAR_KEY, POSITION_KEY


import transformers
from transformers import (
  BertConfig,
  EncoderDecoderConfig,
  EncoderDecoderModel
)

"""
RewardModule
------------
This module is a reward model for RLHF using the baseline figaro (Music Transformer) model.
"""
class RewardModule(pl.LightningModule):
  def __init__(self,
               backbone_checkpoint,
               d_model=512,
               context_size=512,
               lr=1e-4,
               lr_schedule='sqrt_decay',
               warmup_steps=None,
               max_steps=None,
               encoder_layers=6,
               decoder_layers=12,
               intermediate_size=2048,
               num_attention_heads=8):
    super(RewardModule, self).__init__()

    self.description_flavor = "none"
    self.description_options = None

    self.context_size = context_size
    self.d_model = d_model

    self.lr = lr
    self.lr_schedule = lr_schedule
    self.warmup_steps = warmup_steps
    self.max_steps = max_steps

    self.vocab = RemiVocab()
    
    self.backbone = Seq2SeqModule.load_from_checkpoint(checkpoint_path=backbone_checkpoint)
    assert(self.backbone.d_model == self.d_model and self.backbone.context_size == self.context_size)

    self.out_layer = nn.Linear(self.d_model * self.context_size, 1)

    self.save_hyperparameters()

  def get_datamodule(self, feedback_data, midi_root_dir, **kwargs):
    return FeedbackDataModule(
      feedback_data, 
      midi_root_dir,
      self.context_size,
      **kwargs
    )

  def forward(self, x, bar_ids=None, position_ids=None, return_hidden=False):
    # TODO, eventually train all weights, only train last layer for testing
    self.backbone.eval()
    with torch.no_grad():
      hidden = self.backbone.decode(x, bar_ids=bar_ids, position_ids=position_ids, return_hidden=True)
      # (batch_size, context_size * d_hidden)
      hidden = hidden.reshape(hidden.shape[0], -1)
    logits = self.out_layer(hidden)
    return logits

  def get_loss(self, batch, return_logits=False):
    # Shape of x: (batch_size, seq_len, tuple_size)
    x_0 = batch['input_ids_0']
    bar_ids_0 = batch['bar_ids_0']
    position_ids_0 = batch['position_ids_0']
    x_1 = batch['input_ids_1']
    bar_ids_1 = batch['bar_ids_1']
    position_ids_1 = batch['position_ids_1']
    preference = batch['preference']  

    rewards_0 = self(x_0, bar_ids=bar_ids_0, position_ids=position_ids_0)
    rewards_1 = self(x_1, bar_ids=bar_ids_1, position_ids=position_ids_1)

    logits = rewards_1 - rewards_0

    loss = F.binary_cross_entropy_with_logits(logits, torch.unsqueeze(torch.Tensor(preference).to(self.device), 1), reduction='mean')

    # logits = self(x, z=z, labels=labels, bar_ids=bar_ids, position_ids=position_ids, description_bar_ids=desc_bar_ids)
    # # Shape of logits: (batch_size, tgt_len, tuple_size, vocab_size)
    # pred = logits.view(-1, logits.shape[-1])
    # # Shape of pred: (batch_size * tgt_len * tuple_size, vocab_size)
    # labels = labels.reshape(-1)
    
    # loss = self.loss_fn(pred, labels)

    if return_logits:
      return loss, logits
    else:
      return loss
    
  def training_step(self, batch, batch_idx):
    loss = self.get_loss(batch)
    self.log('train_loss', loss.detach(), on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
    return loss
  
  def validation_step(self, batch, batch_idx):
    loss, logits = self.get_loss(batch, return_logits=True)
    self.log('valid_loss', loss.detach(), on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
    return loss
  
  def test_step(self, batch, batch_idx):
    return self.get_loss(batch)
        
  def configure_optimizers(self):
    # set LR to 1, scale with LambdaLR scheduler
    optimizer = transformers.AdamW(self.parameters(), lr=1, weight_decay=0.01)

    if self.lr_schedule == 'sqrt_decay':
      # constant warmup, then 1/sqrt(n) decay starting from the initial LR
      lr_func = lambda step: min(self.lr, self.lr / math.sqrt(max(step, 1)/self.warmup_steps))
    elif self.lr_schedule == 'linear':
      # linear warmup, linear decay
      lr_func = lambda step: min(self.lr, self.lr*step/self.warmup_steps, self.lr*(1 - (step - self.warmup_steps)/self.max_steps))
    elif self.lr_schedule == 'cosine':
      # linear warmup, cosine decay to 10% of initial LR
      lr_func = lambda step: self.lr * min(step/self.warmup_steps, 0.55 + 0.45*math.cos(math.pi*(min(step, self.max_steps) - self.warmup_steps)/(self.max_steps - self.warmup_steps)))
    else:
      # Use no lr scheduling
      lr_func = lambda step: self.lr
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    return [optimizer], [{
      'scheduler': scheduler,
      'interval': 'step',
    }]




    





