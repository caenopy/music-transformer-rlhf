import math
import copy
from collections import deque, namedtuple
from functools import partial
from tqdm import tqdm

# from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

import shutil
import os
import glob
import time
import torch
import random
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import transformers
from torch.utils.tensorboard import SummaryWriter

from models.seq2seq import Seq2SeqModule
from datasets import MidiDataset, SeqCollator
from utils import medley_iterator
from input_representation import remi2midi

from models.seq2seq import Seq2SeqModule
from models.reward import RewardModule

from rlhfutils.optimizer import get_optimizer

from accelerate import Accelerator

from einops import rearrange

# Changes to run it on GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL = os.getenv('MODEL', '')

ROOT_DIR = os.getenv('ROOT_DIR', './lmd_full')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', None)
LOGGING_DIR = os.getenv('LOGGING_DIR', './rlhf-logs')
# OUTPUT_DIR = os.getenv('OUTPUT_DIR', './samples')
MAX_N_FILES = int(float(os.getenv('MAX_N_FILES', -1)))
MAX_ITER = int(os.getenv('MAX_ITER', 4096))
MAX_BARS = int(os.getenv('MAX_BARS', 32))

# Always generate medleys (used as prompt for generation)
MAKE_MEDLEYS = os.getenv('MAKE_MEDLEYS', 'True') == 'True'
N_MEDLEY_PIECES = int(os.getenv('N_MEDLEY_PIECES', 2))
# Medley (prompt) length is 3 bars
N_MEDLEY_BARS = int(os.getenv('N_MEDLEY_BARS', 6))

CHECKPOINT = os.getenv('CHECKPOINT', None)
# Batch size set to 1 for one prompt (medley) at a time
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 1))
VERBOSE = int(os.getenv('VERBOSE', 2))

REWARD_CHECKPOINT = os.getenv('REWARD_CHECKPOINT', None)



# Actor-Critic with Music Transformer

PPOActionCriticReturn = namedtuple('PPOActionCriticReturn', [
  'actions',
  'action_logits',
  'values'
])

# Based on https://github.com/lucidrains/PaLM-rlhf-pytorch/tree/main
class ActorCritic(nn.Module):
  def __init__(
    self,
    figaro: Seq2SeqModule,
    critic_figaro: Seq2SeqModule = None,
    pooled_values = False
  ):
    super().__init__()
    self.actor_figaro = figaro

    self.critic_figaro = critic_figaro

    if not exists(self.critic_figaro):
        self.critic_figaro = copy.deepcopy(figaro)

    self.pooled_values = pooled_values

    self.value_head = nn.Sequential(
        nn.Linear(figaro.d_model, 1),
        Rearrange('... 1 -> ...')
    )

    nn.init.zeros_(self.value_head[0].bias)
    nn.init.orthogonal_(self.value_head[0].weight, gain = math.sqrt(2))


  def actor_parameters(self):
    return self.actor_figaro.parameters()
  
  def critic_parameters(self):
    return [*self.critic_figaro.parameters(), *self.value_head.parameters()]

  @torch.no_grad()
  def generate(self, 
                batch, 
                batch_gt=None,
                max_initial_context=1, 
                output_dir=None, 
                max_len=256,
                max_bars=-1,
                verbose=0):
  
    # Generate a REMI events sequence from the current policy
    batch_size, prompt_len = batch['input_ids'].shape[:2]

    batch_ = { key: batch[key][:, :max_initial_context] for key in ['input_ids', 'bar_ids', 'position_ids'] }
    
    total_max_len = prompt_len + max_len 
    if verbose:
      print(f"Generating sequence ({prompt_len} prompt tokens / {total_max_len} total max tokens / {max_bars} max bars / {batch_size} batch size)")
    # The following returns sample = {'sequences': x, 'bar_ids': bar_ids, 'position_ids': position_ids}
    actions = self.actor_figaro.sample(batch_, max_length=total_max_len, max_bars=max_bars, verbose=0)
    
    # Decode REMI events to MIDI
    # Run ground truth through FIGARO encoding, so vocabulary is restricted for fair comparison
    # xs = batch['input_ids'].detach().cpu()
    # xs_gt = batch_gt['input_ids'].detach().cpu()
    # xs_hat = sample['sequences'].detach().cpu()
    # prompt
    # events = [model.vocab.decode(x) for x in xs]
    # ground truth completion of prompt
    # events_gt = [model.vocab.decode(x) for x in xs_gt]
    # predicted completion of prompt
    # events_hat = [model.vocab.decode(x) for x in xs_hat]

    # state + action is sample['sequences']


    actions['sequences'] = actions['sequences'][:, -max_len:]
    actions['bar_ids'] = actions['bar_ids'][:, -max_len:]
    actions['position_ids'] = actions['position_ids'][:, -max_len:] 
    
    action_logits, value = self.forward(x=actions['sequences'],
                                        position_ids=actions['position_ids'], 
                                        bar_ids=actions['bar_ids'], 
                                        return_hidden=False,
                                        return_values=True
                                      )

    return PPOActionCriticReturn(
      actions,
      action_logits,
      value
     ) 

  def forward(
    self,
    x,
    position_ids=None, 
    bar_ids=None, 
    return_hidden=False,
    return_values = True
  ):
  
    # Figaro decoder is Music Transformer
    action_logits = self.actor_figaro.decode(
      x, 
      labels=None, 
      bar_ids=bar_ids, 
      position_ids=position_ids, 
      encoder_hidden_states=None,
      return_hidden=return_hidden
    )

    if not return_values:
        return action_logits, None

    critic_embeds = self.critic_figaro.decode(
      x, 
      labels=None, 
      bar_ids=bar_ids, 
      position_ids=position_ids, 
      encoder_hidden_states=None,
      return_hidden=True
    )

    # critic_embeds = critic_embeds.reshape(critic_embeds.shape[0], -1)
    critic_embeds = critic_embeds[:, -1, :]

    # TODO: add functionality for pooled values
    # if self.pooled_values:
    #   critic_embeds = shift(critic_embeds, shift = 1, dim = -2)
    #   critic_embeds = masked_mean(critic_embeds, mask, dim = 1)

    values = self.value_head(critic_embeds)

    return action_logits, values 


# PPO Data

Memory = namedtuple('Memory', [
    'sequences',
    'positon_ids',
    'bar_ids',
    'action_prob',
    'sequences_log_prob',
    'reward',
    'value'
])

class ExperienceDataset(Dataset):
  def __init__(
    self,
    data, # data: List[torch.Tensor],
    device = None
  ):
    super().__init__()
    self.data = data
    self.device = device

  def __len__(self):
    return self.data[0].shape[0]

  def __getitem__(self, ind):
    return tuple(map(lambda t: t[ind].to(self.device), self.data))

def create_dataloader(data, batch_size, shuffle = True, device = None, **kwargs):
  ds = ExperienceDataset(data, device = device)
  return DataLoader(ds, batch_size = batch_size, shuffle = shuffle, **kwargs)

# Helper functions

def exists(val):
  return val is not None

def default(val, d):
  if exists(val):
    return val
  return d() if callable(d) else d

def masked_normalize(t, eps = 1e-5, mask = None, dim = None):
    dim = default(dim, tuple(range(t.ndim)))
    kwargs = dict(dim = dim, keepdim = True)

    mean = masked_mean(t, mask = mask, **kwargs)
    mean_centered = t - mean
    var = masked_mean(mean_centered ** 2, mask = mask, **kwargs)

    return mean_centered * var.clamp(min = eps).rsqrt()

def pad_sequence_fixed(sequences, *args, **kwargs):
  first_el = sequences[0]
  has_no_dimension = first_el.ndim == 0

  # if no dimensions, add a single dimension
  if has_no_dimension:
    sequences = tuple(map(lambda t: t[None], sequences))

  out = pad_sequence(sequences, *args, **kwargs)

  if has_no_dimension:
    out = rearrange(out, '... 1 -> ...')

  return out

def log(t, eps = 1e-20):
  return torch.log(t.clamp(min = eps))

def log_prob(prob, indices):
  assert prob.shape[:2] == indices.shape, f'preceding shapes of prob {prob.shape[:2]} and indices {indices.shape} must match'
  return log(prob.gather(-1, indices[..., None])).squeeze(-1)

def masked_mean(seq, mask = None, dim = 1, keepdim = False):
    if not exists(mask):
        return seq.mean(dim = dim)

    if seq.ndim == 3:
        mask = rearrange(mask, 'b n -> b n 1')

    masked_seq = seq.masked_fill(~mask, 0.)
    numer = masked_seq.sum(dim = dim, keepdim = keepdim)
    denom = mask.sum(dim = dim, keepdim = keepdim)

    masked_mean = numer / denom.clamp(min = 1e-3)
    masked_mean = masked_mean.masked_fill(denom == 0, 0.)
    return masked_mean

def masked_entropy(prob, dim = -1, mask = None):
    entropies = (prob * log(prob)).sum(dim = -1)
    return masked_mean(entropies, mask = mask).mean()

def masked_kl_div(prob1, prob2, mask = None, reduce_batch = False):
    """
    need to account for variable sequence lengths, therefore not using the built-in functional version
    """
    kl_divs = (prob1 * (log(prob1) - log(prob2))).sum(dim = -1)
    loss = masked_mean(kl_divs, mask)

    if reduce_batch:
        return loss.mean()

    return loss

def clipped_value_loss(values, rewards, old_values, clip):
    value_clipped = old_values + (values - old_values).clamp(-clip, clip)
    value_loss_1 = (value_clipped.flatten() - rewards) ** 2
    value_loss_2 = (values.flatten() - rewards) ** 2
    return torch.mean(torch.max(value_loss_1, value_loss_2))

# RLHF Trainer

class RLHFTrainer(nn.Module):
  def __init__(
    self,
    *,
    prompt_token_ids = None, #: Optional[torch.Tensor] = None,
    figaro,
    reward_model,
    critic_figaro = None,
    actor_critic = None,
    actor_lr = 1e-4,
    critic_lr = 1e-4,
    actor_wd = 0.,
    critic_wd = 0.,
    actor_adam_eps = 1e-7,
    critic_adam_eps = 1e-7,
    critic_pooled_values = True,
    betas = (0.9, 0.999),
    max_norm = None,
    eps_clip = 0.2,
    value_clip = 0.4,
    beta_s = .01,
    minibatch_size = 16,
    epochs = 1,
    kl_div_loss_weight = 0.1, # between old action probs and new action probs - not sure what the right value is
    accelerate_kwargs: dict = {},
    use_lion = False
):
    super().__init__()

    self.accelerate = Accelerator(**accelerate_kwargs)

    # TODO: assert that must be called with batch of prompt token ids, move prompt id loading outside of class

    # assert (exists(prompt_token_ids) == 1)

    # self.num_prompts = prompt_token_ids.shape[0]
    # self.register_buffer('prompt_token_ids', prompt_token_ids)

    # models

    self.figaro = figaro

    if not exists(actor_critic):
      actor_critic = ActorCritic(
          figaro=figaro,
          critic_figaro=critic_figaro,
          pooled_values=critic_pooled_values,
        ).to(figaro.device)

    self.actor_critic = actor_critic

    self.reward_model = reward_model.eval()

    # train hyperparameters

    self.epochs = epochs
    self.minibatch_size = minibatch_size
    self.max_norm = max_norm

    self.kl_div_loss_weight = kl_div_loss_weight

    # optimizers

    self.actor_optim = transformers.AdamW(actor_critic.actor_parameters(), lr = actor_lr, weight_decay = actor_wd, betas = betas, eps = actor_adam_eps)
    self.critic_optim = transformers.AdamW(actor_critic.critic_parameters(), lr = critic_lr, weight_decay = critic_wd, betas = betas, eps = critic_adam_eps)

    # self.actor_optim = get_optimizer(actor_critic.actor_parameters(), lr = actor_lr, wd = actor_wd, betas = betas, eps = actor_adam_eps, use_lion = use_lion)
    # self.critic_optim = get_optimizer(actor_critic.critic_parameters(), lr = critic_lr, wd = critic_wd, betas = betas, eps = critic_adam_eps, use_lion = use_lion)

    # ppo hyperparams

    self.eps_clip = eps_clip
    self.value_clip = value_clip
    self.beta_s = beta_s

    # prepare with accelerator

    (
        self.actor_critic,
        self.reward_model,
        self.actor_optim,
        self.critic_optim
    ) = self.accelerate.prepare(
        self.actor_critic,
        self.reward_model,
        self.actor_optim,
        self.critic_optim
    )

  def print(self, msg):
    return self.accelerate.print(msg)

  def save(self, filepath = './rlhf-checkpoint.pt'):
    torch.save(self.actor_critic.state_dict(), filepath)

  def load(self, filepath = './rlhf-checkpoint.pt'):
    state_dict = torch.load(filepath)
    self.actor_critic.load_state_dict(state_dict)

  @property
  def device(self):
    return self.accelerate.device
  
  # TODO: add generate function
  # @torch.no_grad()
  # def generate(
  #   self,
  #   max_seq_len,
  #   *args,
  #   prompt,
  #   num_samples = 4,  # sample 4 per prompt and select the one with highest reward
  #   **kwargs
  # ):
  #   assert prompt.ndim == 1, 'only one prompt allowed at a time for now'
  #   prompt = repeat(prompt, 'n -> b n', b = num_samples)

  #   actor_critic = self.accelerate.unwrap_model(self.actor_critic)
  #   reward_model = self.accelerate.unwrap_model(self.reward_model)

  #   actor_critic.eval()

  #   (
  #       actions,
  #       sequences,
  #       mask,
  #       prompt_mask,
  #       action_logits,
  #       _
  #   ) = actor_critic.generate(
  #       prompt,
  #       *args,
  #       max_seq_len = max_seq_len,
  #       return_values = False,
  #       **kwargs
  #   )

  #   rewards = reward_model(
  #       sequences,
  #       prompt_mask = prompt_mask,
  #       mask = mask,
  #       sample = True
  #   )

  #   best_sequence_index = rewards.topk(1, dim = -1).indices

  #   best_sequence = sequences[best_sequence_index]
  #   best_sequence = rearrange(best_sequence, '1 ... -> ...')

  #   return best_sequence




  def learn(
      self,
      memories, #: Deque[Memory]
      writer,
      time
  ):
    # stack all data stored in the memories

    all_memories_stacked_and_padded = list(map(partial(pad_sequence_fixed, batch_first = True), zip(*memories)))

    # prepare dataloader for policy phase training

    dl = create_dataloader(all_memories_stacked_and_padded, self.minibatch_size, device = self.device)

    self.actor_critic.train()

    # PPO training

    'sequences',
    'positon_ids',
    'bar_ids',
    'action_prob',
    'sequences_log_prob',
    'reward',
    'value'

    for _ in range(self.epochs):
      for (
        sequences,
        position_ids,
        bar_ids,
        old_action_probs,
        old_sequences_log_probs,
        rewards,
        old_values
      ) in dl:
        
        sequences = rearrange(sequences, '1 1 n -> 1 n')
        position_ids = rearrange(position_ids, '1 1 n -> 1 n')
        bar_ids = rearrange(bar_ids, '1 1 n -> 1 n')
        rewards = rearrange(rewards, '1 1 n -> 1 n')
        old_sequences_log_probs = rearrange(old_sequences_log_probs, '1 1 n -> 1 n')
        
        action_logits, values = self.actor_critic(x=sequences,
                                                  position_ids=position_ids,
                                                  bar_ids=bar_ids)

        # # TODO: check if this shift is needed
        # # action_logits = shift(action_logits, shift = 1, dim = -2) # need to shift along sequence dimension by 1, since actions start from the last prompt (state) token
        # action_len = old_log_probs.shape[-1]

        action_probs = action_logits.softmax(dim = -1)
        sequences_log_probs = log_prob(action_probs, sequences)

        # calculate entropies, taking into account which part of the sequence is actually an action

        entropies = masked_entropy(action_probs, mask = None)

        # calculate kl div between old action probs and new ones, taking into account which part of the sequence is action or not

        kl_penalty = 0.

        if self.kl_div_loss_weight > 0:
            kl_penalty = masked_kl_div(old_action_probs, action_probs, mask = None) * self.kl_div_loss_weight

        # subtract the kl penalty from the rewards

        rewards = rewards - kl_penalty.mean()

        writer.add_scalar("reward", rewards[0].item(), time)

        # handle non-pooled values

        normalize_kwargs = dict()

        # if old_values.ndim == 2:
        #     old_values, values = map(lambda t: shift(t, shift = 1, dim = -2), (old_values, values))

        #     old_values = old_values[:, -action_len:]
        #     values = values[:, -action_len:]
        #     rewards = rearrange(rewards, 'b -> b 1')
        #     normalize_kwargs = dict(dim = -1, mask = action_masks[:, -action_len:])

        if values.ndim < rewards.ndim:
            values = rearrange(values, '... -> ... 1')

        # calculate clipped surrogate objective, classic PPO loss

        ratios = (sequences_log_probs - old_sequences_log_probs).exp()
        advantages = masked_normalize(rewards - old_values, **normalize_kwargs)

        if advantages.ndim == 1:
            advantages = rearrange(advantages, 'b -> b 1')

        surr1 = ratios * advantages
        surr2 = ratios.clamp(1 - self.eps_clip, 1 + self.eps_clip) * advantages
        policy_loss = - torch.min(surr1, surr2) - self.beta_s * entropies

        # combine losses

        loss = policy_loss.mean()

        # update actor

        self.accelerate.backward(loss)

        self.print(f'policy_loss: {loss.item():.3f}')

        writer.add_scalar("policy_loss", loss.item(), time)

        if exists(self.max_norm):
            self.accelerator.clip_grad_norm_(self.actor_critic.actor_parameters(), self.max_norm)

        self.actor_optim.step()
        self.actor_optim.zero_grad()

        # calculate value loss and update value network separate from policy network

        value_loss = clipped_value_loss(values, rewards.detach(), old_values, self.value_clip)
        value_loss = value_loss.mean()

        self.print(f'critic_loss: {value_loss.item():.3f}')

        writer.add_scalar("critic_loss", value_loss.item(), time)

        self.accelerate.backward(value_loss)

        if exists(self.max_norm):
            self.accelerator.clip_grad_norm_(self.actor_critic.critic_parameters(), self.max_norm)

        self.critic_optim.step()
        self.critic_optim.zero_grad()


  def train(
    self,
    num_episodes = 5000,
    max_timesteps = 1000,
    update_timesteps = 100, # 5000,
    temperature = 1.
  ):
    device = self.device

    writer = SummaryWriter(LOGGING_DIR)

    time = 0
    memories = deque([])

    # Prepare prompts (TODO: move this outside of RLHFTrainer)

    midi_files = glob.glob(os.path.join(ROOT_DIR, '**/*.mid'), recursive=True)
  
    prompt_dm = self.figaro.get_datamodule(midi_files, vae_module=None)
    prompt_dm.setup('test')
    midi_files = prompt_dm.test_ds.files
    random.shuffle(midi_files)

    prompt_dataset = MidiDataset(
      midi_files,
      max_len=-1,
      description_flavor=self.figaro.description_flavor,
      description_options=None,
      max_bars=self.figaro.context_size,
      vae_module=None
    )

    prompt_coll = SeqCollator(context_size=-1)
    prompt_dl = DataLoader(prompt_dataset, batch_size=BATCH_SIZE, collate_fn=prompt_coll)

    # Get 3 bars for prompt
    prompt_dl_short = medley_iterator(prompt_dl, 
      n_pieces=N_MEDLEY_PIECES,
      n_bars=N_MEDLEY_BARS, 
      description_flavor=self.figaro.description_flavor
    )
    # Max number of REMI tokens to use as a prompt
    initial_context=10000

    for eps in tqdm(range(num_episodes), desc = 'episodes'):
      for timestep in range(max_timesteps):
        time += 1

        # select a bunch of random states (prompts)
        # and get the action (sampled sequence from palm as well as the action probs)
        # also calculate the reward using reward model and store

        # rand_prompt_index = randrange(0, self.num_prompts)

        # TODO: currently batch_short is a single prompt, but we want to sample a batch of prompts
        # state = self.prompt_token_ids[rand_prompt_index]
        batch_short = next(iter(prompt_dl_short))

        # get predicted sequence
        # TODO: generate only context_len tokens; look at max_len in generate function
        (
            actions,
            action_logits,
            value
        ) = self.actor_critic.generate(
            batch_short,
            max_initial_context=initial_context,
            output_dir=None,
            max_len=256,
            max_bars=MAX_BARS,
            verbose=10
        )

        # TODO: do we need to shift here?
        # action_logits = shift(action_logits, shift = 1, dim = -2) # need to shift along sequence dimension by 1, since actions start from the last prompt (state) token

        action_prob = action_logits.softmax(dim = -1)

        action_len = action_logits.size()[-1]
        action_log_prob = {}
        action_log_prob['sequences'] = log_prob(action_prob, actions['sequences'])
        action_log_prob['position_ids'] = log_prob(action_prob, actions['position_ids'])
        action_log_prob['bar_ids'] = log_prob(action_prob, actions['bar_ids'])
        # action_log_prob = action_log_prob[:, -action_len:]

        # get reward as given by supervised trained reward model

        reward = self.reward_model(
            x=actions['sequences'],
            bar_ids=actions['bar_ids'],
            position_ids=actions['position_ids']
        )

        # detach to cpu

        for key in actions:
          actions[key] = actions[key].detach().cpu()
        action_prob = action_prob.detach().cpu()
        for key in action_log_prob:
          action_log_prob[key] = action_log_prob[key].detach().cpu()
        reward = reward.detach().cpu()
        value = value.detach().cpu()


        # store memory for learning

        memories.append(Memory(
            actions['sequences'],
            actions['position_ids'],
            actions['bar_ids'],
            action_prob,
            action_log_prob['sequences'],
            reward,
            value
        ))

        # learn from the stored memories

        if time % update_timesteps == 0:
          self.learn(memories, writer, time)
          memories.clear()

    writer.close()
    print('rlhf training complete')


def main():

  # load your pretrained figaro

  model = Seq2SeqModule.load_from_checkpoint(CHECKPOINT).to(device)

  # Load reward checkpoint
  checkpoint_path = './results/reward_9_last.ckpt'
  checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

  # Load reward model
  reward_model = RewardModule(backbone_checkpoint='./checkpoints/baseline.ckpt')
  reward_model.load_state_dict(checkpoint['state_dict'])
  reward_model.eval()

  # ready your list of prompts for reinforcement learning

  # TODO: move out of RLHFTrainer

  # pass it all to the trainer and train

  trainer = RLHFTrainer(
      figaro = model,
      reward_model = reward_model,
  )

  trainer.train(num_episodes = 50000)

  # TODO: then, if it succeeded...
  # generate say 10 samples and use the reward model to return the best one

  # answer = trainer.generate(2048, prompt = prompts[0], num_samples = 10) # (<= 2048,)


if __name__ == '__main__':
  main()
