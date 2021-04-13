# -*- coding: utf-8 -*-
from collections import deque
import random
import atari_py
import cv2
import torch
from torch.nn import functional as F
import clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image

class ClipEnv():
  
  def __init__(self, args):
    print("In init")
    self.device = args.device
    self.ale = atari_py.ALEInterface()
    self.ale.setInt('random_seed', args.seed)
    self.ale.setInt('max_num_frames_per_episode', args.max_episode_length)
    self.ale.setFloat('repeat_action_probability', 0)  # Disable sticky actions
    self.ale.setInt('frame_skip', 0)
    self.ale.setBool('color_averaging', False)
    self.ale.loadROM(atari_py.get_game_path(args.game))  # ROM loading must be done after setting options
    actions = self.ale.getMinimalActionSet()
    self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
    self.lives = 0  # Life counter (used in DeepMind training)
    self.life_termination = False  # Used to check if resetting only from loss of life
    self.window = args.history_length  # Number of frames to concatenate
    self.state_buffer = deque([], maxlen=args.history_length)
    self.training = True  # Consistent with model training mode
    device = "cuda" if torch.cuda.is_available() else "cpu"
    self.clip_model, _ = clip.load("ViT-B/32", device=device)

  def _get_state(self):
    print("_get_state")
    state = cv2.resize(self.ale.getScreenGrayscale(), (224, 224), interpolation=cv2.INTER_LINEAR)
    # ale_shape = self.ale.getScreenRGB() # (210, 160, 3)
    # screen_tensor = torch.tensor(ale_shape, dtype=torch.float32, device=self.device).div_(255) # torch.Size([210, 160, 3])
    # screen_tensor_permuted = screen_tensor.permute(2, 0, 1) # torch.Size([3, 210, 160])
    # padded_state = self._pad_observation(screen_tensor_permuted)
    # print("padded_state shape", padded_state.shape)
    return state

  def _pad_observation(self, image):
    print("_pad_observation")
    ale_shape = (3, 210, 160) # image.shape = 210, 160
    preprocess = Compose([
    Resize(224, interpolation=Image.BICUBIC),
    CenterCrop(224),
    # ToTensor()
    ])
    # pad_width = ((224 - ale_shape[1])//2, (224 - ale_shape[1])//2, (224 - ale_shape[0])//2, (224 - ale_shape[0])//2, 0, 0)
    # padded_image = F.pad(image, pad=pad_width, mode='constant', value=0) # torch.Size([3, 430, 174])
    resized_image = preprocess(image)
    # print("padded_image shape constant", padded_image.shape)
    print("resized_image shape constant", resized_image.shape)
    return resized_image

  def _encode(self, images):
    print("_encode")
    embeddings = self.clip_model(torch.Tensor(images))
    return embeddings

  def _reset_buffer(self):
    print("_reset_buffer")
    for _ in range(self.window):
      # TODO don't hard-code
      self.state_buffer.append(torch.zeros(3, 224, 224, device=self.device))

  def reset(self):
    print("reset")
    if self.life_termination:
      self.life_termination = False  # Reset flag
      self.ale.act(0)  # Use a no-op after loss of life
    else:
      # Reset internals
      self._reset_buffer()
      self.ale.reset_game()
      # Perform up to 30 random no-ops before starting
      for _ in range(random.randrange(30)):
        self.ale.act(0)  # Assumes raw action 0 is always no-op
        if self.ale.game_over():
          self.ale.reset_game()
    # Process and return "initial" state
    observation = self._get_state()
    self.state_buffer.append(observation)
    self.lives = self.ale.lives()
    return torch.stack(list(self.state_buffer), 0)

  def step(self, action):
    print("step")
    # Repeat action 4 times, max pool over last 2 frames
    frame_buffer = torch.zeros(2, 3, 224, 224, device=self.device)
    reward, done = 0, False
    for t in range(4):
      reward += self.ale.act(self.actions.get(action))
      if t == 2:
        frame_buffer[0] = self._get_state()
      elif t == 3:
        frame_buffer[1] = self._get_state()
      done = self.ale.game_over()
      if done:
        break
    observation = frame_buffer.max(0)[0]
    self.state_buffer.append(observation)
    # Detect loss of life as terminal in training mode
    if self.training:
      lives = self.ale.lives()
      if lives < self.lives and lives > 0:  # Lives > 0 for Q*bert
        self.life_termination = not done  # Only set flag when not truly done
        done = True
      self.lives = lives
    # Return state, reward, done
    return torch.stack(list(self.state_buffer), 0), reward, done

  # Uses loss of life as terminal signal
  def train(self):
    print("train")
    self.training = True

  # Uses standard terminal signal
  def eval(self):
    print("eval")
    self.training = False

  def action_space(self):
    print("action_space")
    return len(self.actions)

  def render(self):
    print("render")
    cv2.imshow('screen', self.ale.getScreenRGB()[:, :, ::-1])
    cv2.waitKey(1)

  def close(self):
    print("close")
    cv2.destroyAllWindows()
