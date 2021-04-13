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
import numpy as np

class ClipEnv():
  
  def __init__(self, args):
    # print("In init")
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
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
    self.preprocess = Normalize(
      (0.48145466, 0.4578275, 0.40821073),
      (0.26862954, 0.26130258, 0.27577711))

  def _get_state(self):
    # print("_get_state")
    ale_image = self.ale.getScreenRGB() # (210, 160, 3)
    tensor = torch.tensor(ale_image, device=self.device, dtype=torch.float32)
    tensor = tensor.div_(255)
    permuted = tensor.permute(2, 0, 1).unsqueeze(0)
    return F.interpolate(permuted, size=224, mode='bicubic').squeeze(0)

  def get_clip_features(self, images):
    with torch.no_grad():
      image_features = self.clip_model.encode_image(self.preprocess(images)).float()
    return image_features

  def _reset_buffer(self):
    # print("_reset_buffer")
    for _ in range(self.window):
      # TODO don't hard-code
      self.state_buffer.append(torch.zeros(3, 224, 224, device=self.device))

  def reset(self):
    # print("reset")
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
    state = self.get_clip_features(torch.stack(list(self.state_buffer), 0))
    print(state.shape)
    return state

  def step(self, action):
    # print("step")
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
    # print("self.state_buffer shape", np.array(self.state_buffer).shape)
    # Return state, reward, done
    state = self.get_clip_features(torch.stack(list(self.state_buffer), 0))
    return state, reward, done

  # Uses loss of life as terminal signal
  def train(self):
    # print("train")
    self.training = True

  # Uses standard terminal signal
  def eval(self):
    # print("eval")
    self.training = False

  def action_space(self):
    # print("action_space")
    return len(self.actions)

  def render(self):
    # print("render")
    cv2.imshow('screen', self.ale.getScreenRGB()[:, :, ::-1])
    cv2.waitKey(1)

  def close(self):
    # print("close")
    cv2.destroyAllWindows()
