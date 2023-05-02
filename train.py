#!/usr/bin/env python3
import io
import requests
import PIL.Image, PIL.ImageDraw

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn.functional as F

from IPython.display import clear_output

from lib.CAModel import CAModel
from lib.utils_vis import SamplePool, to_rgb, make_seed, make_circle_masks

device = torch.device("cuda:0")
model_path = "models/new_1.pth"

PADDING = 16   # padding
CHANNEL_N = 16        # the states in the channel
IMAGE_SIZE = 40

n_epoch = 8000
lr_gamma = 0.9999
lr = 2e-3
betas = (0.5, 0.5)

POOL_SIZE = 1024
BATCH_SIZE = 8
CELL_FIRE_RATE = 0.5

TARGET_EMOJI = 0 #@param "🦎"

TEST_MAP = {"Growing":0, "Persistent":1, "Regenerating":2}
TEST_TYPE = "Regenerating"
TEST_N = TEST_MAP[TEST_TYPE]

POOL_PATTERN = [0, 1, 1][TEST_N]
DAMAGE_N = [0, 0, 3][TEST_N]  # amount of patterns to damage

def load_image(url, max_size=IMAGE_SIZE):
  r = requests.get(url)
  img = PIL.Image.open(io.BytesIO(r.content))
  img.thumbnail((max_size, max_size), PIL.Image.ANTIALIAS)
  img = np.float32(img)/255.0
  img[..., :3] *= img[..., 3:]
  return img

def load_emoji(index, path="data/usa.png"):
    url = 'https://em-content.zobj.net/thumbs/120/google/350/red-heart_2764-fe0f.png'
    target_img = load_image(url, 48)
    return target_img

def show_batch(x0, x):
    vis0 = to_rgb(x0)
    vis1 = to_rgb(x)
    print('batch (before/after):')
    plt.figure(figsize=[15,5])
    for i in range(x0.shape[0]):
        plt.subplot(2,x0.shape[0],i+1)
        plt.imshow(vis0[i])
        plt.axis('off')
    for i in range(x0.shape[0]):
        plt.subplot(2,x0.shape[0],i+1+x0.shape[0])
        plt.imshow(vis1[i])
        plt.axis('off')
    plt.show()

def show_loss(loss_log):
    plt.figure(figsize=(10, 4))
    plt.title('Loss history (log10)')
    plt.plot(np.log10(loss_log), '.', alpha=0.1)

target_img = load_emoji(TARGET_EMOJI)


p = PADDING
pad_target = np.pad(target_img, [(p, p), (p, p), (0, 0)])
h, w = pad_target.shape[:2]
pad_target = np.expand_dims(pad_target, axis=0)
pad_target = torch.from_numpy(pad_target.astype(np.float32)).to(device)

seed = make_seed((h, w), CHANNEL_N)
pool = SamplePool(x=np.repeat(seed[None, ...], POOL_SIZE, 0))
batch = pool.sample(BATCH_SIZE).x
ca = CAModel(CHANNEL_N, CELL_FIRE_RATE, device).to(device)
ca.load_state_dict(torch.load(model_path))
optimizer = optim.Adam(ca.parameters(), lr=lr, betas=betas)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, lr_gamma)
loss_log = []

def train(x, target, steps, optimizer, scheduler):
    x = ca(x, steps=steps)
    loss = F.mse_loss(x[:, :, :, :4], target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    return x, loss

def loss_f(x, target):
    return torch.mean(torch.pow(x[..., :4]-target, 2), [-2,-3,-1])

for i in range(n_epoch+1):
    if POOL_PATTERN:
        batch = pool.sample(BATCH_SIZE)
        x0 = torch.from_numpy(batch.x.astype(np.float32)).to(device)
        loss_rank = loss_f(x0, pad_target).detach().cpu().numpy().argsort()[::-1]
        x0 = batch.x[loss_rank]
        x0[:1] = seed
        if DAMAGE_N:
            damage = 1.0-make_circle_masks(DAMAGE_N, h, w)[..., None]
            x0[-DAMAGE_N:] *= damage
    else:
        x0 = np.repeat(seed[None, ...], BATCH_SIZE, 0)
    x0 = torch.from_numpy(x0.astype(np.float32)).to(device)

    x, loss = train(x0, pad_target, np.random.randint(64,96), optimizer, scheduler)

    if POOL_PATTERN:
        batch.x[:] = x.detach().cpu().numpy()
        batch.commit()

    step_i = len(loss_log)
    loss_log.append(loss.item())

    if step_i%100 == 0:
        clear_output()
        print(step_i, "loss =", loss.item())
        show_batch(x0.detach().cpu().numpy(), x.detach().cpu().numpy())
        show_loss(loss_log)
        torch.save(ca.state_dict(), model_path)
