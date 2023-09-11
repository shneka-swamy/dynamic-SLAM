# This code is an edited version of the FlowFormer code.

import sys
sys.path.append('core')

import numpy as np
import torch
import torch.nn.functional as F
from FlowFormer.configs.submission import get_cfg
from FlowFormer.configs.small_things_eval import get_cfg as get_small_things_cfg
from FlowFormer.core.utils import flow_viz
import cv2
import math
from matplotlib import pyplot as plt

from FlowFormer.core.FlowFormer import build_flowformer

from FlowFormer.core.utils.utils import InputPadder


TRAIN_SIZE = [432, 960]

def compute_grid_indices(image_shape, patch_size=TRAIN_SIZE, min_overlap=20):
  if min_overlap >= TRAIN_SIZE[0] or min_overlap >= TRAIN_SIZE[1]:
    raise ValueError(
        f"Overlap should be less than size of patch (got {min_overlap}"
        f"for patch size {patch_size}).")
  if image_shape[0] == TRAIN_SIZE[0]:
    hs = list(range(0, image_shape[0], TRAIN_SIZE[0]))
  else:
    hs = list(range(0, image_shape[0], TRAIN_SIZE[0] - min_overlap))
  if image_shape[1] == TRAIN_SIZE[1]:
    ws = list(range(0, image_shape[1], TRAIN_SIZE[1]))
  else:
    ws = list(range(0, image_shape[1], TRAIN_SIZE[1] - min_overlap))

  # Make sure the final patch is flush with the image boundary
  hs[-1] = image_shape[0] - patch_size[0]
  ws[-1] = image_shape[1] - patch_size[1]
  return [(h, w) for h in hs for w in ws]

def compute_weight(hws, image_shape, patch_size=TRAIN_SIZE, sigma=1.0, wtype='gaussian'):
    patch_num = len(hws)
    h, w = torch.meshgrid(torch.arange(patch_size[0]), torch.arange(patch_size[1]))
    h, w = h / float(patch_size[0]), w / float(patch_size[1])
    c_h, c_w = 0.5, 0.5 
    h, w = h - c_h, w - c_w
    weights_hw = (h ** 2 + w ** 2) ** 0.5 / sigma
    denorm = 1 / (sigma * math.sqrt(2 * math.pi))
    weights_hw = denorm * torch.exp(-0.5 * (weights_hw) ** 2)

    weights = torch.zeros(1, patch_num, *image_shape)
    for idx, (h, w) in enumerate(hws):
        weights[:, idx, h:h+patch_size[0], w:w+patch_size[1]] = weights_hw
    weights = weights.cuda()
    patch_weights = []
    for idx, (h, w) in enumerate(hws):
        patch_weights.append(weights[:, idx:idx+1, h:h+patch_size[0], w:w+patch_size[1]])

    return patch_weights

def compute_flow(model, image1, image2, weights=None):
    image_size = image1.shape[1:]

    image1, image2 = image1[None].cuda(), image2[None].cuda()

    hws = compute_grid_indices(image_size)
    if weights is None:     # no tile
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        flow_pre, _ = model(image1, image2)

        flow_pre = padder.unpad(flow_pre)
        flow = flow_pre[0].permute(1, 2, 0).cpu().numpy()
    else:                   # tile
        flows = 0
        flow_count = 0

        for idx, (h, w) in enumerate(hws):
            image1_tile = image1[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
            image2_tile = image2[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]    
            flow_pre, _ = model(image1_tile, image2_tile)
            padding = (w, image_size[1]-w-TRAIN_SIZE[1], h, image_size[0]-h-TRAIN_SIZE[0], 0, 0)
            flows += F.pad(flow_pre * weights[idx], padding)
            flow_count += F.pad(weights[idx], padding)

        flow_pre = flows / flow_count
        flow = flow_pre[0].permute(1, 2, 0).cpu().numpy()

    return flow

def compute_adaptive_image_size(image_size):
    target_size = TRAIN_SIZE
    scale0 = target_size[0] / image_size[0]
    scale1 = target_size[1] / image_size[1] 

    if scale0 > scale1:
        scale = scale0
    else:
        scale = scale1

    image_size = (int(image_size[1] * scale), int(image_size[0] * scale))

    return image_size

def prepare_image(image1, image2, keep_size):
    image1 = np.array(image1).astype(np.uint8)[..., :3]
    image2 = np.array(image2).astype(np.uint8)[..., :3]
    if not keep_size:
        dsize = compute_adaptive_image_size(image1.shape[0:2])
        image1 = cv2.resize(image1, dsize=dsize, interpolation=cv2.INTER_CUBIC)
        image2 = cv2.resize(image2, dsize=dsize, interpolation=cv2.INTER_CUBIC)
    image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
    image2 = torch.from_numpy(image2).permute(2, 0, 1).float()

    return image1, image2

def build_model():
    print(f"building  model...")
    cfg = get_cfg()
    model = torch.nn.DataParallel(build_flowformer(cfg))
    model.load_state_dict(torch.load(cfg.model))

    model.cuda()
    model.eval()

    return model

def visualize_flow(image1, image2, model, keep_size):
    weights = None
    image1, image2 = prepare_image(image1, image2, keep_size)
    flow = compute_flow(model, image1, image2, weights)


    u_flow, v_flow = flow[:, :, 0], flow[:, :, 1]
    # hist, x_edges, y_edges = np.histogram2d(flow[..., 0].ravel(), flow[..., 1].ravel(), bins=(u_flow, v_flow))
    
    flow_mag = np.sqrt(u_flow ** 2 + v_flow ** 2)
    #flow_angle = np.arctan2(v_flow, u_flow)
    # convert to degrees
    #flow_angle = cv2.cartToPolar(u_flow, v_flow, angleInDegrees=True)[1]
    # accept angles from 100 to 280 and angles from 80 to 260
    #motion_mask = np.logical_or(np.logical_and(flow_angle > 100, flow_angle < 280), np.logical_and(flow_angle > 80, flow_angle < 260))
    
    flow_mean = np.mean(flow_mag)
    motion_mask = flow_mag - flow_mean
    motion_mask_rev = flow_mean - flow_mag
    # remove negative values
    #motion_mask = np.clip(motion_mask, 0, None)
    motion_mask = motion_mask.astype(np.uint8)
    motion_mask_rev = motion_mask_rev.astype(np.uint8)
    # Draw the motion mask
    #cv2.imwrite(viz_fn, motion_mask.astype(np.uint8) * 255)
    
    # flow_img = flow_viz.flow_to_image(flow)
    # image = flow_img[:, :, [2, 1, 0]]
    # # convert to grayscale
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # # print("image shape = ", image.shape)
    # # print("image type = ", image.dtype)
    # #contour_image = np.ones_like(image)
    # contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(image, contours, -1, (0, 255, 0), 1)
    
    image1 = motion_mask
    image2 = motion_mask_rev
    # image is intersection of both images
    image = (image1 + image2)/2

    # # Plot 4 images as subplots
    # fig, axs = plt.subplots(2, 2)
    # axs[0, 0].imshow(image1, cmap='gray')
    # axs[0, 0].set_title('image1')
    # axs[0, 1].imshow(image2, cmap='gray')
    # axs[0, 1].set_title('image2')
    # axs[1, 0].imshow(image, cmap='gray')
    # axs[1, 0].set_title('image')
    # plt.show()

    
    image = image1

    return image
       
  