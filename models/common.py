import math
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp
import torch.nn.functional as F

from utils.datasets import letterbox
from utils.general import non_max_suppression, make_divisible, scale_coords, increment_path, xyxy2xywh, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import time_synchronized

from torch.nn import init, Sequential


def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


def DWConv(c1, c2, k=1, s=1, act=True):
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class TransformerLayer(nn.Module):
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)
        self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2)
        p = p.unsqueeze(0)
        p = p.transpose(0, 3)
        p = p.squeeze(3)
        e = self.linear(p)
        x = p + e

        x = self.tr(x)
        x = x.unsqueeze(3)
        x = x.transpose(0, 3)
        x = x.reshape(b, self.c2, w, h)
        return x


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(C3, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C3TR(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class SPP(nn.Module):
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Focus(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Focus, self).__init__()
        self.branch_id = branch_id=0
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):
        x_patch = torch.cat([
            x[..., ::2, ::2], x[..., 1::2, ::2],
            x[..., ::2, 1::2], x[..., 1::2, 1::2]
        ], 1)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


class Contract(nn.Module):
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
     
        N, C, H, W = x.size()
        s = self.gain
        x = x.view(N, C, H // s, s, W // s, s)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        return x.view(N, C * s * s, H // s, W // s)


class Expand(nn.Module):
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()
        s = self.gain
        x = x.view(N, s, s, C // s ** 2, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
        return x.view(N, C // s ** 2, H * s, W * s)


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class Add(nn.Module):
    def __init__(self, arg):
        super(Add, self).__init__()
        self.arg = arg

    def forward(self, x):
        return torch.add(torch.add(x[0], x[1]),x[2])



class Add2(nn.Module):
    def __init__(self, c1, index, *args):
        super().__init__()
        self.index = index
        self.c1 = c1

    def forward(self, x):
        if self.index == 0:
            return torch.add(x[0], x[1][0])
        elif self.index == 1:
            return torch.add(x[0], x[1][1])
        elif self.index == 2:
            return torch.add(x[0], x[1][2])
        return torch.add(x[0], x[1][0])



class NMS(nn.Module):
    conf = 0.25
    iou = 0.45
    classes = None

    def __init__(self):
        super(NMS, self).__init__()

    def forward(self, x):
        return non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)


class autoShape(nn.Module):
    conf = 0.25
    iou = 0.45
    classes = None

    def __init__(self, model):
        super(autoShape, self).__init__()
        self.model = model.eval()

    def autoshape(self):
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):
        t = [time_synchronized()]
        p = next(self.model.parameters())
        if isinstance(imgs, torch.Tensor):
            with amp.autocast(enabled=p.device.type != 'cpu'):
                return self.model(imgs.to(p.device).type_as(p), augment, profile)

        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])
        shape0, shape1, files = [], [], []
        for i, im in enumerate(imgs):
            f = f'image{i}'
            if isinstance(im, str):
                im, f = np.asarray(Image.open(requests.get(im, stream=True).raw if im.startswith('http') else im)), im
            elif isinstance(im, Image.Image):
                im, f = np.asarray(im), getattr(im, 'filename', f) or f
            files.append(Path(f).with_suffix('.jpg').name)
            if im.shape[0] < 5:
                im = im.transpose((1, 2, 0))
            im = im[:, :, :3] if im.ndim == 3 else np.tile(im[:, :, None], 3)
            s = im.shape[:2]
            shape0.append(s)
            g = (size / max(s))
            shape1.append([y * g for y in s])
            imgs[i] = im if im.data.contiguous else np.ascontiguousarray(im)
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]
        x = np.stack(x, 0) if n > 1 else x[0][None]
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255.
        t.append(time_synchronized())

        with amp.autocast(enabled=p.device.type != 'cpu'):
            y = self.model(x, augment, profile)[0]
            t.append(time_synchronized())

            y = non_max_suppression(y, conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])

            t.append(time_synchronized())
            return Detections(imgs, y, files, t, self.names, x.shape)


class Detections:
    def __init__(self, imgs, pred, files, times=None, names=None, shape=None):
        super(Detections, self).__init__()
        d = pred[0].device
        gn = [torch.tensor([*[im.shape[i] for i in [1, 0, 1, 0]], 1., 1.], device=d) for im in imgs]
        self.imgs = imgs
        self.pred = pred
        self.names = names
        self.files = files
        self.xyxy = pred
        self.xywh = [xyxy2xywh(x) for x in pred]
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]
        self.n = len(self.pred)
        self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))
        self.s = shape

    def display(self, pprint=False, show=False, save=False, crop=False, render=False, save_dir=Path('')):
        for i, (im, pred) in enumerate(zip(self.imgs, self.pred)):
            str_ = f'image {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '
            if pred is not None:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()
                    str_ += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "
                if show or save or render or crop:
                    for *box, conf, cls in pred:
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if crop:
                            save_one_box(box, im, file=save_dir / 'crops' / self.names[int(cls)] / self.files[i])
                        else:
                            plot_one_box(box, im, label=label, color=colors(cls))

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im
            if pprint:
                print(str_.rstrip(', '))
            if show:
                im.show(self.files[i])
            if save:
                f = self.files[i]
                im.save(save_dir / f)
                print(f"{'Saved' * (i == 0)} {f}", end=',' if i < self.n - 1 else f' to {save_dir}\n')
            if render:
                self.imgs[i] = np.asarray(im)

    def print(self):
        self.display(pprint=True)
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' % self.t)

    def show(self):
        self.display(show=True)

    def save(self, save_dir='runs/hub/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/hub/exp', mkdir=True)
        self.display(save=True, save_dir=save_dir)

    def crop(self, save_dir='runs/hub/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/hub/exp', mkdir=True)
        self.display(crop=True, save_dir=save_dir)
        print(f'Saved results to {save_dir}\n')

    def render(self):
        self.display(render=True)
        return self.imgs

    def pandas(self):
        new = copy(self)
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        x = [Detections([self.imgs[i]], [self.pred[i]], self.names, self.s) for i in range(self.n)]
        for d in x:
            for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
                setattr(d, k, getattr(d, k)[0])
        return x

    def __len__(self):
        return self.n


class Classify(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)
        return self.flat(self.conv(z))


class Eventpooling(nn.Module):
    def __init__(self, embed_dims=[64, 128, 256], inplanes=3, downsamping=4, ca_num_heads=4, kernel=[3, 5, 7, 9], padding=[1, 2, 3, 4], stage=2, **kwargs):
        super().__init__()
        self.ca_num_heads = ca_num_heads
        self.kernel = kernel
        self.padding = padding
        self.embed_dims = embed_dims
        self.stage = stage
        self.conv0_0 = nn.Conv2d(in_channels=inplanes, out_channels=embed_dims[0], kernel_size=1, stride=1, padding=0)
        self.conv0_1 = nn.Conv2d(in_channels=embed_dims[0], out_channels=embed_dims[0], kernel_size=1, stride=1, padding=0)

        self.max_pool0_0 = nn.MaxPool2d(kernel_size=int(downsamping/2), stride=int(downsamping/2))
        self.avg_pool0_0 = nn.AvgPool2d(kernel_size=int(downsamping/2), stride=int(downsamping/2))

        for i in range(self.ca_num_heads):
            max_pool = nn.MaxPool2d(kernel_size=self.kernel[i], stride=1, padding=self.padding[i])
            setattr(self, f"max_pool_{i}", max_pool)

        for i in range(self.stage):
            conv0 = nn.Conv2d(in_channels=embed_dims[i], out_channels=embed_dims[i+1], kernel_size=2, stride=2, padding=0)
            setattr(self, f"conv_{i+1}_0", conv0)
            conv1 = nn.Conv2d(in_channels=embed_dims[i+1], out_channels=embed_dims[i+1], kernel_size=1, stride=1,padding=0)
            setattr(self, f"conv_{i+1}_1", conv1)
            conv2 = nn.Conv2d(in_channels=embed_dims[i+1], out_channels=embed_dims[i+1], kernel_size=1, stride=1, padding=0)
            setattr(self, f"conv_{i+1}_2", conv2)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv0_0(x))
        x = self.max_pool0_0(x)
        x = self.relu(self.conv0_1(x))
        x_avg = self.avg_pool0_0(x)
        x_max = self.max_pool0_0(x)
        x = x_max + x_avg

        for i in range(self.stage):
            conv0 = getattr(self, f"conv_{i+1}_0")
            x = self.relu(conv0(x))
            x_ori = x
            conv1 = getattr(self, f"conv_{i+1}_1")
            x = self.relu(conv1(x))
            channel_splits = torch.split(x, int(self.embed_dims[i+1]/4), dim=1)
            for s in range(self.ca_num_heads):
                if s == 0 or s == 1:
                    x_s = channel_splits[s]
                else:
                    x_s = x_s + channel_splits[s]
                max_pool = getattr(self, f"max_pool_{s}")
                x_s = max_pool(x_s)
                if s == 0:
                    x_out = x_s
                else:
                    x_out = torch.cat([x_out, x_s], dim=1)
            conv2 = getattr(self, f"conv_{i+1}_2")
            x_out = conv2(x_out)
            x = x_out + x_ori
            x = self.relu(x)
        return x


class EventReliabilityEstimator(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.estimator = nn.Sequential(
            nn.Conv2d(channels, channels//4, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels//4, 1, 1),
            nn.Sigmoid()
        )
        
        self.spatial_pool = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, event_feature):
        global_info = self.pool(event_feature)
        reliability = self.estimator(global_info)
        
        spatial_map = self.spatial_pool(event_feature)
        spatial_std = torch.std(spatial_map.view(spatial_map.size(0), -1), dim=1, keepdim=True).view(-1, 1, 1, 1)
        
        reliability = reliability * (spatial_std * 5).clamp(0, 1)
        
        return reliability


class AdaptiveEventMoEGate(nn.Module):
    def __init__(self, channels, reliability_threshold=0.3):
        super().__init__()
        self.event_estimator = EventReliabilityEstimator(channels)
        self.reliability_threshold = reliability_threshold
        
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels*3, channels//2, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels//2, 5, 1)
        )
        
        self.fallback_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels*2, channels//2, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels//2, 3, 1)
        )
    
    def forward(self, rgb_fea, ir_fea, event_fea, weight_dtype):
        event_reliability = self.event_estimator(event_fea)
        
        use_event = event_reliability > self.reliability_threshold
        
        batch_size = rgb_fea.size(0)
        final_weights = torch.zeros(batch_size, 5, 1, 1, device=rgb_fea.device, dtype=weight_dtype)
        
        if use_event.any():
            concat_input = torch.cat([rgb_fea, ir_fea, event_fea], dim=1)
            regular_weights = self.gate(concat_input)
            regular_weights = regular_weights.to(weight_dtype)
            
            rgb_weight = regular_weights[:, 0:1]
            ir_weight = regular_weights[:, 1:2]
            event_weight = regular_weights[:, 2:3] * event_reliability.to(weight_dtype)
            rgb_ir_fusion_weight = regular_weights[:, 3:4]
            tri_fusion_weight = regular_weights[:, 4:5] * event_reliability.to(weight_dtype)
            
            event_mask = use_event.float().view(-1, 1, 1, 1).to(weight_dtype)
            final_weights[:, 0:1] = rgb_weight * event_mask
            final_weights[:, 1:2] = ir_weight * event_mask
            final_weights[:, 2:3] = event_weight * event_mask
            final_weights[:, 3:4] = rgb_ir_fusion_weight * event_mask
            final_weights[:, 4:5] = tri_fusion_weight * event_mask
        
        fallback_input = torch.cat([rgb_fea, ir_fea], dim=1)
        fallback_weights = self.fallback_gate(fallback_input)
        fallback_weights = fallback_weights.to(weight_dtype)
        
        fallback_rgb = fallback_weights[:, 0:1]
        fallback_ir = fallback_weights[:, 1:2]
        fallback_rgb_ir = fallback_weights[:, 2:3]
        
        non_event_mask = (~use_event).float().view(-1, 1, 1, 1).to(weight_dtype)
        if non_event_mask.sum() > 0:
            final_weights[:, 0:1] = final_weights[:, 0:1] + fallback_rgb * non_event_mask
            final_weights[:, 1:2] = final_weights[:, 1:2] + fallback_ir * non_event_mask
            final_weights[:, 3:4] = final_weights[:, 3:4] + fallback_rgb_ir * non_event_mask
        
        sum_weights = final_weights.sum(dim=1, keepdim=True)
        sum_weights = torch.where(sum_weights > 0, sum_weights, torch.ones_like(sum_weights))
        final_weights = final_weights / sum_weights
        
        return final_weights, event_reliability, use_event


class ImprovedTriModalLightweightMoECAFF(nn.Module):
    def __init__(self, channels, reduction=4, heads=4, reliability_threshold=0.3, temperature=10.0):
        super().__init__()
        
        self.channels = channels
        self.reduced_dim = max(channels // reduction, 32)
        self.heads = heads
        self.reliability_threshold = reliability_threshold
        self.temperature = temperature
        
        self.reduce_rgb = nn.Sequential(
            nn.Conv2d(channels, self.reduced_dim, 1),
            nn.BatchNorm2d(self.reduced_dim),
            nn.SiLU(inplace=True)
        )
        
        self.reduce_ir = nn.Sequential(
            nn.Conv2d(channels, self.reduced_dim, 1),
            nn.BatchNorm2d(self.reduced_dim),
            nn.SiLU(inplace=True)
        )
        
        self.reduce_event = nn.Sequential(
            nn.Conv2d(channels, self.reduced_dim, 1),
            nn.BatchNorm2d(self.reduced_dim),
            nn.SiLU(inplace=True)
        )
        
        self.adaptive_moe_gate = AdaptiveEventMoEGate(
            channels, reliability_threshold=reliability_threshold
        )
        
        self.rgb_expert = nn.Sequential(
            nn.Conv2d(self.reduced_dim, self.reduced_dim, 3, padding=1, groups=self.reduced_dim//4),
            nn.BatchNorm2d(self.reduced_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.reduced_dim, self.reduced_dim, 1)
        )
        
        self.ir_expert = nn.Sequential(
            nn.Conv2d(self.reduced_dim, self.reduced_dim, 3, padding=1, groups=self.reduced_dim//4),
            nn.BatchNorm2d(self.reduced_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.reduced_dim, self.reduced_dim, 1)
        )
        
        self.event_expert = nn.Sequential(
            nn.Conv2d(self.reduced_dim, self.reduced_dim, 3, padding=1, groups=self.reduced_dim//4),
            nn.BatchNorm2d(self.reduced_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.reduced_dim, self.reduced_dim, 1)
        )
        
        self.rgb_ir_expert = nn.Sequential(
            nn.Conv2d(self.reduced_dim*2, self.reduced_dim, 1),
            nn.BatchNorm2d(self.reduced_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.reduced_dim, self.reduced_dim, 3, padding=1, groups=self.reduced_dim//4),
            nn.BatchNorm2d(self.reduced_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.reduced_dim, self.reduced_dim, 1)
        )
        
        self.rgb_to_ir_attn = nn.Sequential(
            nn.Conv2d(self.reduced_dim, self.reduced_dim, 1),
            nn.Sigmoid()
        )
        
        self.ir_to_rgb_attn = nn.Sequential(
            nn.Conv2d(self.reduced_dim, self.reduced_dim, 1),
            nn.Sigmoid()
        )
        
        self.rgb_to_event_attn = nn.Sequential(
            nn.Conv2d(self.reduced_dim, self.reduced_dim, 1),
            nn.Sigmoid()
        )
        
        self.event_to_rgb_attn = nn.Sequential(
            nn.Conv2d(self.reduced_dim, self.reduced_dim, 1),
            nn.Sigmoid()
        )
        
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.reduced_dim*3, self.reduced_dim, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.reduced_dim, self.reduced_dim, 1),
            nn.Sigmoid()
        )
        
        self.dual_global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.reduced_dim*2, self.reduced_dim, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.reduced_dim, self.reduced_dim, 1),
            nn.Sigmoid()
        )
        
        self.small_object_enhancer = nn.Sequential(
            nn.Conv2d(self.reduced_dim, self.reduced_dim, 3, padding=2, dilation=2, groups=self.reduced_dim//4),
            nn.BatchNorm2d(self.reduced_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.reduced_dim, self.reduced_dim, 1),
            nn.BatchNorm2d(self.reduced_dim),
            nn.SiLU(inplace=True)
        )
        
        self.out_rgb = nn.Sequential(
            nn.Conv2d(self.reduced_dim, channels, 1),
            nn.BatchNorm2d(channels)
        )
        
        self.out_ir = nn.Sequential(
            nn.Conv2d(self.reduced_dim, channels, 1),
            nn.BatchNorm2d(channels)
        )
        
        self.out_event = nn.Sequential(
            nn.Conv2d(self.reduced_dim, channels, 1),
            nn.BatchNorm2d(channels)
        )
        
    def forward(self, x):
        rgb_fea, ir_fea, event_fea = x[0], x[1], x[2]
        weight_dtype = self.out_rgb[0].weight.dtype
        
        rgb_fea = rgb_fea.to(weight_dtype)
        ir_fea = ir_fea.to(weight_dtype)
        event_fea = event_fea.to(weight_dtype)
        
        batch_size = rgb_fea.shape[0]
        
        rgb_reduced = self.reduce_rgb(rgb_fea).to(weight_dtype)
        ir_reduced = self.reduce_ir(ir_fea).to(weight_dtype)
        event_reduced = self.reduce_event(event_fea).to(weight_dtype)
        
        event_reliability = self.adaptive_moe_gate.event_estimator(event_fea)
        
        event_weight = torch.sigmoid((event_reliability - self.reliability_threshold) * self.temperature)
        event_weight = event_weight.view(-1, 1, 1, 1).to(weight_dtype)
        dual_weight = 1.0 - event_weight
        
        moe_output = self.adaptive_moe_gate(rgb_fea, ir_fea, event_fea, weight_dtype)
        if isinstance(moe_output, tuple) and len(moe_output) == 3:
            moe_weights, _, _ = moe_output
        else: 
            moe_weights = moe_output

        rgb_moe_weight = moe_weights[:, 0:1, :, :]
        ir_moe_weight = moe_weights[:, 1:2, :, :]
        event_moe_weight = moe_weights[:, 2:3, :, :] * event_weight
        rgb_ir_fusion_moe_weight = moe_weights[:, 3:4, :, :]
        tri_fusion_moe_weight = moe_weights[:, 4:5, :, :] * event_weight
        
        rgb_expert_out = self.rgb_expert(rgb_reduced).to(weight_dtype)
        ir_expert_out = self.ir_expert(ir_reduced).to(weight_dtype)
        event_expert_out = self.event_expert(event_reduced).to(weight_dtype)
        
        rgb_ir_concat = torch.cat([rgb_reduced, ir_reduced], dim=1)
        rgb_ir_expert_out = self.rgb_ir_expert(rgb_ir_concat).to(weight_dtype)
        
        rgb2ir_weights = self.rgb_to_ir_attn(rgb_reduced).to(weight_dtype)
        ir2rgb_weights = self.ir_to_rgb_attn(ir_reduced).to(weight_dtype)
        rgb2event_weights = self.rgb_to_event_attn(rgb_reduced).to(weight_dtype)
        event2rgb_weights = self.event_to_rgb_attn(event_reduced).to(weight_dtype)
        
        tri_rgb_guided = rgb_reduced + ir_reduced * ir2rgb_weights + event_reduced * event2rgb_weights * event_weight
        tri_ir_guided = ir_reduced + rgb_reduced * rgb2ir_weights + event_reduced * event2rgb_weights * event_weight
        tri_event_guided = event_reduced + rgb_reduced * rgb2event_weights
        
        tri_rgb_guided = tri_rgb_guided.to(weight_dtype)
        tri_ir_guided = tri_ir_guided.to(weight_dtype)
        tri_event_guided = tri_event_guided.to(weight_dtype)
        
        tri_rgb_enhanced = self.small_object_enhancer(tri_rgb_guided).to(weight_dtype)
        tri_ir_enhanced = self.small_object_enhancer(tri_ir_guided).to(weight_dtype)
        tri_event_enhanced = self.small_object_enhancer(tri_event_guided).to(weight_dtype)
        
        tri_context_input = torch.cat([
            tri_rgb_enhanced, 
            tri_ir_enhanced, 
            tri_event_enhanced * event_weight
        ], dim=1).to(weight_dtype)
        
        tri_context = self.global_context(tri_context_input).to(weight_dtype)
        
        tri_rgb_fusion = tri_rgb_enhanced * tri_context
        tri_ir_fusion = tri_ir_enhanced * tri_context
        tri_event_fusion = tri_event_enhanced * tri_context
        
        dual_rgb_guided = rgb_reduced + ir_reduced * ir2rgb_weights
        dual_ir_guided = ir_reduced + rgb_reduced * rgb2ir_weights
        
        dual_rgb_guided = dual_rgb_guided.to(weight_dtype)
        dual_ir_guided = dual_ir_guided.to(weight_dtype)
        
        dual_rgb_enhanced = self.small_object_enhancer(dual_rgb_guided).to(weight_dtype)
        dual_ir_enhanced = self.small_object_enhancer(dual_ir_guided).to(weight_dtype)
        
        dual_context_input = torch.cat([dual_rgb_enhanced, dual_ir_enhanced], dim=1).to(weight_dtype)
        dual_context = self.dual_global_context(dual_context_input).to(weight_dtype)
        
        dual_rgb_fusion = dual_rgb_enhanced * dual_context
        dual_ir_fusion = dual_ir_enhanced * dual_context
        
        fallback_input = torch.cat([rgb_fea, ir_fea], dim=1)
        fallback_weights = self.adaptive_moe_gate.fallback_gate(fallback_input).to(weight_dtype)
        
        fallback_rgb = fallback_weights[:, 0:1]
        fallback_ir = fallback_weights[:, 1:2]
        fallback_rgb_ir = fallback_weights[:, 2:3]
        
        rgb_tri_path = (rgb_expert_out * rgb_moe_weight + 
                        rgb_ir_expert_out * rgb_ir_fusion_moe_weight +
                        tri_rgb_fusion * tri_fusion_moe_weight).to(weight_dtype)
        
        rgb_dual_path = (rgb_expert_out * fallback_rgb + 
                         dual_rgb_fusion * fallback_rgb_ir).to(weight_dtype)
        
        rgb_combined = rgb_tri_path * event_weight + rgb_dual_path * dual_weight
        
        ir_tri_path = (ir_expert_out * ir_moe_weight + 
                       rgb_ir_expert_out * rgb_ir_fusion_moe_weight +
                       tri_ir_fusion * tri_fusion_moe_weight).to(weight_dtype)
        
        ir_dual_path = (ir_expert_out * fallback_ir + 
                        dual_ir_fusion * fallback_rgb_ir).to(weight_dtype)
        
        ir_combined = ir_tri_path * event_weight + ir_dual_path * dual_weight
        
        event_combined = (event_expert_out * event_moe_weight + 
                         tri_event_fusion * tri_fusion_moe_weight).to(weight_dtype)
        
        rgb_out = self.out_rgb(rgb_combined) + rgb_fea
        ir_out = self.out_ir(ir_combined) + ir_fea
        
        event_out_processed = self.out_event(event_combined)
        event_out = event_fea * dual_weight + (event_out_processed + event_fea) * event_weight
        
        rgb_out = rgb_out.to(weight_dtype)
        ir_out = ir_out.to(weight_dtype)
        event_out = event_out.to(weight_dtype)
        
        return [rgb_out, ir_out, event_out], event_reliability.to(weight_dtype)


class AdaptiveMobileViTFusion(nn.Module):
    def __init__(self, input_channels, output_channels, depth=2, reduction=4, patch_size=2, reliability_threshold=0.4):
        super().__init__()
        self.fusion = ImprovedTriModalLightweightMoECAFF(
            channels=output_channels,
            reduction=reduction,
            heads=4,
            reliability_threshold=reliability_threshold
        )
    
    def forward(self, x):
        outputs, reliability = self.fusion(x)
        return outputs
        
    def forward_with_reliability(self, x):
        return self.fusion(x)
