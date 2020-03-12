import argparse
import math
import random
import os

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb

except ImportError:
    wandb = None

from model import Generator, Discriminator, Encoder
from dataset import MultiResolutionDataset, ImgDataset
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def transfer_loss(g_t_feats, content_feat, style_feats):
    loss_c = F.mse_loss(g_t_feats[-1], content_feat)

    loss_s = 0
    assert(len(g_t_feats) == len(style_feats))
    for i in range(len(g_t_feats)-1):
        style_mean, style_std = calc_mean_std(style_feats[i])
        target_mean, target_std = calc_mean_std(g_t_feats[i])

        loss_s += F.mse_loss(style_mean, target_mean) + F.mse_loss(style_std, target_std)

    return loss_c, loss_s


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


def train(args, loader, content_loader, style_loader,
          content_loader_path, style_loader_path,
          content_loader_sample, style_loader_sample,
          encoder, generator, discriminator,
          g_optim, d_optim, g_ema,
          device):
    loader = sample_data(loader)
    content_loader = sample_data(content_loader)
    style_loader = sample_data(style_loader)
    content_loader_path = sample_data(content_loader_path)
    style_loader_path = sample_data(style_loader_path)
    content_loader_sample = sample_data(content_loader_sample)
    style_loader_sample = sample_data(style_loader_sample)


    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module

    else:
        g_module = generator
        d_module = discriminator
        
    accum = 0.5 ** (32 / (10 * 1000))

    # TODO: Use real data
    # sample_z = torch.randn(args.n_sample, args.latent, device=device)
    # style_imgs_sample = torch.zeros(args.n_sample, 3, 256, 256).to(device)
    # content_imgs_sample = torch.zeros(args.n_sample, 3, 256, 256).to(device)
    style_imgs_sample = next(style_loader_sample).to(device)
    content_imgs_sample = next(content_loader_sample).to(device)
    utils.save_image(
        content_imgs_sample,
        f'sample/content.png',
        nrow=int(args.n_sample ** 0.5),
        normalize=True,
        range=(-1, 1),
    )
    utils.save_image(
        style_imgs_sample,
        f'sample/style.png',
        nrow=int(args.n_sample ** 0.5),
        normalize=True,
        range=(-1, 1),
    )

    _1, _2, sample_z, sample_content = encoder(style_imgs_sample, content_imgs_sample)

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print('Done!')

            break

        real_img = next(loader)
        real_img = real_img.to(device)

        requires_grad(generator, False)
        requires_grad(discriminator, True)

        # TODO: Use real data
        # style_imgs = torch.zeros(args.batch, 3, 256, 256).to(device)
        # content_imgs = torch.zeros(args.batch, 3, 256, 256).to(device)
        style_imgs = next(style_loader).to(device)
        content_imgs = next(content_loader).to(device)
        style_feats, content_feat, latent_code, content_code = encoder(style_imgs, content_imgs)
        # noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        fake_img, _ = generator([latent_code], content_code)
        fake_pred = discriminator(fake_img)

        real_pred = discriminator(real_img)
        d_loss = d_logistic_loss(real_pred, fake_pred)

        loss_dict['d'] = d_loss
        loss_dict['real_score'] = real_pred.mean()
        loss_dict['fake_score'] = fake_pred.mean()

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real_img.requires_grad = True
            real_pred = discriminator(real_img)
            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

            d_optim.step()

        loss_dict['r1'] = r1_loss

        requires_grad(generator, True)
        requires_grad(discriminator, False)

        # TODO: Use real data
        # style_imgs = torch.zeros(args.batch, 3, 256, 256).to(device)
        # content_imgs = torch.zeros(args.batch, 3, 256, 256).to(device)
        style_imgs = next(style_loader).to(device)
        content_imgs = next(content_loader).to(device)
        style_feats, content_feat, latent_code, content_code = encoder(style_imgs, content_imgs)
        # noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        fake_img, _ = generator([latent_code], content_code)
        fake_pred = discriminator(fake_img)
        g_loss = g_nonsaturating_loss(fake_pred)

        fake_style_feats, fake_content_feat, _1, _2 = encoder(fake_img, fake_img)
        c_loss, s_loss = transfer_loss(fake_style_feats, content_feat, style_feats)

        g_loss += args.content_weight * c_loss
        g_loss += args.style_weight * s_loss

        loss_dict['g'] = g_loss
        loss_dict['c'] = c_loss
        loss_dict['s'] = s_loss

        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        g_regularize = i % args.g_reg_every == 0

        if g_regularize:
            path_batch_size = max(1, args.batch // args.path_batch_shrink)
            # TODO: Use real data
            # style_imgs = torch.zeros(path_batch_size, 3, 256, 256).to(device)
            # content_imgs = torch.zeros(path_batch_size, 3, 256, 256).to(device)
            style_imgs = next(style_loader_path).to(device)
            content_imgs = next(content_loader_path).to(device)
            style_feats, content_feat, latent_code, content_code = encoder(style_imgs, content_imgs)
            #noise = mixing_noise(
            #    path_batch_size, args.latent, args.mixing, device
            #)
            fake_img, latents = generator([latent_code], content_code, return_latents=True)

            path_loss, mean_path_length, path_lengths = g_path_regularize(
                fake_img, latents, mean_path_length
            )

            generator.zero_grad()
            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

            if args.path_batch_shrink:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

            weighted_path_loss.backward()

            g_optim.step()

            mean_path_length_avg = (
                reduce_sum(mean_path_length).item() / get_world_size()
            )

        loss_dict['path'] = path_loss
        loss_dict['path_length'] = path_lengths.mean()

        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced['d'].mean().item()
        g_loss_val = loss_reduced['g'].mean().item()
        r1_val = loss_reduced['r1'].mean().item()
        path_loss_val = loss_reduced['path'].mean().item()
        real_score_val = loss_reduced['real_score'].mean().item()
        fake_score_val = loss_reduced['fake_score'].mean().item()
        path_length_val = loss_reduced['path_length'].mean().item()
        c_loss_val = loss_reduced['c'].mean().item()
        s_loss_val = loss_reduced['s'].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f'd: {d_loss_val:.4f}; g: {g_loss_val:.4f}; c: {c_loss_val}; s: {s_loss_val}; r1: {r1_val:.4f}; '
                    f'path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}'
                )
            )
            
            writer.add_scalar('d_loss_val', d_loss_val, i)
            writer.add_scalar('g_loss_val', g_loss_val, i)
            writer.add_scalar('c_loss_val', c_loss_val, i)
            writer.add_scalar('s_loss_val', s_loss_val, i)
            writer.add_scalar('path_loss_val', path_loss_val, i)
            writer.add_scalar('r1_val', r1_val, i)
            writer.add_scalar('real_score_val', real_score_val, i)
            writer.add_scalar('fake_score_val', fake_score_val, i)
            writer.add_scalar('path_length_val', path_length_val, i)

            if wandb and args.wandb:
                wandb.log(
                    {
                        'Generator': g_loss_val,
                        'Discriminator': d_loss_val,
                        'R1': r1_val,
                        'Path Length Regularization': path_loss_val,
                        'Mean Path Length': mean_path_length,
                        'Real Score': real_score_val,
                        'Fake Score': fake_score_val,
                        'Path Length': path_length_val,
                    }
                )

            if i % 1000 == 0:
                with torch.no_grad():
                    g_ema.eval()
                    sample, _ = g_ema([sample_z], sample_content)
                    utils.save_image(
                        sample,
                        f'sample/{str(i).zfill(6)}.png',
                        nrow=int(args.n_sample ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )

            if (i+1) % 5000 == 0:
                torch.save(
                    {
                        'g': g_module.state_dict(),
                        'd': d_module.state_dict(),
                        'g_ema': g_ema.state_dict(),
                        'g_optim': g_optim.state_dict(),
                        'd_optim': d_optim.state_dict(),
                    },
                    os.path.join(args.save_path, f'{str(i+1).zfill(6)}.pt'),
                )

def sample_transform():
    transform_list = [
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('path', type=str)
    parser.add_argument('--iter', type=int, default=800000)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--n_sample', type=int, default=64)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--r1', type=float, default=10)
    parser.add_argument('--path_regularize', type=float, default=2)
    parser.add_argument('--path_batch_shrink', type=int, default=2)
    parser.add_argument('--d_reg_every', type=int, default=16)
    parser.add_argument('--g_reg_every', type=int, default=4)
    parser.add_argument('--mixing', type=float, default=0.9)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--name', type=str, default='hope')
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--vgg_path', type=str, default="/home/huzy/models/vgg_normalised.pth")
    parser.add_argument('--content_weight', type=float, default=2.0)
    parser.add_argument('--style_weight', type=float, default=0.5)
    parser.add_argument('--content_sample_path', type=str, default="/home/huzy/datasets/COCO/sample/")
    parser.add_argument('--style_sample_path', type=str, default="/home/huzy/datasets/WikiArt/sample/")
    parser.add_argument('--content_train_path', type=str, default="/home/huzy/datasets/COCO/train/")
    parser.add_argument('--style_train_path', type=str, default="/home/huzy/datasets/WikiArt/train/")
    parser.add_argument('--content_path_path', type=str, default="/home/huzy/datasets/COCO/path/")
    parser.add_argument('--style_path_path', type=str, default="/home/huzy/datasets/WikiArt/path/")

    args = parser.parse_args()

    log_path = os.path.join('./log', args.name)
    if os.path.exists(log_path):
        os.removedirs(log_path)
    os.mkdir(log_path)
    writer = SummaryWriter(log_path)

    save_path = os.path.join('./checkpoint', args.name)
    if os.path.exists(save_path):
        os.removedirs(save_path)
    os.mkdir(save_path)
    args.save_path = save_path

    n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        synchronize()

    args.latent = 512
    args.n_mlp = 8
    args.init_feat_size = 32

    args.start_iter = 0

    encoder = Encoder(args.vgg_path, args.latent, args.init_feat_size).to(device)
    encoder.eval()
    generator = Generator(
        args.size, args.latent, args.n_mlp, args.init_feat_size, channel_multiplier=args.channel_multiplier
    ).to(device)
    discriminator = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema = Generator(
        args.size, args.latent, args.n_mlp, args.init_feat_size, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    if args.ckpt is not None:
        print('load model:', args.ckpt)
        
        ckpt = torch.load(args.ckpt)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])
            
        except ValueError:
            pass
            
        generator.load_state_dict(ckpt['g'])
        discriminator.load_state_dict(ckpt['d'])
        g_ema.load_state_dict(ckpt['g_ema'])

        g_optim.load_state_dict(ckpt['g_optim'])
        d_optim.load_state_dict(ckpt['d_optim'])

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )


    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    dataset = MultiResolutionDataset(args.path, transform, args.size)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )

    content_sample_dataset = ImgDataset(args.content_sample_path, sample_transform())
    style_sample_dataset = ImgDataset(args.style_sample_path, sample_transform())
    content_loader_sample = data.DataLoader(
        content_sample_dataset,
        batch_size=args.n_sample,
        sampler=data_sampler(content_sample_dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )
    style_loader_sample = data.DataLoader(
        style_sample_dataset,
        batch_size=args.n_sample,
        sampler=data_sampler(style_sample_dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )

    content_dataset = ImgDataset(args.content_train_path)
    style_dataset = ImgDataset(args.style_train_path)
    content_loader = data.DataLoader(
        content_dataset,
        batch_size=args.batch,
        sampler=data_sampler(content_dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )
    style_loader = data.DataLoader(
        style_dataset,
        batch_size=args.batch,
        sampler=data_sampler(style_dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )

    content_path_dataset = ImgDataset(args.content_path_path)
    style_path_dataset = ImgDataset(args.style_path_path)
    path_batch_size = max(1, args.batch // args.path_batch_shrink)
    content_loader_path = data.DataLoader(
        content_path_dataset,
        batch_size=path_batch_size,
        sampler=data_sampler(content_path_dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )
    style_loader_path = data.DataLoader(
        style_path_dataset,
        batch_size=path_batch_size,
        sampler=data_sampler(style_path_dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project='stylegan 2')
    train(args, loader, content_loader, style_loader,
          content_loader_path, style_loader_path,
          content_loader_sample, style_loader_sample,
          encoder, generator, discriminator,
          g_optim, d_optim, g_ema,
          device)
