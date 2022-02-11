""" Training of AtlasNet [1] with differential geometry properties based
    regularizers [2] for point-cloud auto-encoding task on ShapeNet.

[1] Groueix Thibault et al. AtlasNet: A Papier-Mâché Approach to Learning 3D
    Surface Generation. CVPR 2018.
[2] Bednarik Jan et al. Shape Reconstruction by Learning Differentiable Surface
    Representations. CoRR 2019.

Author: Jan Bednarik, jan.bednarik@epfl.ch
"""

# Python std
import argparse
from timeit import default_timer as timer

# project files
import helpers
from model import AtlasNetReimpl
from data_loader import ShapeNet, DataLoaderDevice

# 3rd party
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# neural atlas
import easydict
import yaml
import pytorch_lightning as pl
import neural_atlas as neat

# Settings.
print_loss_tr_every = 50
print_loss_val_every = 10
save_collapsed_every = 50
gpu = torch.cuda.is_available()

# Parse arguments.
parser = argparse.ArgumentParser()
parser.add_argument(
    "neat_config", type=str, help='Path to the neural atlas config file.'
)
parser.add_argument('--conf', help='Path to the main config file of the model.',
                    default='config.yaml')
parser.add_argument('--output', help='Path to the output directory for storing '
                                     'weights and tensorboard data.',
                    default='config.yaml')
args = parser.parse_args()

# Load the config file, prepare paths.
conf = helpers.load_conf(args.conf)

# load the neural atlas config from the config file
with open(args.neat_config) as f:
    neat_config = easydict.EasyDict(yaml.full_load(f))

# set configs from the neural atlas config
assert neat_config.data.train_eff_batch_size \
       == neat_config.data.val_eff_batch_size
conf['batch_size'] = neat_config.data.train_eff_batch_size

# Prepare TB writers.
writer_tr = SummaryWriter(helpers.jn(args.output, 'tr'))
writer_va = SummaryWriter(helpers.jn(args.output, 'va'))

# Build a model.
model = AtlasNetReimpl(
    M=conf['M'], code=conf['code'], num_patches=conf['num_patches'],
    normalize_cw=conf['normalize_cw'],
    freeze_encoder=conf['enc_freeze'],
    enc_load_weights=conf['enc_weights'],
    dec_activ_fns=conf['dec_activ_fns'],
    dec_use_tanh=conf['dec_use_tanh'],
    dec_batch_norm=conf['dec_batch_norm'],
    loss_scaled_isometry=conf['loss_scaled_isometry'],
    alpha_scaled_isometry=conf['alpha_scaled_isometry'],
    alphas_sciso=conf['alphas_sciso'], gpu=gpu)

# seed all pseudo-random generators
neat_config.seed = pl.seed_everything(neat_config.seed, workers=True)

# instantiate the neural atlas data module
datamodule = neat.data.datamodule.DataModule(
    neat_config.seed,
    neat_config.input,
    neat_config.target,
    num_nodes=1,
    gpus=[ 0 ],
    **neat_config.data
)
datamodule.setup(stage="fit")

ds_tr = datamodule.train_dataset
ds_va = datamodule.val_dataset
dl_tr = datamodule.train_dataloader()['dataset']
dl_va = datamodule.val_dataloader().loaders['dataset']

# define the device
device = helpers.Device(gpu=gpu).device

print('Train ds: {} samples'.format(len(ds_tr)))
print('Valid ds: {} samples'.format(len(ds_va)))

# Prepare training.
opt = torch.optim.Adam(model.parameters(), lr=conf['lr'])
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    opt, milestones=conf['milestones'], gamma=conf['gamma']
)

# Prepare savers.
saver = helpers.TrainStateSaver(
    helpers.jn(args.output, 'chkpt.tar'), model=model, optimizer=opt,
    scheduler=scheduler)

# Training loop.
iters_tr = int(np.ceil(len(ds_tr) / float(conf['batch_size'])))
iters_va = int(np.ceil(len(ds_va) / float(conf['batch_size'])))
losses_tr = helpers.RunningLoss()
losses_va = helpers.RunningLoss()
for ep in range(1, conf['epochs'] + 1):
    # Training.
    tstart = timer()
    model.train()
    for bi, batch in enumerate(dl_tr, start=1):
        batch = easydict.EasyDict(batch)

        it = (ep - 1) * iters_tr + bi
        model(batch.input.pcl.to(device), it=it)
        losses = model.loss(batch.target.pcl.to(device), areas_gt=batch.target.area.to(device))

        opt.zero_grad()
        losses['loss_tot'].backward()
        opt.step()

        losses_tr.update(**{k: v.item() for k, v in losses.items()})
        if bi % print_loss_tr_every == 0:
            losses_avg = losses_tr.get_losses()
            for k, v in losses_avg.items():
                writer_tr.add_scalar(k, v, it)
            losses_tr.reset()
            writer_tr.add_scalar('lr', opt.param_groups[0]['lr'], it)

            strh = '\rep {}/{}, it {}/{}, {:.0f} s - '.\
                format(ep, conf['epochs'], bi, iters_tr, timer() - tstart)
            strl = ', '.join(['{}: {:.4f}'.format(k, v)
                              for k, v in losses_avg.items()])
            print(strh + strl, end='')

        # Save number of collapsed patches.
        if bi % save_collapsed_every == 0 and 'fff' in model.geom_props:
            num_collpased = np.sum(
                [inds.shape[0] for inds in model.collapsed_patches_A()]) / \
                            model.pc_pred.shape[0]
            writer_tr.add_scalar('collapsed_patches', num_collpased,
                                 global_step=it)

    # Validation.
    if ep % print_loss_val_every == 0:
        model.eval()
        it = ep * iters_tr
        loss_va_run = 0.
        for bi, batch in enumerate(dl_va):
            batch = easydict.EasyDict(batch)

            curr_bs = batch.input.pcl.shape[0]
            model(batch.input.pcl.to(device))
            lv = model.loss(batch.target.pcl.to(device), areas_gt=batch.target.area.to(device))['loss_tot']
            loss_va_run += lv.item() * curr_bs

            # Save number of collapsed patches.
            if bi == 1 and 'fff' in model.geom_props:
                num_collpased = np.sum(
                    [inds.shape[0] for inds in model.collapsed_patches_A()]) / \
                                model.pc_pred.shape[0]
                writer_va.add_scalar('collapsed_patches', num_collpased,
                                    global_step=it)

        loss_va = loss_va_run / len(ds_va)
        writer_va.add_scalar('loss_tot', loss_va, it)
        print(' ltot_va: {:.4f}'.format(loss_va))

    # LR scheduler.
    scheduler.step()

    # Save train state.
    saver(epoch=ep, iterations=it)
