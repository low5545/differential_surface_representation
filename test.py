# Python std
import os
import time
import argparse
from timeit import default_timer as timer

# project files
import helpers
from model import AtlasNetReimpl

# 3rd party
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# neural atlas
import easydict
import tqdm
import yaml
import pytorch_lightning as pl
import neural_atlas as neat

from neural_atlas.data import samplers
from neural_atlas import loss_metric
from neural_atlas.utils import autograd

# --------------------------------------------------------------------------- #
class Test:
    def __init__(
        self,
        input,
        num_charts,
        metric,
        uv_space_scale,
        pcl_normalization_scale,
        eval_target_pcl_nml_size,
        model,
        device
    ):
        # save some attributes
        self.input_set = set(input)
        self.num_charts = num_charts
        self.chart_target_pcl_nml_size = easydict.EasyDict({
            "val": eval_target_pcl_nml_size // num_charts,
            "test": eval_target_pcl_nml_size // num_charts
        })
        self.model = model
        self.device = device

        # instantiate regular sampler and metric components
        UV_SAMPLE_DIMS = 2
        self.regular_sampler = samplers.RegularSampler(
            low=[ -uv_space_scale ] * UV_SAMPLE_DIMS,
            high=[ uv_space_scale ] * UV_SAMPLE_DIMS,
        )

        f_score_dist_threshold = metric.f_score_scale_ratio \
                                 * pcl_normalization_scale
        overlap_dist_threshold = metric.overlap_scale_ratio \
                                 * pcl_normalization_scale
        self.metric = loss_metric.metric.Metric(
            num_charts,
            uv_space_scale,
            eval_target_pcl_nml_size,
            metric.default_chamfer_dist,
            metric.default_normal,
            metric.default_f_score,
            metric.default_distortion,
            metric.default_stitching_err,
            f_score_dist_threshold,
            metric.distortion_eps,
            metric.degen_chart_area_ratio,
            overlap_dist_threshold
        )

    @torch.no_grad()
    def test_step(self, batch):
        stage = "test"
        batch = easydict.EasyDict(batch)
        batch.size = len(batch.dataset.target.pcl)

        # convert the unicode code point tensors of class & sample IDs to strs
        batch.dataset.common.class_id = self.unicode_code_pt_tensor_to_str(
            batch.dataset.common.class_id
        )
        batch.dataset.common.sample_id = self.unicode_code_pt_tensor_to_str(
            batch.dataset.common.sample_id
        )

        # infer the latent codes in batch
        batch.latent_code = self.infer_latent_code(batch)
        # remove redundant tensors to attempt to free-up memory
        if len(self.input_set) > 0:
            batch.dataset.pop("input")

        # obtain regular UV samples as true padded input UV samples
        # TODO: cache this, since this is a constant
        batch.chart_uv_sample_size = torch.tensor(
            self.chart_target_pcl_nml_size[stage], device=self.device
        )
        batch.chart_uv_sample_size_sqrt = (
            batch.chart_uv_sample_size.sqrt().ceil().to(torch.int64)
        )
        batch.chart_uv_sample_size = (
            batch.chart_uv_sample_size_sqrt.square()
        )

        batch.true_input_uv, batch.true_padded_input_uv = (                     # (num_charts, batch.chart_uv_sample_size, 2 / 3)
            self.sample_regular_uv(batch.chart_uv_sample_size_sqrt)
        )
        with torch.enable_grad():
            # enable grad to calc. distortion & normal consistency metrics
            batch.true_padded_input_uv = batch.true_padded_input_uv \
                                                .unsqueeze(dim=1)                 # (num_charts, 1, batch.chart_uv_sample_size, 3)
            batch.true_padded_input_uv = batch.true_padded_input_uv.expand(     # (num_charts, batch.size, batch.chart_uv_sample_size, 3)
                -1, batch.size, -1, -1
            )
        batch.pop("true_input_uv")                      # free-up memory

        # infer the mapped point cloud associated to the true padded input UV
        # samples
        with torch.enable_grad():
            # enable grad to calc. distortion & normal consistency metrics
            batch.mapped_pcl = self.infer_pcl(                                  # (num_charts, batch.size, batch.chart_uv_sample_size, 3)
                batch.latent_code.cond_homeomorphism,
                batch.true_padded_input_uv
            )

        # infer the tangent vectors associated to the mapped point cloud
        with torch.enable_grad():
            batch.mapped_tangent_vectors = self.infer_mapped_tangent_vectors(   # (num_charts, batch.size, batch.chart_uv_sample_size, 2, 3)
                batch.true_padded_input_uv, batch.mapped_pcl, stage=stage
            )
        batch.mapped_pcl = self.to_batch_size_num_charts_contiguous(            # (num_charts, batch.size, batch.chart_uv_sample_size, 3)
            batch.mapped_pcl
        )
        batch.mapped_tangent_vectors = (                                        # (num_charts, batch.size, batch.chart_uv_sample_size, 2, 3)
            self.to_batch_size_num_charts_contiguous(
                batch.mapped_tangent_vectors
            )
        )

        # compute the metric terms, for each sample in the batch
        if batch.size == 1 or self.device.type == "cpu":
            # can also be used for debugging
            batch.stream = [ None ] * batch.size
        elif self.device.type == "cuda":
            batch.stream = [ torch.cuda.Stream(self.device)
                                for _ in range(batch.size) ]
        batch.metric = self.metric.init_batch_metric(batch.size, self.device)
        batch.encoded_mesh = [ None ] * batch.size
        batch.mapped_mesh = [ None ] * batch.size
        batch.optimal_sim_scale = [ None ] * batch.size
        batch.optimal_area_scale = [ None ] * batch.size
        sample = easydict.EasyDict({})

        if batch.size > 1 and self.device.type == "cuda":
            for sample_stream in batch.stream:
                sample_stream.wait_stream(
                    torch.cuda.current_stream(self.device)
                )
        for sample.index in range(batch.size):
            # parallelize operations, if possible, for each sample in the batch
            sample.stream = batch.stream[sample.index]
            with torch.cuda.stream(sample.stream):
                # extract data that is associated to this sample from the batch
                sample.mapped_pcl = batch.mapped_pcl[:, sample.index, ...]      # (num_charts, batch.chart_uv_sample_size, 3)
                sample.mapped_tangent_vectors = (                               # (num_charts, batch.chart_uv_sample_size, 2, 3)
                    batch.mapped_tangent_vectors[:, sample.index, ...]
                )
                sample.dataset_target = easydict.EasyDict({})
                sample.dataset_target.pcl = (                                   # (eval_target_pcl_nml_size, 3)
                    batch.dataset.target.pcl[sample.index, ...]
                )
                # target surface normal is always given during evaluation
                sample.dataset_target.nml = (                                   # (eval_target_pcl_nml_size, 3)
                    batch.dataset.target.nml[sample.index, ...]
                )

                # compute the metric terms & save them in the batch
                sample.is_occupied = torch.ones(                                # (num_charts, batch.chart_uv_sample_size)
                    sample.mapped_pcl.shape[:2],
                    dtype=torch.bool,
                    device=sample.mapped_pcl.device
                )
                sample.metric, batch.encoded_mesh[sample.index], \
                batch.mapped_mesh[sample.index], \
                batch.optimal_sim_scale[sample.index], \
                batch.optimal_area_scale[sample.index] = self.metric.compute(
                    sample.is_occupied,
                    sample.mapped_pcl,
                    sample.dataset_target.pcl,
                    sample.mapped_tangent_vectors,
                    sample.dataset_target.nml
                )
                for metric_name, metric_value in sample.metric.items():
                    batch.metric[metric_name][sample.index] = metric_value
        if batch.size > 1 and self.device.type == "cuda":
            for sample_stream in batch.stream:
                torch.cuda.current_stream(self.device).wait_stream(
                    sample_stream
                )

        # average the metric terms over the samples of the batch
        batch.mean_metric = easydict.EasyDict({
            metric_name: sum(metric_values) / batch.size
            for metric_name, metric_values in batch.metric.items()
        })
        
        batch.std_mean_optimal_sim_scale = torch.std_mean(                      # 2-tuple of () tensor
            torch.as_tensor(batch.optimal_sim_scale), unbiased=False
        )
        batch.std_mean_optimal_area_scale = torch.std_mean(                     # 2-tuple of () tensor
            torch.as_tensor(batch.optimal_area_scale), unbiased=False
        )

        batch.mean_metric.std_optimal_sim_scale = batch.std_mean_optimal_sim_scale[0]
        batch.mean_metric.mean_optimal_sim_scale = batch.std_mean_optimal_sim_scale[1]
        batch.mean_metric.std_optimal_area_scale = batch.std_mean_optimal_area_scale[0]
        batch.mean_metric.mean_optimal_area_scale = batch.std_mean_optimal_area_scale[1]
        
        return batch.mean_metric

    @staticmethod
    def unicode_code_pt_tensor_to_str(batch_unicode_code_pt_tensor):            # (batch.size, S)
        # for each sample in the batch, convert all Unicode code point integers
        # to characters, then concatenate them to form a string & finally
        # remove any trailing spaces used for padding 
        batch_str = [
            "".join(map(chr, sample_unicode_code_pt_tensor)).rstrip()
            for sample_unicode_code_pt_tensor in batch_unicode_code_pt_tensor
        ]
        return batch_str

    def preprocess_input_uv(self, batch_input_uv, batch_size):
        # remove the extra batch dimension of 1 on the input UV samples
        batch_input_uv = batch_input_uv.squeeze(dim=0)                          # (train/val/test_batch_size, train/eval_uv_max_sample_size, 2)
        
        # match the batch size of the input UV samples to other inputs, which
        # might be different for the last batch during evaluation
        batch_input_uv = batch_input_uv[:batch_size, ...]                       # (batch_size, train/eval_uv_max_sample_size, 2)
        
        # virtually split the input UV samples evenly to each chart
        batch_input_uv = batch_input_uv.view(                                   # (num_charts, batch.size, train/eval_uv_max_sample_size // num_charts, 2)
            self.num_charts, batch_size, -1, 2
        )

        # pad zeros to the input UV samples as (last) W-coordinates
        batch_padded_input_uv = torch.nn.functional.pad(batch_input_uv, (0, 1)) # (num_charts, batch.size, train/eval_uv_max_sample_size // num_charts, 3)

        # require gradients for the padded input UV samples in order to compute
        # gradients of the conditional SDFs & homeomorphisms
        batch_padded_input_uv.requires_grad_()

        # set input UV samples as a view of its padded variant to save memory
        batch_input_uv = batch_padded_input_uv[..., :2]                         # (num_charts, batch.size, train/eval_uv_max_sample_size // num_charts, 2)

        return batch_input_uv, batch_padded_input_uv

    def infer_latent_code(self, batch):
        latent_code = easydict.EasyDict({})

        # infer the latent code common to all latent code reduction layers, or
        # conditional homeomorphisms, if no latent reduction is required
        if self.input_set == { "pcl" }:
            dataset_input_data = batch.dataset.input.pcl                        # (batch.size, input_pcl_nml_size, 3)
        elif self.input_set == { "img" }:
            dataset_input_data = batch.dataset.input.img                        # (batch.size, 3, img_size, img_size)
        latent_code.common = self.model.enc(dataset_input_data)                 # (batch.size, latent_dims)

        # infer chart-specific latent codes for conditional homeomorphism
        latent_code.cond_homeomorphism = latent_code.common \
                                                    .unsqueeze(dim=0)           # (1, batch.size, latent_dims)
        latent_code.cond_homeomorphism = (                                      # (num_charts, batch.size, latent_dims)
            latent_code.cond_homeomorphism.expand(
                self.num_charts, -1, -1
            )
        )
        return latent_code

    @staticmethod
    def to_batch_size_num_charts_contiguous(input):                             # (num_charts, batch.size, ...)
        # return a copy of the input tensor that is contiguous in the shape
        # (batch.size, num_charts, ...)
        input = input.transpose(0, 1).clone(                                    # (batch.size, num_charts, ...)  
            memory_format=torch.contiguous_format
        )
        input = input.transpose(0, 1)                                           # (num_charts, batch.size, ...)
        return input

    def infer_pcl(
        self,
        latent_code_cond_homeomorphism,
        input_pcl
    ):
        expanded_shape = list(                                                  # ie. ([batch.size,] 1, latent_dims)
            latent_code_cond_homeomorphism[0, ...].unsqueeze(-2).shape
        )             
        expanded_shape[-2] = input_pcl.shape[-2]                                # ie. [batch.size,] P, latent_dims

        return torch.stack([                                                    # (num_charts, [batch.size,] P, 3)
            chart_cond_homeomorphism(                                           # ([batch.size,] P, 3)
                torch.cat(                                                          # ([batch.size,] P, 2 + latent_dims)
                    (chart_input_pcl[..., :-1],                                     # ([batch.size,] P, 2)
                     chart_cond_homeomorphism_latent_code.unsqueeze(-2)
                                                         .expand(*expanded_shape)), # ([batch.size,] P, latent_dims)
                    dim=-1
                )
            )
            for chart_cond_homeomorphism,
                chart_cond_homeomorphism_latent_code,                           # ([batch.size,] latent_dims)
                chart_input_pcl                                                 # ([batch.size,] P, 3)
            in zip(self.model.dec,
                   latent_code_cond_homeomorphism,                              # (num_charts, [batch.size,] latent_dims)
                   input_pcl)                                                   # (num_charts, [batch.size,] P, 3)
        ], dim=0)

    def infer_mapped_tangent_vectors(
        self,
        true_padded_input_uv,                                                   # (num_charts, [batch.size,] batch.chart_uv_sample_size, 3)
        mapped_pcl,                                                             # (num_charts, [batch.size,] batch.chart_uv_sample_size, 3)
        stage
    ):
        # create the computational graph for the tangent vectors only in train
        create_graph = {
            "train": True,
            "val": False,
            "test": False
        }[stage]

        # `jacobian[*indices, :, :]` is the 3x3 transposed Jacobian matrix of
        # the cond homeomorphism with UVW-coordinate inputs & XYZ-coordinate 
        # outputs at `true_padded_input_uv[*indices, :]`
        jacobian = autograd.jacobian(                                           # (num_charts, [batch.size,] batch.chart_uv_sample_size, 3, 3)
            output=mapped_pcl,
            inputs=true_padded_input_uv,
            create_graph=create_graph
        )

        # 1. `mapped_tangent_vectors[*indices, 0, :]` is the gradient of XYZ 
        #    wrt. U & `mapped_tangent_vectors[*indices, 1, :]` is the gradient
        #    of XYZ wrt. V, at `padded_input_uv[*indices, :]`
        # 2. `mapped_tangent_vectors[*indices, 0, :]` & 
        #    `mapped_tangent_vectors[*indices, 1, :]` are the tangent vectors
        #    (not normalized to unit vectors) at `mapped_pcl[*indices, :]`
        mapped_tangent_vectors = jacobian[..., :2, :]                           # (num_charts, [batch.size,] batch.chart_uv_sample_size, 2, 3)

        return mapped_tangent_vectors

    def sample_regular_uv(self, sample_chart_uv_sample_size_sqrt):
        size = ( sample_chart_uv_sample_size_sqrt, ) * 2
        regular_uv = self.regular_sampler(size, self.device)                    # (sample_chart_uv_sample_size_sqrt, sample_chart_uv_sample_size_sqrt, 2)

        # pad zeros to the regular UV samples as (last) W-coordinates
        padded_regular_uv = torch.nn.functional.pad(regular_uv, (0, 1))         # (sample_chart_uv_sample_size_sqrt, sample_chart_uv_sample_size_sqrt, 3)

        # linearize & expand the padded regular UV samples across a new
        # chart dimension
        padded_regular_uv = padded_regular_uv.view(1, -1, 3)                    # (1, sample.chart_uv_sample_size, 3)
        padded_regular_uv = padded_regular_uv.expand(                           # (num_charts, sample.chart_uv_sample_size, 3)
            self.num_charts, -1, -1
        )

        # require gradients for the padded regular UV samples in order to
        # compute gradients of the conditional SDFs & homeomorphisms
        padded_regular_uv.requires_grad_()

        # set regular UV samples as a view of its padded variant to save memory
        regular_uv = padded_regular_uv[..., :2]                                 # (num_charts, sample.chart_uv_sample_size, 2)

        return regular_uv, padded_regular_uv

# --------------------------------------------------------------------------- #

# Settings.
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
conf['batch_size'] = neat_config.data.test_eff_batch_size
svr = ("img" in neat_config.input)

# Prepare TB writers.
writer_test = SummaryWriter(helpers.jn(args.output, 'test'))

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
    alphas_sciso=conf['alphas_sciso'],
    gpu=gpu,
    svr=svr,
    checkpoint_filepath=conf['checkpoint_filepath'],
    load_encoder=conf['load_encoder'],
    load_decoder=conf['load_decoder'],
    freeze_decoder=conf['freeze_decoder']
)

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
datamodule.setup(stage="test")

ds_test = datamodule.test_dataset
dl_test = datamodule.test_dataloader().loaders['dataset']

# define the device
device = helpers.Device(gpu=gpu).device

print('Valid ds: {} samples'.format(len(ds_test)))

iters_test = int(np.ceil(len(ds_test) / float(conf['batch_size'])))
losses_test = helpers.RunningLoss()

# Validation.
test = Test(
    input=neat_config.input,
    num_charts=conf['num_patches'],
    metric=neat_config.metric,
    uv_space_scale=neat_config.data.uv_space_scale,
    pcl_normalization_scale=neat_config.data.uv_space_scale,
    eval_target_pcl_nml_size=neat_config.data.eval_target_pcl_nml_size,
    model=model,
    device=device
)

model.eval()
it = 0
final_metric = easydict.EasyDict({})

for bi, batch in enumerate(tqdm.tqdm(dl_test)):
    batch = easydict.EasyDict({
        "dataset": batch
    })

    # send tensors to GPU
    if svr:
        batch.dataset.input.img = batch.dataset.input.img.to(device)
    else:
        batch.dataset.input.pcl = batch.dataset.input.pcl.to(device)
    batch.dataset.target.pcl = batch.dataset.target.pcl.to(device)
    batch.dataset.target.nml = batch.dataset.target.nml.to(device)
    batch.dataset.target.area = batch.dataset.target.area.to(device)

    mean_metric = test.test_step(batch)

    batch_size = batch.dataset.target.pcl.shape[0]
    for metric_name, mean_metric_value in mean_metric.items():
        if bi == 0:
            final_metric[metric_name] = 0.0
        final_metric[metric_name] += mean_metric_value.item() * batch_size

for metric_name, final_metric_value in final_metric.items():
    final_metric[metric_name] = final_metric_value / len(ds_test)
    writer_test.add_scalar(f'test/{metric_name}', final_metric[metric_name], it)
    print(f'test/{metric_name}: {final_metric[metric_name]}')
    time.sleep(0.01)    # delay a bit to allow the logs to be flushed

# save the test metrics to the log dir
METRICS_FILENAME = "metrics.yaml"
metrics_filepath = os.path.join(args.output, METRICS_FILENAME)
with open(metrics_filepath, 'w') as f:
    yaml.dump(dict(final_metric), f)
