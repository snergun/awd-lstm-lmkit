import math

import torch
import torch.nn as nn
import wandb

from top_models.smax.smax import Smax

"""
Code courtesy: Ganea et al. 2019.

https://github.com/pytorch/pytorch/blob/35cdb785228b8abea1d3bdb844aa5980e6642f8d/tools/autograd/derivatives.yaml
Because of the above, plif smax is not reproducible for same seed when run twice.

Other helpful links:
https://github.com/pytorch/pytorch/blob/35cdb785228b8abea1d3bdb844aa5980e6642f8d/tools/autograd/derivatives.yaml
"""

class PlifSmax(Smax):

    # base_interval: defines the logits range on which the monotonic layer
    # will be applied
    def __init__(self, ntoken, nhidlast, K, T, w_variance):
        super(PlifSmax, self).__init__(ntoken, nhidlast)
        self.T = T
        self.K = K
        self.plif_w = nn.Parameter(
            torch.randn(self.K) * w_variance + math.log(math.exp(1) - 1)
        )

    # logits : size = num_ctxts * bs * num_vocab_words ,
    # i.e. <h,w> dot products
    def func(self, logits, plotting=False):
        if not plotting:
            self.sample_batch = logits.detach()  # Store a sample batch for visualization
        size = logits.size()
        logits = logits.view(-1)
        delta = 2. * self.T / self.K
        indices = torch.clamp(
            ((logits + self.T) / delta).detach().long(),
            max=self.K - 1, min=0
        )
        all_pos_w = nn.Softplus()(self.plif_w)
        all_pos_cumsum = torch.cumsum(all_pos_w, dim=-1) - all_pos_w
        pos_w = torch.gather(all_pos_w, -1, indices)
        # use gather, not take
        pos_w_cumsum = torch.gather(all_pos_cumsum, -1, indices)
        knots = (-self.T + delta * indices.float()).to(logits.device)
        result = (logits - knots) * pos_w + delta * pos_w_cumsum
        return result.view(size)

    def forward(self, input, extras):
        return super(PlifSmax, self).forward(input, extras)
    
    def get_logs(self):
        """Return dictionary of scalar and line plots for visualization (e.g., WandB)."""
        out = {}
        xmax = self.T
        xmin = -xmax
        if self.sample_batch is not None:
            batch_min = self.sample_batch.min().item()
            batch_max = self.sample_batch.max().item()
            xmin = min(xmin, batch_min - 5)
            xmax = max(xmax, batch_max + 5)
                
        # --- Generate function plot ---
        xs = torch.linspace(xmin, xmax, 500).to(self.plif_w.device)
        thresholds_out = None
        y = self.func(xs.unsqueeze(0), plotting=True) 
        y = y.squeeze()

        # --- Line plot ---
        xs_list = xs.squeeze().cpu().tolist()
        y_list = y.squeeze().cpu().tolist()
        out[f"plif_plot"] = wandb.plot.line_series(
            xs=xs_list,
            ys=[y_list],
            keys=["f(x)"],
            title=f"PLIF Plot",
            xname="x"
        )
        
        # # Histogram of logits in intervals
        # if self.sample_batch is not None:
        #     sample_logits = self.sample_batch
        #     delta = 2. * self.T / self.K
        #     bin_ids = torch.clamp(
        #         ((sample_logits + self.T) / delta).detach().long(),
        #         max=self.K - 1, min=0
        #     )            
        #     hist_counts = torch.bincount(bin_ids.flatten().cpu(), minlength=self.K)
        #     out[f"logits_hist"] = wandb.plot_table(
        #     vega_spec_name="ucsd-alon/logit_distribution",
        #     data_table=wandb.Table(data=[[j, count.item()] for j, count in enumerate(hist_counts)], columns=["bin", "count"]),
        #     fields={
        #     "x": "bin",
        #     "y": "count",
        #     "title": f"Logit Distribution"
        # }
        # )
        # Also plot the histogram of the sample batch itself
        if self.sample_batch is not None:
            n_samples = 10000
            flat_logits = self.sample_batch.view(-1)
            random_indices = torch.randperm(len(flat_logits))[:n_samples]
            sample_batch = flat_logits[random_indices]
            out[f"sample_batch_hist"] = wandb.plot.histogram(
                table= wandb.Table(data=sample_batch.unsqueeze(1).cpu().tolist(), columns=["logit"]),
                value='logit',
                title=f"Sample Batch Distribution"
            )
            out["logits_max"] = self.sample_batch.max().item()
            out["logits_min"] = self.sample_batch.min().item()
            out["logits_mean"] = self.sample_batch.mean().item()
            out["logits_std"] = self.sample_batch.std().item()

        return out
