import wandb
import pandas as pd
from plotting.accuracy_evolution import plot_accuracy_evolution
from plotting.cache_performance import plot_cache_metrics
from plotting.resource_performance import plot_resource_performance
from plotting.hyperparameter_impact import plot_hyperparameter_impact
from plotting.backbone_comparison import plot_backbone_comparison
from plotting.adaptation_progress import plot_adaptation_progress
from plotting.resource_tradeoff import plot_resource_tradeoff

# Get your W&B data
api = wandb.Api()
runs = api.runs("ETTA-CLIP")

# Separate data by backbone
rn50_runs = [run for run in runs if 'RN50' in run.config['backbone']]
vit_runs = [run for run in runs if 'ViT-B/16' in run.config['backbone']]

# Convert to DataFrames
rn50_data = pd.DataFrame([run.history() for run in rn50_runs])
vit_data = pd.DataFrame([run.history() for run in vit_runs])
wandb_data = pd.concat([rn50_data, vit_data])

# Create all visualizations
acc_plot = plot_accuracy_evolution(wandb_data)
acc_plot.savefig('accuracy_evolution.png')

cache_plot = plot_cache_metrics(wandb_data)
cache_plot.savefig('cache_metrics.png')

resource_plot = plot_resource_performance(wandb_data)
resource_plot.savefig('resource_performance.png')

backbone_plot = plot_backbone_comparison(rn50_data, vit_data)
backbone_plot.savefig('backbone_comparison.png')

adaptation_plot = plot_adaptation_progress(wandb_data)
adaptation_plot.savefig('adaptation_progress.png')

tradeoff_plot = plot_resource_tradeoff(wandb_data)
tradeoff_plot.savefig('resource_tradeoff.png')

# For sweep results
sweep_data = pd.DataFrame()  # Convert your sweep results to DataFrame
hyperparameter_plot = plot_hyperparameter_impact(sweep_data)
hyperparameter_plot.savefig('hyperparameter_impact.png') 