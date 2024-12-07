import random
import argparse
import wandb
from tqdm import tqdm
from datetime import datetime
import time
import numpy as np

import torch
import torch.nn.functional as F
import operator

import clip
from utils import *


def get_arguments():
    """Get arguments of the test-time adaptation."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', required=True, help='settings of TDA on specific dataset in yaml format.')
    parser.add_argument('--wandb-log', dest='wandb', action='store_true', help='Whether you want to log to wandb. Include this flag to enable logging.')
    parser.add_argument('--datasets', dest='datasets', type=str, required=True, help="Datasets to process, separated by a slash (/). Example: I/A/V/R/S")
    parser.add_argument('--data-root', dest='data_root', type=str, default='./dataset/', help='Path to the datasets directory. Default is ./dataset/')
    parser.add_argument('--backbone', dest='backbone', type=str, choices=['RN50', 'ViT-B/16'], required=True, help='CLIP model backbone to use: RN50 or ViT-B/16.')

    args = parser.parse_args()

    return args


def update_cache(cache, pred, features_loss, shot_capacity, include_prob_map=False):
    """Update cache with new features and loss, maintaining the maximum shot capacity."""
    with torch.no_grad():
        item = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]]
        if pred in cache:
            if len(cache[pred]) < shot_capacity:
                cache[pred].append(item)
            elif features_loss[1] < cache[pred][-1][1]:
                cache[pred][-1] = item
            cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
        else:
            cache[pred] = [item]


def compute_cache_logits(image_features, cache, alpha, beta, clip_weights, neg_mask_thresholds=None):
    """Compute logits using positive/negative cache."""
    with torch.no_grad():
        cache_keys = []
        cache_values = []
        for class_index in sorted(cache.keys()):
            for item in cache[class_index]:
                cache_keys.append(item[0])
                if neg_mask_thresholds:
                    cache_values.append(item[2])
                else:
                    cache_values.append(class_index)

        cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
        if neg_mask_thresholds:
            cache_values = torch.cat(cache_values, dim=0)
            cache_values = (((cache_values > neg_mask_thresholds[0]) & (cache_values < neg_mask_thresholds[1])).type(torch.int8)).cuda().half()
        else:
            cache_values = (F.one_hot(torch.Tensor(cache_values).to(torch.int64), num_classes=clip_weights.size(1))).cuda().half()

        affinity = image_features @ cache_keys
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        return alpha * cache_logits

def run_test_tda(pos_cfg, neg_cfg, loader, clip_model, clip_weights):
    with torch.no_grad():
        pos_cache, neg_cache, accuracies = {}, {}, []
        cache_stats = {
            'pos_cache_size': [],
            'neg_cache_size': [],
            'cache_update_times': [],
            'inference_times': [],
            'pre_cache_accuracies': [],
            'post_cache_accuracies': []
        }
        
        #Unpack all hyperparameters
        pos_enabled, neg_enabled = pos_cfg['enabled'], neg_cfg['enabled']
        if pos_enabled:
            pos_params = {k: pos_cfg[k] for k in ['shot_capacity', 'alpha', 'beta']}
        if neg_enabled:
            neg_params = {k: neg_cfg[k] for k in ['shot_capacity', 'alpha', 'beta', 'entropy_threshold', 'mask_threshold']}

        #Test-time adaptation
        for i, (images, target) in enumerate(tqdm(loader, desc='Processed test images: ')):
            start_time = time.time()
            image_features, clip_logits, loss, prob_map, pred = get_clip_logits(images ,clip_model, clip_weights)
            target, prop_entropy = target.cuda(), get_entropy(loss, clip_weights)
            
            # Track pre-cache accuracy
            pre_cache_acc = cls_acc(clip_logits, target)
            cache_stats['pre_cache_accuracies'].append(pre_cache_acc)

            cache_update_start = time.time()
            if pos_enabled:
                update_cache(pos_cache, pred, [image_features, loss], pos_params['shot_capacity'])

            if neg_enabled and neg_params['entropy_threshold']['lower'] < prop_entropy < neg_params['entropy_threshold']['upper']:
                update_cache(neg_cache, pred, [image_features, loss, prob_map], neg_params['shot_capacity'], True)
            
            cache_update_time = time.time() - cache_update_start
            cache_stats['cache_update_times'].append(cache_update_time)

            final_logits = clip_logits.clone()
            if pos_enabled and pos_cache:
                final_logits += compute_cache_logits(image_features, pos_cache, pos_params['alpha'], pos_params['beta'], clip_weights)
            if neg_enabled and neg_cache:
                final_logits -= compute_cache_logits(image_features, neg_cache, neg_params['alpha'], neg_params['beta'], clip_weights, (neg_params['mask_threshold']['lower'], neg_params['mask_threshold']['upper']))

            # Track post-cache accuracy and inference time
            post_cache_acc = cls_acc(final_logits, target)
            inference_time = time.time() - start_time
            
            cache_stats['post_cache_accuracies'].append(post_cache_acc)
            cache_stats['inference_times'].append(inference_time)
            cache_stats['pos_cache_size'].append(sum(len(v) for v in pos_cache.values()))
            cache_stats['neg_cache_size'].append(sum(len(v) for v in neg_cache.values()))
            
            accuracies.append(post_cache_acc)
            
            # Log metrics to wandb
            if i % 100 == 0:  # Log every 100 steps to avoid overwhelming wandb
                wandb.log({
                    "Averaged test accuracy": sum(accuracies)/len(accuracies),
                    "Pre-cache accuracy": pre_cache_acc,
                    "Post-cache accuracy": post_cache_acc,
                    "Cache update time (s)": cache_update_time,
                    "Total inference time (s)": inference_time,
                    "Positive cache size": cache_stats['pos_cache_size'][-1],
                    "Negative cache size": cache_stats['neg_cache_size'][-1],
                    "Samples processed": i
                }, commit=True)

            if i%1000==0:
                print("---- TDA's test accuracy: {:.2f}. ----\n".format(sum(accuracies)/len(accuracies)))
        
        # Log final statistics
        wandb.log({
            "Final test accuracy": sum(accuracies)/len(accuracies),
            "Average cache update time": np.mean(cache_stats['cache_update_times']),
            "Average inference time": np.mean(cache_stats['inference_times']),
            "Average pre-cache accuracy": np.mean(cache_stats['pre_cache_accuracies']),
            "Average post-cache accuracy": np.mean(cache_stats['post_cache_accuracies']),
            "Final positive cache size": cache_stats['pos_cache_size'][-1],
            "Final negative cache size": cache_stats['neg_cache_size'][-1]
        })
        
        return sum(accuracies)/len(accuracies)



def main():
    args = get_arguments()
    config_path = args.config

    # Initialize CLIP model
    clip_model, preprocess = clip.load(args.backbone)
    clip_model.eval()

    # Set random seed
    random.seed(1)
    torch.manual_seed(1)

    if args.wandb:
        date = datetime.now().strftime("%b%d_%H-%M-%S")
        group_name = f"{args.backbone}_{args.datasets}_{date}"
    
    # Run TDA on each dataset
    datasets = args.datasets.split('/')
    for dataset_name in datasets:
        print(f"Processing {dataset_name} dataset.")
        
        cfg = get_config_file(config_path, dataset_name)
        print("\nRunning dataset configurations:")
        print(cfg, "\n")
        
        if dataset_name == 'C':
            # Special handling for CIFAR-10-C
            test_loaders = build_test_data_loader(dataset_name, args.data_root, preprocess)
            all_accs = []
            
            # Initialize W&B once before the loop
            if args.wandb:
                run = wandb.init(project="ETTA-CLIP", config=cfg, group=group_name)

            all_accs = []  # Initialize list to store accuracies for average calculation

            for loader_idx, (test_loader, classnames, template) in enumerate(test_loaders):
                # Calculate corruption type and severity
                corruption_idx = loader_idx // 5  # Integer division to get corruption index
                severity = (loader_idx % 5) + 1   # Remainder + 1 to get severity level
                corruption_type = CIFAR10C_CORRUPTIONS[corruption_idx]
                
                print(f"\nTesting on corruption: {corruption_type}, severity: {severity}")
                clip_weights = clip_classifier(classnames, template, clip_model)

                # Finish the previous run if it exists
                if args.wandb and 'run' in locals():
                    run.finish()

                # Add run name for W&B
                if args.wandb:
                    run_name = f"cifar10c_{corruption_type}_s{severity}_2"
                    run = wandb.init(project="ETTA-CLIP", config=cfg, group=group_name, name=run_name)

                acc = run_test_tda(cfg['positive'], cfg['negative'], test_loader, clip_model, clip_weights)
                all_accs.append(acc)

                if args.wandb:
                    wandb.log({
                        f"cifar10c_{corruption_type}_s{severity}_2": acc,
                        "corruption_type": corruption_type,
                        "severity": severity
                    })

            # Calculate and log average accuracy across all corruptions
            if all_accs:  # Ensure there are accuracies to average
                avg_acc = sum(all_accs) / len(all_accs)
                print(f"\nAverage accuracy across all corruptions: {avg_acc:.2f}")
                if args.wandb:
                    wandb.log({"cifar10c_average": avg_acc})

            # Finish the W&B run after the loop
            if args.wandb:
                run.finish()
        else:
            test_loader, classnames, template = build_test_data_loader(dataset_name, args.data_root, preprocess)
            clip_weights = clip_classifier(classnames, template, clip_model)

            if args.wandb:
                run_name = f"{dataset_name}"
                run = wandb.init(project="ETTA-CLIP", config=cfg, group=group_name, name=run_name)

            acc = run_test_tda(cfg['positive'], cfg['negative'], test_loader, clip_model, clip_weights)

            if args.wandb:
                wandb.log({f"{dataset_name}": acc})
                run.finish()

if __name__ == "__main__":
    main()