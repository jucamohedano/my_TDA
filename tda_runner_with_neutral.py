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
    parser.add_argument('--config', dest='config', required=True, 
                       help='settings of TDA on specific dataset in yaml format.')
    parser.add_argument('--wandb-log', dest='wandb', action='store_true', 
                       help='Whether you want to log to wandb.')
    parser.add_argument('--datasets', dest='datasets', type=str, required=True,
                       help="Datasets to process, separated by a slash (/). Example: I/A/V/R/S")
    parser.add_argument('--data-root', dest='data_root', type=str, default='./dataset/',
                       help='Path to the datasets directory.')
    parser.add_argument('--backbone', dest='backbone', type=str, 
                       choices=['RN50', 'ViT-B/16'], required=True,
                       help='CLIP model backbone to use.')
    parser.add_argument('--use-neutral-cache', dest='use_neutral_cache', 
                       action='store_true', 
                       help='Use neutral cache for intermediate confidence predictions')
    return parser.parse_args()

def update_cache(cache, pred, features_loss, shot_capacity, include_prob_map=False):
    """Update cache with new features and loss, maintaining the maximum shot capacity."""
    with torch.no_grad():
        item = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]]
        cache_stats = {'hit': False}
        
        if pred in cache:
            if len(cache[pred]) < shot_capacity:
                cache[pred].append(item)
                cache_stats['hit'] = False
            elif features_loss[1] < cache[pred][-1][1]:
                cache[pred][-1] = item
                cache_stats['hit'] = True
            cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
        else:
            cache[pred] = [item]
            cache_stats['hit'] = False
        
        return cache_stats

def compute_cache_logits(image_features, cache, alpha, beta, clip_weights, neg_mask_thresholds=None):
    """Compute logits using cache."""
    with torch.no_grad():
        cache_keys = []
        cache_values = []
        cache_stats = {'hits': 0, 'total_queries': 0}
        
        for class_index in sorted(cache.keys()):
            for item in cache[class_index]:
                cache_keys.append(item[0])
                if neg_mask_thresholds:
                    cache_values.append(item[2])
                else:
                    cache_values.append(class_index)

        if not cache_keys:
            return torch.zeros_like(clip_weights[0]), cache_stats

        cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
        if neg_mask_thresholds:
            cache_values = torch.cat(cache_values, dim=0)
            cache_values = (((cache_values > neg_mask_thresholds[0]) & 
                           (cache_values < neg_mask_thresholds[1])).type(torch.int8)).cuda().half()
        else:
            cache_values = (F.one_hot(torch.Tensor(cache_values).to(torch.int64), 
                                    num_classes=clip_weights.size(1))).cuda().half()

        affinity = image_features @ cache_keys
        cache_stats['total_queries'] = affinity.size(0) * affinity.size(1)
        cache_stats['hits'] = torch.sum(affinity > 0.5).item()
        
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        return alpha * cache_logits, cache_stats

def run_test_tda(pos_cfg, neg_cfg, neutral_cfg, loader, clip_model, clip_weights):
    """Run test-time adaptation with three caches: positive, negative, and neutral."""
    with torch.no_grad():
        pos_cache, neg_cache, neutral_cache, accuracies = {}, {}, {}, []
        cache_stats = {
            'pos_cache_size': [], 'neg_cache_size': [], 'neutral_cache_size': [],
            'cache_update_times': [], 'inference_times': [],
            'pre_cache_accuracies': [], 'post_cache_accuracies': [],
            'memory_usage': [],
            'pos_cache_hits': [], 'pos_cache_misses': [],
            'neg_cache_hits': [], 'neg_cache_misses': [],
            'neutral_cache_hits': [], 'neutral_cache_misses': [],
            'pos_cache_hit_rates': [], 'neg_cache_hit_rates': [], 'neutral_cache_hit_rates': []
        }

        # Unpack hyperparameters
        pos_enabled = pos_cfg['enabled']
        neg_enabled = neg_cfg['enabled']
        neutral_enabled = neutral_cfg['enabled']

        if pos_enabled:
            pos_params = {k: pos_cfg[k] for k in ['shot_capacity', 'alpha', 'beta']}
        if neg_enabled:
            neg_params = {k: neg_cfg[k] for k in ['shot_capacity', 'alpha', 'beta', 
                                                 'entropy_threshold', 'mask_threshold']}
        if neutral_enabled:
            neutral_params = {k: neutral_cfg[k] for k in ['shot_capacity', 'alpha', 'beta', 
                                                        'entropy_threshold', 'weight']}

        for i, (images, target) in enumerate(tqdm(loader, desc='Processed test images: ')):
            start_time = time.time()
            
            image_features, clip_logits, loss, prob_map, pred = get_clip_logits(
                images, clip_model, clip_weights)
            target = target.cuda()
            prop_entropy = get_entropy(loss, clip_weights)
            
            # Track pre-cache accuracy
            pre_cache_acc = cls_acc(clip_logits, target)
            cache_stats['pre_cache_accuracies'].append(pre_cache_acc)

            cache_update_start = time.time()
            pos_update_stats = {'hit': False}
            neg_update_stats = {'hit': False}
            neutral_update_stats = {'hit': False}

            # Update caches based on entropy thresholds
            if prop_entropy < neutral_cfg['entropy_threshold']['lower'] and pos_enabled:
                pos_update_stats = update_cache(pos_cache, pred, [image_features, loss], 
                                             pos_params['shot_capacity'])
            elif prop_entropy > neutral_cfg['entropy_threshold']['upper'] and neg_enabled:
                neg_update_stats = update_cache(neg_cache, pred, [image_features, loss, prob_map], 
                                             neg_params['shot_capacity'], True)
            elif neutral_enabled:
                neutral_update_stats = update_cache(neutral_cache, pred, [image_features, loss], 
                                                 neutral_params['shot_capacity'])

            cache_update_time = time.time() - cache_update_start
            cache_stats['cache_update_times'].append(cache_update_time)

            # Compute final logits
            final_logits = clip_logits.clone()
            pos_query_stats = {'hits': 0, 'total_queries': 0}
            neg_query_stats = {'hits': 0, 'total_queries': 0}
            neutral_query_stats = {'hits': 0, 'total_queries': 0}
            
            if pos_enabled and pos_cache:
                pos_logits, pos_query_stats = compute_cache_logits(
                    image_features, pos_cache, pos_params['alpha'], pos_params['beta'], clip_weights)
                final_logits += pos_logits

            if neutral_enabled and neutral_cache:
                neutral_logits, neutral_query_stats = compute_cache_logits(
                    image_features, neutral_cache, neutral_params['alpha'], 
                    neutral_params['beta'], clip_weights)
                final_logits += neutral_logits * neutral_params['weight']

            if neg_enabled and neg_cache:
                neg_logits, neg_query_stats = compute_cache_logits(
                    image_features, neg_cache, neg_params['alpha'], neg_params['beta'], 
                    clip_weights, (neg_params['mask_threshold']['lower'], 
                                 neg_params['mask_threshold']['upper']))
                final_logits -= neg_logits

            # Update cache statistics
            for cache_type in ['pos', 'neg', 'neutral']:
                query_stats = locals()[f'{cache_type}_query_stats']
                cache_stats[f'{cache_type}_cache_hits'].append(query_stats['hits'])
                cache_stats[f'{cache_type}_cache_misses'].append(
                    query_stats['total_queries'] - query_stats['hits'])
                hit_rate = query_stats['hits'] / max(query_stats['total_queries'], 1)
                cache_stats[f'{cache_type}_cache_hit_rates'].append(hit_rate)

            # Track post-cache accuracy and inference time
            post_cache_acc = cls_acc(final_logits, target)
            inference_time = time.time() - start_time
            
            # Calculate memory usage
            total_memory = sum(
                sum(feat[0].nelement() * feat[0].element_size() for feat in items) 
                for cache in [pos_cache, neg_cache, neutral_cache] 
                for items in cache.values()
            ) / (1024 * 1024)  # Convert to MB
            
            # Update statistics
            cache_stats['post_cache_accuracies'].append(post_cache_acc)
            cache_stats['inference_times'].append(inference_time)
            cache_stats['pos_cache_size'].append(sum(len(v) for v in pos_cache.values()))
            cache_stats['neg_cache_size'].append(sum(len(v) for v in neg_cache.values()))
            cache_stats['neutral_cache_size'].append(sum(len(v) for v in neutral_cache.values()))
            cache_stats['memory_usage'].append(total_memory)
            
            accuracies.append(post_cache_acc)
            
            # Log metrics to wandb every 100 steps
            if i % 100 == 0:
                wandb.log({
                    "Averaged test accuracy": sum(accuracies)/len(accuracies),
                    "Pre-cache accuracy": pre_cache_acc,
                    "Post-cache accuracy": post_cache_acc,
                    "Cache update time (s)": cache_update_time,
                    "Total inference time (s)": inference_time,
                    "Positive cache size": cache_stats['pos_cache_size'][-1],
                    "Negative cache size": cache_stats['neg_cache_size'][-1],
                    "Neutral cache size": cache_stats['neutral_cache_size'][-1],
                    "Memory usage (MB)": total_memory,
                    "Positive cache hit rate": cache_stats['pos_cache_hit_rates'][-1],
                    "Negative cache hit rate": cache_stats['neg_cache_hit_rates'][-1],
                    "Neutral cache hit rate": cache_stats['neutral_cache_hit_rates'][-1],
                    "Samples processed": i
                })

            if i % 1000 == 0:
                print(f"---- TDA's test accuracy: {sum(accuracies)/len(accuracies):.2f}. "
                      f"Memory usage: {total_memory:.2f}MB ----")
                print(f"---- Cache hit rates - Pos: {cache_stats['pos_cache_hit_rates'][-1]:.2f}, "
                      f"Neg: {cache_stats['neg_cache_hit_rates'][-1]:.2f}, "
                      f"Neutral: {cache_stats['neutral_cache_hit_rates'][-1]:.2f} ----\n")

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
            
            if args.wandb:
                run = wandb.init(project="ETTA-CLIP", config=cfg, group=group_name)

            all_accs = []

            for loader_idx, (test_loader, classnames, template) in enumerate(test_loaders):
                corruption_idx = loader_idx // 5
                severity = (loader_idx % 5) + 1
                corruption_type = CIFAR10C_CORRUPTIONS[corruption_idx]
                
                print(f"\nTesting on corruption: {corruption_type}, severity: {severity}")
                clip_weights = clip_classifier(classnames, template, clip_model)

                if args.wandb and 'run' in locals():
                    run.finish()

                if args.wandb:
                    run_name = f"cifar10c_{corruption_type}_s{severity}_neutral"
                    run = wandb.init(project="ETTA-CLIP", config=cfg, 
                                   group=group_name, name=run_name)

                acc = run_test_tda(cfg['positive'], cfg['negative'], cfg['neutral'], 
                                 test_loader, clip_model, clip_weights)
                all_accs.append(acc)

                if args.wandb:
                    wandb.log({
                        f"cifar10c_{corruption_type}_s{severity}": acc,
                        "corruption_type": corruption_type,
                        "severity": severity
                    })

            if all_accs:
                avg_acc = sum(all_accs) / len(all_accs)
                print(f"\nAverage accuracy across all corruptions: {avg_acc:.2f}")
                if args.wandb:
                    wandb.log({"cifar10c_average": avg_acc})

            if args.wandb:
                run.finish()
        else:
            test_loader, classnames, template = build_test_data_loader(
                dataset_name, args.data_root, preprocess)
            clip_weights = clip_classifier(classnames, template, clip_model)

            if args.wandb:
                run_name = f"{dataset_name}_neutral"
                run = wandb.init(project="ETTA-CLIP", config=cfg, 
                               group=group_name, name=run_name)

            acc = run_test_tda(cfg['positive'], cfg['negative'], cfg['neutral'], 
                             test_loader, clip_model, clip_weights)

            if args.wandb:
                wandb.log({f"{dataset_name}": acc})
                run.finish()

if __name__ == "__main__":
    main()