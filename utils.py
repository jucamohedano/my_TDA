import os
import yaml
import torch
import math
import numpy as np
import clip
from datasets.imagenet import ImageNet
from datasets import build_dataset
from datasets.utils import build_data_loader, AugMixAugmenter
import torchvision.transforms as transforms
from PIL import Image

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

# Add CIFAR-10-C corruptions list
CIFAR10C_CORRUPTIONS = [
    'brightness', 'contrast', 'defocus_blur', 'elastic_transform',
    'fog', 'frost', 'gaussian_noise', 'glass_blur', 'impulse_noise',
    'jpeg_compression', 'motion_blur', 'pixelate', 'shot_noise',
    'snow', 'zoom_blur'
]

def get_entropy(loss, clip_weights):
    max_entropy = math.log2(clip_weights.size(1))
    return float(loss / max_entropy)


def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def clip_classifier(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights


def get_clip_logits(images, clip_model, clip_weights):
    with torch.no_grad():
        if isinstance(images, list):
            images = torch.cat(images, dim=0).cuda()
        else:
            images = images.cuda()

        image_features = clip_model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        clip_logits = 100. * image_features @ clip_weights

        if image_features.size(0) > 1:
            batch_entropy = softmax_entropy(clip_logits)
            selected_idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * 0.1)]
            output = clip_logits[selected_idx]
            image_features = image_features[selected_idx].mean(0).unsqueeze(0)
            clip_logits = output.mean(0).unsqueeze(0)

            loss = avg_entropy(output)
            prob_map = output.softmax(1).mean(0).unsqueeze(0)
            pred = int(output.mean(0).unsqueeze(0).topk(1, 1, True, True)[1].t())
        else:
            loss = softmax_entropy(clip_logits)
            prob_map = clip_logits.softmax(1)
            pred = int(clip_logits.topk(1, 1, True, True)[1].t()[0])

        return image_features, clip_logits, loss, prob_map, pred


def get_ood_preprocess():
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                std=[0.26862954, 0.26130258, 0.27577711])
    base_transform = transforms.Compose([
        transforms.Resize(224, interpolation=BICUBIC),
        transforms.CenterCrop(224)])
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        normalize])
    aug_preprocess = AugMixAugmenter(base_transform, preprocess, n_views=63, augmix=True)

    return aug_preprocess


def get_config_file(config_path, dataset_name):
    """Get configuration file for the dataset.
    
    Args:
        config_path: Either a direct path to a config file or a directory containing config files
        dataset_name: Name of the dataset
    
    Returns:
        cfg: Configuration dictionary
    """
    # If config_path is a direct path to a yaml file, use it directly
    if config_path.endswith('.yaml'):
        config_file = config_path
    else:
        # Otherwise, construct path based on dataset name
        if dataset_name == "I":
            config_name = "imagenet.yaml"
        elif dataset_name in ["A", "V", "R", "S"]:
            config_name = f"imagenet_{dataset_name.lower()}.yaml"
        elif dataset_name == "C":
            config_name = "cifar10c.yaml"
        else:
            config_name = f"{dataset_name}.yaml"
        
        config_file = os.path.join(config_path, config_name)
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"The configuration file {config_file} was not found.")
    
    print(f"Loading config file: {config_file}")
    with open(config_file, 'r') as file:
        cfg = yaml.load(file, Loader=yaml.SafeLoader)
    
    return cfg


def build_test_data_loader(dataset_name, root_path, preprocess):
    if dataset_name == 'I':
        dataset = ImageNet(root_path, preprocess)
        test_loader = torch.utils.data.DataLoader(dataset.test, batch_size=1, num_workers=8, shuffle=True)
    
    elif dataset_name in ['A','V','R','S']:
        preprocess = get_ood_preprocess()
        dataset = build_dataset(f"imagenet-{dataset_name.lower()}", root_path)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=1, is_train=False, tfm=preprocess, shuffle=True)

    elif dataset_name == 'C':
        # For CIFAR-10-C, we'll test on all corruptions and severities
        all_results = []
        for corruption in CIFAR10C_CORRUPTIONS:
            for severity in range(1, 6):
                # Create dataset with transform
                dataset = build_dataset("cifar10-c", root_path, corruption_type=corruption, severity=severity)
                # Create data loader directly since CIFAR10C is already a Dataset
                test_loader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=1,
                    num_workers=8,
                    shuffle=True,
                    pin_memory=(torch.cuda.is_available())
                )
                all_results.append((test_loader, dataset.classnames, dataset.template))
        return all_results

    elif dataset_name in ['caltech101','dtd','eurosat','fgvc','food101','oxford_flowers','oxford_pets','stanford_cars','sun397','ucf101']:
        dataset = build_dataset(dataset_name, root_path)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=1, is_train=False, tfm=preprocess, shuffle=True)
    
    else:
        raise ValueError("Dataset is not from the chosen list")
    
    return test_loader, dataset.classnames, dataset.template