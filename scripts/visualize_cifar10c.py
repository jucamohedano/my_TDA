import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import torch

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def load_cifar10c_sample(root_dir, corruption_type='gaussian_noise', severity=1, index=0):
    """Load a sample image from CIFAR-10-C dataset."""
    # Construct paths
    severity_dir = os.path.join(root_dir, 'cifar10-c', 'corrupted', f'severity-{severity}')
    data_path = os.path.join(severity_dir, f'{corruption_type}.npy')
    
    # Load data
    data = np.load(data_path)
    img = data[index]
    return img

def visualize_upscaling(img_array, save_path='cifar10c_upscale_comparison.png'):
    """Visualize original and upscaled versions of the image."""
    # Convert numpy array to PIL Image
    img_pil = Image.fromarray(img_array)
    
    # Create upscaling transform
    transform = transforms.Compose([
        transforms.Resize(224, interpolation=BICUBIC),
        transforms.CenterCrop(224)
    ])
    
    # Apply transform
    img_upscaled = transform(img_pil)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot original image
    ax1.imshow(img_array)
    ax1.set_title(f'Original ({img_array.shape[0]}x{img_array.shape[1]})')
    ax1.axis('off')
    
    # Plot upscaled image
    ax2.imshow(img_upscaled)
    ax2.set_title(f'Upscaled (224x224)')
    ax2.axis('off')
    
    # Add main title
    plt.suptitle('CIFAR-10-C Image Upscaling Comparison', y=0.95)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Comparison saved to {save_path}")

def main():
    # Configuration
    root_dir = './dataset'  # Adjust this path as needed
    corruption_types = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
                       'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog']
    severity = 1
    num_samples = 5
    
    # Create output directory
    output_dir = 'cifar10c_visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate visualizations for different corruptions and samples
    for corruption in corruption_types:
        for i in range(num_samples):
            try:
                # Load image
                img = load_cifar10c_sample(root_dir, corruption, severity, i)
                
                # Create visualization
                save_path = os.path.join(output_dir, f'{corruption}_sample{i}_comparison.png')
                visualize_upscaling(img, save_path)
                
            except Exception as e:
                print(f"Error processing {corruption} sample {i}: {str(e)}")

if __name__ == '__main__':
    main() 