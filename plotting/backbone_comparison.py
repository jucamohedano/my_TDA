import matplotlib.pyplot as plt
import seaborn as sns

def plot_backbone_comparison(rn50_data, vit_data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy Comparison
    models = ['RN50', 'ViT-B/16']
    metrics = ['Pre-cache accuracy', 'Post-cache accuracy']
    
    # Box plot for accuracy distribution
    ax1.boxplot([
        rn50_data['Post-cache accuracy'],
        vit_data['Post-cache accuracy']
    ], labels=models)
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy Distribution by Backbone')
    
    # Efficiency Comparison
    efficiency_data = {
        'RN50': rn50_data['Post-cache accuracy'] / rn50_data['Memory usage (MB)'],
        'ViT-B/16': vit_data['Post-cache accuracy'] / vit_data['Memory usage (MB)']
    }
    ax2.boxplot(efficiency_data.values(), labels=models)
    ax2.set_ylabel('Efficiency (Accuracy/MB)')
    ax2.set_title('Memory Efficiency by Backbone')
    
    plt.tight_layout()
    return fig 