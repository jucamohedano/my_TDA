import matplotlib.pyplot as plt

def plot_resource_tradeoff(wandb_data):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scatter = ax.scatter(
        wandb_data['Memory usage (MB)'],
        wandb_data['Total inference time (s)'],
        c=wandb_data['Post-cache accuracy'],
        s=100,
        cmap='viridis'
    )
    
    plt.colorbar(scatter, label='Accuracy')
    ax.set_xlabel('Memory Usage (MB)')
    ax.set_ylabel('Inference Time (s)')
    ax.set_title('Memory-Time Trade-off with Accuracy')
    
    # Add contour lines for constant efficiency
    efficiency = wandb_data['Post-cache accuracy'] / (wandb_data['Memory usage (MB)'] * wandb_data['Total inference time (s)'])
    ax.tricontour(
        wandb_data['Memory usage (MB)'],
        wandb_data['Total inference time (s)'],
        efficiency,
        colors='black',
        alpha=0.3,
        linestyles='--'
    )
    
    return fig 