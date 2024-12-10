import matplotlib.pyplot as plt

def plot_resource_performance(wandb_data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Memory vs Accuracy
    scatter1 = ax1.scatter(wandb_data['Memory usage (MB)'], 
                          wandb_data['Post-cache accuracy'],
                          c=wandb_data['Samples processed'], 
                          cmap='viridis')
    ax1.set_xlabel('Memory Usage (MB)')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Memory Usage vs Accuracy')
    plt.colorbar(scatter1, ax=ax1, label='Samples Processed')
    
    # Inference Time vs Accuracy
    scatter2 = ax2.scatter(wandb_data['Total inference time (s)'], 
                          wandb_data['Post-cache accuracy'],
                          c=wandb_data['Samples processed'], 
                          cmap='viridis')
    ax2.set_xlabel('Inference Time (s)')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Inference Time vs Accuracy')
    plt.colorbar(scatter2, ax=ax2, label='Samples Processed')
    
    plt.tight_layout()
    return fig 