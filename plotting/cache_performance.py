import matplotlib.pyplot as plt

def plot_cache_metrics(wandb_data):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Cache sizes
    ax1.plot(wandb_data['Samples processed'], wandb_data['Positive cache size'], 
             label='Positive Cache')
    ax1.plot(wandb_data['Samples processed'], wandb_data['Negative cache size'], 
             label='Negative Cache')
    ax1.set_xlabel('Samples Processed')
    ax1.set_ylabel('Cache Size')
    ax1.set_title('Cache Size Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Hit rates
    ax2.plot(wandb_data['Samples processed'], wandb_data['Positive cache hit rate'], 
             label='Positive Cache')
    ax2.plot(wandb_data['Samples processed'], wandb_data['Negative cache hit rate'], 
             label='Negative Cache')
    ax2.set_xlabel('Samples Processed')
    ax2.set_ylabel('Hit Rate')
    ax2.set_title('Cache Hit Rates')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig 