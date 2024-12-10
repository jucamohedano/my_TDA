import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_accuracy_evolution(wandb_data):
    plt.figure(figsize=(10, 6))
    
    # Create three lines for different accuracy metrics
    plt.plot(wandb_data['Samples processed'], wandb_data['Pre-cache accuracy'], 
             label='Pre-cache Accuracy', linestyle='--')
    plt.plot(wandb_data['Samples processed'], wandb_data['Post-cache accuracy'], 
             label='Post-cache Accuracy')
    plt.plot(wandb_data['Samples processed'], wandb_data['Averaged test accuracy'], 
             label='Running Average Accuracy')
    
    plt.xlabel('Number of Samples Processed')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Evolution Over Samples')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return plt 