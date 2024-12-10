import matplotlib.pyplot as plt
import seaborn as sns

def plot_hyperparameter_impact(sweep_results):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))
    
    # Shot capacity vs Accuracy
    sns.scatterplot(data=sweep_results, 
                   x='positive.shot_capacity', 
                   y='Averaged test accuracy',
                   size='memory_budget',
                   hue='compute_budget',
                   ax=ax1)
    ax1.set_title('Shot Capacity Impact')
    
    # Alpha parameter impact
    sns.scatterplot(data=sweep_results,
                   x='positive.alpha',
                   y='Averaged test accuracy',
                   size='memory_budget',
                   hue='compute_budget',
                   ax=ax2)
    ax2.set_title('Alpha Parameter Impact')
    
    # Beta parameter impact
    sns.scatterplot(data=sweep_results,
                   x='positive.beta',
                   y='Averaged test accuracy',
                   size='memory_budget',
                   hue='compute_budget',
                   ax=ax3)
    ax3.set_title('Beta Parameter Impact')
    
    # Adaptation window impact
    sns.scatterplot(data=sweep_results,
                   x='adaptation_window',
                   y='Averaged test accuracy',
                   size='memory_budget',
                   hue='compute_budget',
                   ax=ax4)
    ax4.set_title('Adaptation Window Impact')
    
    plt.tight_layout()
    return fig 