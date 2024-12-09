import wandb
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from utils import CIFAR10C_CORRUPTIONS

def fetch_cifar10c_results(project_name="ETTA-CLIP"):
    """Fetch both standard and waiting results for CIFAR-10-C from wandb.
    
    Args:
        project_name (str): Name of the wandb project to fetch results from.
        
    Returns:
        pd.DataFrame: DataFrame containing results with columns:
            - corruption_type: Type of corruption applied
            - severity: Severity level (1-5)
            - accuracy: Test accuracy (as percentage)
            - method: Either 'standard' or 'waiting'
    """
    api = wandb.Api()
    runs = api.runs(project_name)
    
    results = {
        'corruption_type': [],
        'severity': [],
        'accuracy': [],
        'method': []
    }
    
    print("\nProcessing CIFAR-10-C runs...")
    
    for run in runs:
        try:
            name = run.name
            if not name.startswith('cifar10c_'):
                continue
                
            is_waiting = name.endswith('_with_waiting')
            method = 'waiting' if is_waiting else 'standard'
            
            name_parts = name.split('_')
            if is_waiting:
                name_parts = name_parts[:-2]
            severity = int(name_parts[-1][1:])
            corruption_type = '_'.join(name_parts[1:-1])
            
            accuracy = run.summary.get('Averaged test accuracy')
            if accuracy is None:
                print(f"Warning: No accuracy found for run {name}")
                continue
            
            # Convert to percentage (accuracy is already in percentage form)
            # accuracy = accuracy * 100
            
            results['corruption_type'].append(corruption_type)
            results['severity'].append(severity)
            results['accuracy'].append(accuracy)
            results['method'].append(method)
            
            print(f"Processed run: {name}")
            print(f"- Corruption: {corruption_type}")
            print(f"- Severity: {severity}")
            print(f"- Method: {method}")
            print(f"- Accuracy: {accuracy:.2f}%")
            
        except Exception as e:
            print(f"Error processing run {name}: {str(e)}")
    
    return pd.DataFrame(results)

def plot_side_by_side_heatmaps(df, output_path='cifar10c_comparison_heatmaps.png'):
    """Create side-by-side heatmaps comparing standard and waiting methods.
    
    This visualization shows two heatmaps side by side:
    - Left: Standard TDA performance
    - Right: TDA + Neutral Cache performance
    Each cell shows the accuracy for a specific corruption type and severity level.
    
    Args:
        df (pd.DataFrame): DataFrame containing the results
        output_path (str): Path to save the output image
        
    The heatmap uses a RdYlGn (Red-Yellow-Green) colormap where:
    - Red indicates lower accuracy
    - Yellow indicates medium accuracy
    - Green indicates higher accuracy
    """
    standard_matrix = df[df['method'] == 'standard'].pivot(
        index='corruption_type', columns='severity', values='accuracy')
    waiting_matrix = df[df['method'] == 'waiting'].pivot(
        index='corruption_type', columns='severity', values='accuracy')
    
    plt.style.use('default')
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10})
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), dpi=300)
    
    sns.heatmap(standard_matrix, ax=ax1, cmap='RdYlGn', center=50,
                annot=True, fmt='.1f', cbar_kws={'label': 'Accuracy (%)'},
                vmin=0, vmax=100)
    ax1.set_title('Standard TDA\nArchitecture: CLIP ViT-B/16', 
                 pad=20, fontsize=14, fontweight='bold')
    
    sns.heatmap(waiting_matrix, ax=ax2, cmap='RdYlGn', center=50,
                annot=True, fmt='.1f', cbar_kws={'label': 'Accuracy (%)'},
                vmin=0, vmax=100)
    ax2.set_title('TDA + Neutral Cache\nArchitecture: CLIP ViT-B/16', 
                 pad=20, fontsize=14, fontweight='bold')
    
    for ax in [ax1, ax2]:
        ax.set_xlabel('Severity Level', labelpad=10)
        ax.set_ylabel('Corruption Type', labelpad=10)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()

def plot_difference_heatmap(df, output_path='cifar10c_improvement_heatmap.png'):
    """Create a heatmap showing the improvement from standard to waiting.
    
    This visualization shows a single heatmap where each cell represents the
    difference in accuracy between TDA + Neutral Cache and standard TDA.
    
    Args:
        df (pd.DataFrame): DataFrame containing the results
        output_path (str): Path to save the output image
        
    The heatmap uses a diverging colormap where:
    - Blue indicates improvement with neutral cache
    - White indicates no change
    - Red indicates degradation with neutral cache
    
    Annotations show the exact percentage point difference.
    """
    standard_matrix = df[df['method'] == 'standard'].pivot(
        index='corruption_type', columns='severity', values='accuracy')
    waiting_matrix = df[df['method'] == 'waiting'].pivot(
        index='corruption_type', columns='severity', values='accuracy')
    
    diff_matrix = waiting_matrix - standard_matrix
    
    plt.style.use('default')
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10})
    
    plt.figure(figsize=(12, 8), dpi=300)
    
    # Use RdYlGn colormap for consistency with other heatmaps
    cmap = 'RdYlGn'
    
    sns.heatmap(diff_matrix, cmap=cmap, center=0,
                annot=True, fmt='.1f', 
                cbar_kws={'label': 'Accuracy Improvement (%)'},
                vmin=-5, vmax=5)
    
    plt.title('Improvement with Neutral Cache\nArchitecture: CLIP ViT-B/16', 
              pad=20, fontsize=14, fontweight='bold')
    plt.xlabel('Severity Level', labelpad=10)
    plt.ylabel('Corruption Type', labelpad=10)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()

def plot_violin_comparison(df, output_path='cifar10c_violin_comparison.png'):
    """Create violin plots comparing the distribution of accuracies.
    
    This visualization shows the distribution of accuracies for each corruption type,
    split by method (standard vs waiting).
    
    Args:
        df (pd.DataFrame): DataFrame containing the results
        output_path (str): Path to save the output image
        
    The violin plot shows:
    - The full distribution shape for each method
    - Box plots inside showing quartiles
    - Split violins to directly compare methods
    - Different colors for standard vs waiting
    """
    plt.style.use('default')
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10})
    
    plt.figure(figsize=(15, 8), dpi=300)
    
    sns.violinplot(data=df, x='corruption_type', y='accuracy', hue='method',
                  split=True, inner='box', palette=['#2E86C1', '#E74C3C'])
    
    plt.title('Accuracy Distribution by Corruption Type\nArchitecture: CLIP ViT-B/16',
              pad=20, fontsize=14, fontweight='bold')
    plt.xlabel('Corruption Type', labelpad=10)
    plt.ylabel('Accuracy (%)', labelpad=10)
    plt.xticks(rotation=45, ha='right')
    
    plt.legend(title='Method', labels=['Standard TDA', 'TDA + Neutral Cache'])
    
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()

def plot_scatter_comparison(df, output_path='cifar10c_scatter_comparison.png'):
    """Create a scatter plot comparing standard vs waiting accuracies.
    
    This visualization shows a direct comparison between methods where:
    - Each point represents one corruption type at one severity level
    - X-axis shows standard TDA accuracy
    - Y-axis shows TDA + Neutral Cache accuracy
    - Points above the diagonal line indicate improvement
    
    Args:
        df (pd.DataFrame): DataFrame containing the results
        output_path (str): Path to save the output image
        
    The plot includes:
    - Diagonal line showing where methods perform equally
    - Color coding by severity level
    - Statistics about number of improvements/degradations
    - Grid lines for easier comparison
    """
    standard_data = df[df['method'] == 'standard'][['corruption_type', 'severity', 'accuracy']]
    waiting_data = df[df['method'] == 'waiting'][['corruption_type', 'severity', 'accuracy']]
    
    comparison_data = pd.merge(
        standard_data, waiting_data,
        on=['corruption_type', 'severity'],
        suffixes=('_standard', '_waiting')
    )
    
    plt.style.use('default')
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10})
    
    plt.figure(figsize=(10, 10), dpi=300)
    
    min_val = min(comparison_data['accuracy_standard'].min(),
                 comparison_data['accuracy_waiting'].min())
    max_val = max(comparison_data['accuracy_standard'].max(),
                 comparison_data['accuracy_waiting'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    scatter = plt.scatter(comparison_data['accuracy_standard'],
                         comparison_data['accuracy_waiting'],
                         c=comparison_data['severity'],
                         cmap='viridis',
                         alpha=0.6)
    
    plt.colorbar(scatter, label='Severity Level')
    
    plt.title('Standard TDA vs TDA + Neutral Cache\nArchitecture: CLIP ViT-B/16',
              pad=20, fontsize=14, fontweight='bold')
    plt.xlabel('Standard TDA Accuracy (%)', labelpad=10)
    plt.ylabel('TDA + Neutral Cache Accuracy (%)', labelpad=10)
    
    above = (comparison_data['accuracy_waiting'] > comparison_data['accuracy_standard']).sum()
    below = (comparison_data['accuracy_waiting'] < comparison_data['accuracy_standard']).sum()
    equal = (comparison_data['accuracy_waiting'] == comparison_data['accuracy_standard']).sum()
    
    plt.text(0.02, 0.98, f'Above line: {above} (improved)\nBelow line: {below} (degraded)\nOn line: {equal} (unchanged)',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()

def plot_severity_averaged_comparison(df, output_path='cifar10c_severity_averaged_comparison.png'):
    """Create comparison plots for CIFAR-10-C corruptions, averaged across severity levels.
    
    This visualization shows:
    1. Bar plot comparing standard TDA vs TDA + Neutral Cache for each corruption
    2. Difference plot showing the impact of the neutral cache strategy
    
    Args:
        df (pd.DataFrame): DataFrame containing the results
        output_path (str): Path to save the output image
    """
    if df.empty:
        raise ValueError("No data available for plotting")
    
    # Set style to a built-in clean style
    plt.style.use('default')
    
    # Create figure with custom size and DPI
    fig = plt.figure(figsize=(20, 14), dpi=300)  # Increased width
    
    # Set the font family and size
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
    })
    
    # Define colors
    colors = {
        'standard': '#2E86C1',  # Deep blue
        'waiting': '#E74C3C',   # Deep red
        'grid': '#EAECEE',      # Light gray
        'text': '#2C3E50'       # Dark gray
    }
    
    # Calculate mean and std across severity levels for each corruption type and method
    pivot_data = df.pivot_table(
        index='corruption_type',
        columns='method',
        values='accuracy',
        aggfunc=['mean', 'std']
    )
    
    # Sort corruptions by standard method performance for better visualization
    pivot_data = pivot_data.sort_values(('mean', 'standard'), ascending=True)
    
    # Create subplot for the bar plot with more space
    ax1 = plt.subplot(2, 1, 1)
    ax1.set_facecolor('white')
    
    # Prepare data for plotting
    corruptions = pivot_data.index
    x = np.arange(len(corruptions))
    width = 0.35
    
    # Get means and standard deviations
    means_standard = pivot_data['mean']['standard']
    means_waiting = pivot_data['mean']['waiting']
    std_standard = pivot_data['std'].get('standard', np.zeros_like(means_standard))
    std_waiting = pivot_data['std'].get('waiting', np.zeros_like(means_waiting))
    
    # Replace NaN values with zeros for plotting
    std_standard = np.nan_to_num(std_standard, 0)
    std_waiting = np.nan_to_num(std_waiting, 0)
    
    # Create bars with enhanced styling
    plt.bar(x - width/2, means_standard, width, yerr=std_standard,
            label='Standard TDA', color=colors['standard'], capsize=5,
            alpha=0.8, edgecolor='white', linewidth=1)
    plt.bar(x + width/2, means_waiting, width, yerr=std_waiting,
            label='TDA + Neutral Cache', color=colors['waiting'], capsize=5,
            alpha=0.8, edgecolor='white', linewidth=1)
    
    # Customize plot
    plt.title('CIFAR-10-C Performance by Corruption Type\nArchitecture: CLIP ViT-B/16', 
              pad=20, color=colors['text'], fontsize=14, fontweight='bold')
    plt.xlabel('Corruption Type', labelpad=10, color=colors['text'], fontsize=12)
    plt.ylabel('Average Accuracy Across Severities (%)', labelpad=10, color=colors['text'], fontsize=12)
    plt.xticks(x, corruptions, rotation=45, ha='right', color=colors['text'])
    
    # Add some padding to the y-axis limits
    ymin, ymax = plt.ylim()
    plt.ylim(ymin, ymax + 8)  # Increased padding for alternating labels
    
    # Enhance legend
    legend = plt.legend(loc='upper right', frameon=True, fancybox=True, 
                       shadow=True, fontsize=10)
    legend.get_frame().set_facecolor('white')
    
    # Enhance grid
    plt.grid(True, linestyle='--', alpha=0.3, color=colors['grid'], zorder=0)
    
    # Add value labels on the bars at their natural height
    def add_value_labels(x, values, std):
        for i, v in enumerate(values):
            if not np.isnan(v):
                plt.text(x[i], v + 1, f'{v:.1f}%',  # Set label directly above the bar
                         ha='center', va='bottom', color=colors['text'],
                         fontsize=8, fontweight='bold')  # Smaller font size
    
    add_value_labels(x - width/2, means_standard, std_standard)
    add_value_labels(x + width/2, means_waiting, std_waiting)
    
    # Create subplot for the difference plot with more space
    ax2 = plt.subplot(2, 1, 2)
    ax2.set_facecolor('white')
    
    # Calculate accuracy differences (waiting - standard)
    differences = means_waiting - means_standard
    
    # Create bar plot for differences with gradient colors
    colors_diff = ['#27AE60' if x >= 0 else '#C0392B' for x in differences]  # Green/Red
    bars = plt.bar(corruptions, differences, color=colors_diff, alpha=0.7,
                  edgecolor='white', linewidth=1)
    
    # Calculate combined standard deviation for differences
    diff_std = np.sqrt(
        np.nan_to_num(std_waiting**2, 0) + 
        np.nan_to_num(std_standard**2, 0)
    )
    
    # Add error bars for the differences
    plt.errorbar(corruptions, differences, yerr=diff_std, fmt='none', 
                color='black', capsize=5, alpha=0.5)
    
    # Customize plot
    plt.title('Impact of Neutral Cache on Accuracy', 
              pad=20, color=colors['text'], fontsize=14, fontweight='bold')
    plt.xlabel('Corruption Type', labelpad=10, color=colors['text'], fontsize=12)
    plt.ylabel('Accuracy Improvement (%)', labelpad=10, color=colors['text'], fontsize=12)
    
    # Enhance grid
    plt.grid(True, linestyle='--', alpha=0.3, color=colors['grid'], zorder=0)
    plt.xticks(rotation=45, ha='right', color=colors['text'])
    
    # Add padding to y-axis limits for the difference plot
    ymin, ymax = plt.ylim()
    padding = max(abs(ymin), abs(ymax)) * 0.2  # 20% padding
    plt.ylim(ymin - padding, ymax + padding)
    
    # Add value labels on the bars with adjusted position
    for i, (v, s) in enumerate(zip(differences, diff_std)):
        if not np.isnan(v):
            color = '#27AE60' if v >= 0 else '#C0392B'
            # Adjust vertical position based on value sign
            if v >= 0:
                y_pos = v + s + padding * 0.3
                va = 'bottom'
            else:
                y_pos = v - s - padding * 0.3
                va = 'top'
            plt.text(i, y_pos, f'{v:+.1f}%', 
                    ha='center', va=va,
                    color=color, fontsize=10, fontweight='bold')
    
    # Add horizontal line at y=0
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Set figure background color
    fig.patch.set_facecolor('white')
    
    # Adjust layout with more space between subplots
    plt.tight_layout(h_pad=3.0)  # Increased spacing between subplots
    
    # Save with high quality
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()

def main():
    """Main function to generate all visualizations."""
    try:
        print("Fetching results from wandb...")
        df = fetch_cifar10c_results()
        
        if df.empty:
            print("No data available to visualize")
            return
        
        print("\nGenerating visualizations...")
        plot_side_by_side_heatmaps(df)
        plot_difference_heatmap(df)
        plot_violin_comparison(df)
        plot_scatter_comparison(df)
        plot_severity_averaged_comparison(df)
        
        print("\nVisualization complete! Check:")
        print("1. cifar10c_comparison_heatmaps.png (side-by-side comparison)")
        print("2. cifar10c_improvement_heatmap.png (improvement heatmap)")
        print("3. cifar10c_violin_comparison.png (distribution comparison)")
        print("4. cifar10c_scatter_comparison.png (direct comparison)")
        print("5. cifar10c_severity_averaged_comparison.png (severity-averaged comparison)")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nPlease check that:")
        print("1. You're logged into wandb")
        print("2. Your wandb project contains CIFAR-10-C runs")
        print("3. The runs have completed and contain accuracy metrics")

if __name__ == "__main__":
    main() 