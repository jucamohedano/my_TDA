import wandb
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def fetch_imagenet_runs(project_name="ETTA-CLIP"):
    """Fetch wandb runs for ImageNet variants."""
    api = wandb.Api()
    runs = api.runs(project_name)
    
    # Dictionary to store results
    results = {
        'dataset': [],
        'accuracy': [],
        'method': []  # 'standard' or 'waiting'
    }
    
    # Dataset mapping
    dataset_names = {
        'I': 'ImageNet',
        'A': 'ImageNet-A',
        'R': 'ImageNet-R',
        'V': 'ImageNet-V',
        'S': 'ImageNet-S'
    }
    
    print("\nProcessing ImageNet variant runs...")
    
    for run in runs:
        try:
            # Check if run name matches our patterns
            name = run.name
            
            # Skip CIFAR-10-C runs
            if name.startswith('cifar10c'):
                continue
                
            # Check if it's a valid ImageNet run
            base_name = name.replace('_with_waiting', '')
            if base_name not in dataset_names:
                continue
            
            # Determine method and dataset
            method = 'waiting' if '_with_waiting' in name else 'standard'
            dataset = dataset_names[base_name]
            
            # Try different metric names
            accuracy = None
            metric_names = [
                dataset_names[base_name],  # e.g., 'ImageNet', 'ImageNet-A', etc.
                base_name,                 # e.g., 'I', 'A', etc.
                'Final test accuracy',
                'Averaged test accuracy'
            ]
            
            for metric in metric_names:
                if metric in run.summary:
                    accuracy = run.summary[metric]
                    print(f"Found accuracy in metric: {metric}")
                    break
            
            if accuracy is None:
                print(f"Warning: No accuracy found for run {name}")
                continue
            
            # Convert to percentage if needed
            if accuracy > 1.0:
                accuracy = accuracy / 100.0
            
            results['dataset'].append(dataset)
            results['accuracy'].append(accuracy * 100)  # Convert to percentage
            results['method'].append(method)
            
            print(f"Processed run: {name}")
            print(f"- Dataset: {dataset}")
            print(f"- Method: {method}")
            print(f"- Accuracy: {accuracy*100:.2f}%")
            
        except Exception as e:
            print(f"Error processing run {run.name}: {str(e)}")
    
    df = pd.DataFrame(results)
    
    # Print validation information
    print("\nData Validation:")
    print("-" * 50)
    
    # Check which datasets and methods we have
    for dataset in dataset_names.values():
        print(f"\n{dataset}:")
        for method in ['standard', 'waiting']:
            mask = (df['dataset'] == dataset) & (df['method'] == method)
            if mask.any():
                accuracies = df[mask]['accuracy']
                print(f"  {method}: {len(accuracies)} runs, "
                      f"mean = {accuracies.mean():.2f}%, "
                      f"std = {accuracies.std():.2f}%")
            else:
                print(f"  {method}: No runs found")
    
    return df

def plot_comparison(df, output_path='imagenet_comparison.png'):
    """Create comparison plots for ImageNet variants."""
    if df.empty:
        raise ValueError("No data available for plotting")
    
    # Set style to a built-in clean style
    plt.style.use('default')
    
    # Create figure with custom size and DPI
    fig = plt.figure(figsize=(15, 12), dpi=300)
    
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
    
    # Create subplot for the bar plot with specific background color
    ax1 = plt.subplot(2, 1, 1)
    ax1.set_facecolor('white')
    
    # Calculate mean accuracy for each dataset and method
    pivot_data = df.pivot_table(
        index='dataset',
        columns='method',
        values='accuracy',
        aggfunc=['mean', 'std']
    )
    
    # Prepare data for plotting
    datasets = pivot_data.index
    x = np.arange(len(datasets))
    width = 0.35
    
    # Plot bars for each method
    means_standard = pivot_data['mean']['standard']
    means_waiting = pivot_data['mean']['waiting']
    
    # Handle NaN values in standard deviations
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
            label='TDA with Waiting', color=colors['waiting'], capsize=5,
            alpha=0.8, edgecolor='white', linewidth=1)
    
    # Customize plot
    plt.title('ImageNet Variants Performance Comparison\nArchitecture: CLIP ViT-B/16', 
              pad=20, color=colors['text'], fontsize=14, fontweight='bold')
    plt.xlabel('Dataset', labelpad=10, color=colors['text'], fontsize=12)
    plt.ylabel('Accuracy (%)', labelpad=10, color=colors['text'], fontsize=12)
    plt.xticks(x, datasets, rotation=45, ha='right', color=colors['text'])
    
    # Enhance legend
    legend = plt.legend(loc='upper right', frameon=True, fancybox=True, 
                       shadow=True, fontsize=10)
    legend.get_frame().set_facecolor('white')
    
    # Enhance grid
    plt.grid(True, linestyle='--', alpha=0.3, color=colors['grid'], zorder=0)
    
    # Add value labels on the bars
    def add_value_labels(x, values, std, offset):
        for i, (v, s) in enumerate(zip(values, std)):
            if not np.isnan(v):
                plt.text(x[i] + offset, v + s + 1, f'{v:.1f}%', 
                        ha='center', va='bottom', color=colors['text'],
                        fontsize=10, fontweight='bold')
    
    add_value_labels(x, means_standard, std_standard, -width/2)
    add_value_labels(x, means_waiting, std_waiting, width/2)
    
    # Create subplot for the difference plot
    ax2 = plt.subplot(2, 1, 2)
    ax2.set_facecolor('white')
    
    # Calculate accuracy differences (waiting - standard)
    differences = means_waiting - means_standard
    
    # Create bar plot for differences with gradient colors
    colors_diff = ['#27AE60' if x >= 0 else '#C0392B' for x in differences]  # Green/Red
    bars = plt.bar(datasets, differences, color=colors_diff, alpha=0.7,
                  edgecolor='white', linewidth=1)
    
    # Calculate combined standard deviation for differences
    # Use zero for NaN values in the calculation
    diff_std = np.sqrt(
        np.nan_to_num(std_waiting**2, 0) + 
        np.nan_to_num(std_standard**2, 0)
    )
    
    # Add error bars for the differences
    plt.errorbar(datasets, differences, yerr=diff_std, fmt='none', 
                color='black', capsize=5, alpha=0.5)
    
    # Customize plot
    plt.title('Impact of Waiting Strategy on Accuracy', 
              pad=20, color=colors['text'], fontsize=14, fontweight='bold')
    plt.xlabel('Dataset', labelpad=10, color=colors['text'], fontsize=12)
    plt.ylabel('Accuracy Improvement (%)', labelpad=10, color=colors['text'], fontsize=12)
    
    # Enhance grid
    plt.grid(True, linestyle='--', alpha=0.3, color=colors['grid'], zorder=0)
    plt.xticks(rotation=45, ha='right', color=colors['text'])
    
    # Add value labels on the bars
    for i, (v, s) in enumerate(zip(differences, diff_std)):
        if not np.isnan(v):
            color = '#27AE60' if v >= 0 else '#C0392B'
            plt.text(i, v + np.sign(v)*(s + 0.5), f'{v:+.1f}%', 
                    ha='center', va='bottom' if v >= 0 else 'top',
                    color=color, fontsize=10, fontweight='bold')
    
    # Add horizontal line at y=0 with enhanced style
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Set figure background color
    fig.patch.set_facecolor('white')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save with high quality
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()

def generate_statistics(df):
    """Generate and print summary statistics."""
    print("\nSummary Statistics for CLIP ViT-B/16:")
    print("-" * 50)
    
    # Overall statistics by method
    print("\nOverall Performance:")
    for method in ['standard', 'waiting']:
        method_stats = df[df['method'] == method]['accuracy']
        print(f"\n{method.capitalize()} method:")
        print(f"Mean accuracy: {method_stats.mean():.2f}%")
        print(f"Std deviation: {method_stats.std():.2f}%")
        print(f"Min accuracy: {method_stats.min():.2f}%")
        print(f"Max accuracy: {method_stats.max():.2f}%")
    
    # Performance by dataset
    print("\nPerformance by Dataset:")
    pivot_data = df.pivot_table(
        index='dataset',
        columns='method',
        values='accuracy',
        aggfunc=['mean', 'std']
    )
    print("\nMean and Standard Deviation:")
    print(pivot_data)
    
    # Calculate improvements
    improvements = pivot_data['mean']['waiting'] - pivot_data['mean']['standard']
    print("\nImprovements with Waiting:")
    for dataset, imp in improvements.items():
        print(f"{dataset}: {imp:+.2f}%")
    
    # Overall improvement
    mean_improvement = improvements.mean()
    print(f"\nAverage improvement across all datasets: {mean_improvement:+.2f}%")

def main():
    """Main function to generate visualizations and statistics."""
    try:
        print("Fetching results from wandb...")
        df = fetch_imagenet_runs()
        
        if df.empty:
            print("No data available to visualize")
            return
        
        print("\nGenerating visualizations...")
        plot_comparison(df)
        
        print("\nGenerating statistics...")
        generate_statistics(df)
        
        print("\nVisualization complete! Check imagenet_comparison.png")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nPlease check that:")
        print("1. You're logged into wandb")
        print("2. Your wandb project contains ImageNet variant runs")
        print("3. The runs have completed and contain accuracy metrics")

if __name__ == "__main__":
    main() 