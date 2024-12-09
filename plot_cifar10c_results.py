import wandb
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from utils import CIFAR10C_CORRUPTIONS

def inspect_wandb_metrics(project_name="ETTA-CLIP"):
    """Inspect available metrics in wandb runs."""
    api = wandb.Api()
    runs = api.runs(project_name)
    
    print("\nInspecting wandb runs:")
    print("-" * 50)
    
    for run in runs:
        print(f"\nRun: {run.name}")
        print("Config:")
        for key, value in run.config.items():
            print(f"  {key}: {value}")
        print("Summary metrics:")
        for key, value in run.summary.items():
            print(f"  {key}: {value}")
        print("History keys:")
        try:
            # Get first row of history to see available metrics
            history = run.history(samples=1)
            if not history.empty:
                print("  " + "\n  ".join(history.columns))
        except Exception as e:
            print(f"  Error getting history: {str(e)}")
        print("-" * 50)

def fetch_wandb_runs(project_name="ETTA-CLIP"):
    """Fetch wandb runs matching the CIFAR-10-C pattern."""
    api = wandb.Api()
    runs = api.runs(project_name)
    
    # Debug information
    print(f"\nFound {len(runs)} total runs")
    
    cifar_runs = []
    for run in runs:
        if run.config.get('dataset') == 'C' or 'cifar10c' in run.name.lower():
            cifar_runs.append(run)
            print(f"- Run name: {run.name}")
            print(f"  Config: {run.config}")
            # Print available metrics
            print("  Available metrics:")
            for key in run.summary.keys():
                print(f"    {key}: {run.summary[key]}")
    
    print(f"\nFound {len(cifar_runs)} CIFAR-10-C runs")
    return cifar_runs

def create_results_matrix():
    """Create and populate the results matrix from wandb runs."""
    # Initialize results dictionary
    results = {
        'corruption_type': [],
        'severity': [],
        'accuracy': [],
        'run_name': []  # Add run name for debugging
    }
    
    # Fetch runs from wandb
    api = wandb.Api()
    runs = api.runs("ETTA-CLIP")
    
    print(f"\nProcessing wandb runs...")
    
    # Dictionary to store the latest run for each combination
    latest_runs = {}
    
    # Process each run
    for run in runs:
        try:
            # Only process CIFAR-10-C runs
            if not run.name.startswith('cifar10c_'):
                continue
                
            # Get metrics directly from summary
            corruption_type = run.summary.get('corruption_type')
            severity = run.summary.get('severity')
            accuracy = run.summary.get('Averaged test accuracy')
            
            if not all([corruption_type, severity, accuracy]):
                print(f"Warning: Missing required metrics in run {run.name}")
                continue
            
            # Create a unique identifier for this combination
            combination = (corruption_type, severity)
            
            # Store this run's information
            if combination not in latest_runs or run.created_at > latest_runs[combination]['created_at']:
                latest_runs[combination] = {
                    'run_name': run.name,
                    'accuracy': accuracy,
                    'created_at': run.created_at
                }
    
        except Exception as e:
            print(f"Error processing run {run.name}: {str(e)}")
    
    # Use only the latest run for each combination
    for (corruption_type, severity), run_info in sorted(latest_runs.items()):
        results['corruption_type'].append(corruption_type)
        results['severity'].append(severity)
        results['accuracy'].append(run_info['accuracy'])
        results['run_name'].append(run_info['run_name'])
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Print data validation information
    print("\nData Validation:")
    print("-" * 50)
    
    # Count runs per combination
    combination_counts = df.groupby(['corruption_type', 'severity']).size()
    if (combination_counts > 1).any():
        print("\nWarning: Multiple runs found for some combinations:")
        print(combination_counts[combination_counts > 1])
    
    # Check unique corruption types and severities
    unique_corruptions = sorted(df['corruption_type'].unique())
    unique_severities = sorted(df['severity'].unique())
    
    print(f"\nFound {len(unique_corruptions)} corruption types:")
    for c in unique_corruptions:
        severities = sorted(df[df['corruption_type'] == c]['severity'].unique())
        print(f"- {c}: severities {severities}")
    
    print(f"\nFound {len(unique_severities)} severity levels: {unique_severities}")
    
    # Check for missing combinations
    expected_corruptions = set([
        'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
        'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
        'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
    ])
    
    missing_corruptions = expected_corruptions - set(unique_corruptions)
    if missing_corruptions:
        print("\nMissing corruption types:")
        for c in sorted(missing_corruptions):
            print(f"- {c}")
    
    # Check for unexpected corruption types
    unexpected_corruptions = set(unique_corruptions) - expected_corruptions
    if unexpected_corruptions:
        print("\nUnexpected corruption types:")
        for c in sorted(unexpected_corruptions):
            print(f"- {c}")
    
    print("\nCreated DataFrame:")
    print(df)
    
    # Create pivot table with the latest runs
    matrix = df.pivot_table(
        index='corruption_type',
        columns='severity',
        values='accuracy',
        aggfunc='first'  # Take the first value since we've already selected the latest runs
    )
    
    # Ensure all severities from 1 to 5 are present
    for i in range(1, 6):
        if i not in matrix.columns:
            matrix[i] = np.nan
    matrix = matrix.sort_index(axis=1)
    
    print("\nPivoted Matrix:")
    print(matrix)
    
    return matrix

def plot_heatmap(matrix, output_path='cifar10c_results.png'):
    """Create and save a heatmap visualization."""
    if matrix.empty:
        raise ValueError("Cannot create heatmap from empty matrix")
    
    # Set style to a built-in clean style
    plt.style.use('default')
    
    # Set the font family and size
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
    })
    
    # Define colors
    colors = {
        'text': '#2C3E50',  # Dark gray
        'grid': '#EAECEE'   # Light gray
    }
        
    plt.figure(figsize=(12, 8), dpi=300)
    
    # Create mask for NaN values
    mask = np.isnan(matrix)
    
    # Create heatmap with custom settings for better visualization
    sns.heatmap(matrix, 
                annot=True,  # Show values in cells
                fmt='.1f',   # Format as float with 1 decimal
                cmap='RdYlGn',  # Red (low) to Green (high)
                center=50,    # Center colormap at 50%
                vmin=0,      # Minimum accuracy
                vmax=100,    # Maximum accuracy
                cbar_kws={'label': 'Accuracy (%)', 'pad': 0.01},
                square=True,  # Make cells square
                linewidths=0.5,  # Add cell borders
                robust=True,  # Use robust quantiles for color scaling
                mask=mask,    # Mask NaN values
                annot_kws={'size': 9, 'weight': 'bold'})
    
    # Customize plot
    plt.title('CIFAR-10-C Performance Across Corruptions\nArchitecture: CLIP ViT-B/16', 
              pad=20, color=colors['text'], fontsize=14, fontweight='bold')
    plt.xlabel('Severity Level', labelpad=10, color=colors['text'], fontsize=12)
    plt.ylabel('Corruption Type', labelpad=10, color=colors['text'], fontsize=12)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=0, color=colors['text'])
    plt.yticks(rotation=0, color=colors['text'])
    
    # Set background colors
    plt.gca().set_facecolor('white')
    plt.gcf().patch.set_facecolor('white')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()

def plot_corruption_trends(matrix, output_path='cifar10c_trends.png'):
    """Create a line plot showing accuracy trends across severity levels."""
    if matrix.empty:
        raise ValueError("Cannot create trend plot from empty matrix")
    
    # Set style to a built-in clean style
    plt.style.use('default')
    
    # Set the font family and size
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
    })
    
    # Define theme colors
    theme_colors = {
        'text': '#2C3E50',  # Dark gray
        'grid': '#EAECEE'   # Light gray
    }
        
    plt.figure(figsize=(15, 8), dpi=300)
    
    # Create a color palette for different corruption types
    n_corruptions = len(matrix.index)
    color_palette = sns.color_palette('husl', n_colors=n_corruptions)
    
    # Plot trend for each corruption type with different colors and markers
    for idx, corruption in enumerate(matrix.index):
        values = matrix.loc[corruption]
        if not values.isna().all():  # Only plot if we have some non-NA values
            plt.plot(matrix.columns, values, 
                    marker='o', 
                    label=corruption, 
                    color=color_palette[idx], 
                    linewidth=2, 
                    markersize=8)
    
    # Customize plot
    plt.title('CIFAR-10-C Accuracy Trends by Corruption Type and Severity\nArchitecture: CLIP ViT-B/16', 
              pad=20, color=theme_colors['text'], fontsize=14, fontweight='bold')
    plt.xlabel('Severity Level', labelpad=10, color=theme_colors['text'], fontsize=12)
    plt.ylabel('Accuracy (%)', labelpad=10, color=theme_colors['text'], fontsize=12)
    
    # Enhance grid
    plt.grid(True, linestyle='--', alpha=0.3, color=theme_colors['grid'], zorder=0)
    
    # Set x-axis ticks to severity levels
    plt.xticks(matrix.columns, color=theme_colors['text'])
    plt.yticks(color=theme_colors['text'])
    
    # Add legend with better placement
    if plt.gca().get_legend_handles_labels()[0]:
        legend = plt.legend(bbox_to_anchor=(1.05, 1), 
                          loc='upper left', 
                          borderaxespad=0,
                          frameon=True,
                          edgecolor='black',
                          fontsize=10)
        legend.get_frame().set_facecolor('white')
    
    # Set background colors
    plt.gca().set_facecolor('white')
    plt.gcf().patch.set_facecolor('white')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()

def plot_average_performance(matrix, output_path='cifar10c_averages.png'):
    """Create a bar plot showing average performance across severity levels."""
    if matrix.empty:
        raise ValueError("Cannot create average performance plot from empty matrix")
    
    # Set style to a built-in clean style
    plt.style.use('default')
    
    # Set the font family and size
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
    })
    
    # Define theme colors
    theme_colors = {
        'text': '#2C3E50',  # Dark gray
        'grid': '#EAECEE',  # Light gray
        'bar': '#3498DB'    # Blue
    }
    
    # Increase the figure size (width, height)
    plt.figure(figsize=(12, 6), dpi=300)  # Increased height from 8 to 10
    
    # Calculate means and standard deviations across severity levels
    means = matrix.mean(axis=1)
    stds = matrix.std(axis=1)
    
    # Sort corruptions by mean performance
    sorted_indices = means.argsort()
    sorted_means = means.iloc[sorted_indices]
    sorted_stds = stds.iloc[sorted_indices]
    
    # Create bar plot
    x = np.arange(len(sorted_means))
    bars = plt.bar(x, sorted_means, 
                  yerr=sorted_stds,
                  capsize=5,
                  color=theme_colors['bar'],
                  alpha=0.8,
                  edgecolor='white',
                  linewidth=1)
    
    # Customize plot
    plt.title('Average CIFAR-10-C Performance by Corruption Type\nArchitecture: CLIP ViT-B/16', 
              pad=20, color=theme_colors['text'], fontsize=14, fontweight='bold')
    plt.xlabel('Corruption Type', labelpad=10, color=theme_colors['text'], fontsize=12)
    plt.ylabel('Average Accuracy (%)', labelpad=10, color=theme_colors['text'], fontsize=12)
    
    # Set y-axis limit from 0 to 100
    plt.ylim(0, 100)
    
    # Rotate x-axis labels for better readability
    plt.xticks(x, sorted_means.index, rotation=45, ha='right', color=theme_colors['text'])
    plt.yticks(color=theme_colors['text'])
    
    # Add value labels on the bars, closer to the bars
    for i, v in enumerate(sorted_means):
        plt.text(i, v + sorted_stds.iloc[i] * 0.5, f'{v:.1f}%',  # Adjusted offset to be closer
                ha='center', va='bottom', 
                color=theme_colors['text'],
                fontsize=10, 
                fontweight='bold')
    
    # Enhance grid
    plt.grid(True, linestyle='--', alpha=0.3, color=theme_colors['grid'], zorder=0)
    
    # Set background colors
    plt.gca().set_facecolor('white')
    plt.gcf().patch.set_facecolor('white')
    
    # Add horizontal line at mean performance
    overall_mean = means.mean()
    plt.axhline(y=overall_mean, color='red', linestyle='--', alpha=0.5)
    
    # Adjust the vertical position of the overall mean text to avoid clashing
    plt.text(len(means) - 1, overall_mean + 20,  # Increased offset to 2
            f'Overall Mean: {overall_mean:.1f}%',
            ha='right', va='bottom',
            color='red',
            fontsize=10,
            fontweight='bold')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()

def generate_summary_statistics(matrix):
    """Generate and print summary statistics."""
    print("\nSummary Statistics for CLIP ViT-B/16:")
    print("-" * 50)
    
    if matrix.empty:
        print("\nNo data available for statistics")
        return
        
    # Overall statistics
    mean_acc = matrix.mean().mean()
    std_dev = matrix.std().std()
    print(f"Overall Mean Accuracy: {mean_acc:.2f}%" if not np.isnan(mean_acc) else "Overall Mean Accuracy: No data")
    print(f"Overall Std Deviation: {std_dev:.2f}%" if not np.isnan(std_dev) else "Overall Std Deviation: No data")
    
    # Best and worst performing corruptions
    mean_by_corruption = matrix.mean(axis=1)
    if not mean_by_corruption.empty and not mean_by_corruption.isna().all():
        best_corruption = mean_by_corruption.idxmax()
        worst_corruption = mean_by_corruption.idxmin()
        print(f"\nBest performing corruption: {best_corruption} ({mean_by_corruption.max():.2f}%)")
        print(f"Worst performing corruption: {worst_corruption} ({mean_by_corruption.min():.2f}%)")
        
        # Print all corruptions sorted by performance
        print("\nCorruptions ranked by average performance:")
        for corruption, acc in mean_by_corruption.sort_values(ascending=False).items():
            print(f"{corruption:20s}: {acc:.2f}%")
    else:
        print("\nNo corruption performance data available")
    
    # Impact of severity
    mean_by_severity = matrix.mean()
    if not mean_by_severity.empty and not mean_by_severity.isna().all():
        print(f"\nAccuracy by severity level:")
        for severity, acc in mean_by_severity.items():
            if not np.isnan(acc):
                print(f"Severity {severity}: {acc:.2f}%")
        
        # Calculate severity impact
        if len(mean_by_severity) > 1:
            severity_impact = mean_by_severity.max() - mean_by_severity.min()
            print(f"\nSeverity Impact (max - min): {severity_impact:.2f}%")
    else:
        print("\nNo severity level data available")

def main():
    """Main function to generate all visualizations and statistics."""
    try:
        print("Fetching results from wandb...")
        matrix = create_results_matrix()
        
        if matrix.empty:
            print("No data available to visualize")
            return
            
        print("\nGenerating visualizations...")
        plot_heatmap(matrix)
        plot_corruption_trends(matrix)
        plot_average_performance(matrix)
        
        print("\nGenerating statistics...")
        generate_summary_statistics(matrix)
        
        print("\nVisualization complete! Check:")
        print("1. cifar10c_results.png (heatmap)")
        print("2. cifar10c_trends.png (severity trends)")
        print("3. cifar10c_averages.png (average performance)")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nPlease check that:")
        print("1. You're logged into wandb")
        print("2. Your wandb project contains CIFAR-10-C runs")
        print("3. The runs have completed and contain accuracy metrics")

if __name__ == "__main__":
    main() 