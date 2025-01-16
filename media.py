import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_heatmaps(distance_df):
    # Iterate over each combination of Gun and AimStyle
    for (gun, aim_style), group in distance_df.groupby(['Gun', 'AimStyle']):
        # Extract x and y coordinates for hits and destroyed positions
        x_hits = []
        y_hits = []
        x_destroyed = []
        y_destroyed = []

        for _, row in group.iterrows():
            x_hit, y_hit, _ = row['pos_hit']
            x_destroy, y_destroy, _ = row['pos_destroyed']
            x_hits.append(x_hit)
            y_hits.append(y_hit)
            x_destroyed.append(x_destroy)
            y_destroyed.append(y_destroy)

        # Normalize the coordinates so that the destroyed position is at (0, 0)
        x_hits_normalized = np.array(x_hits) - np.array(x_destroyed)
        y_hits_normalized = np.array(y_hits) - np.array(y_destroyed)

        # Create a 2D histogram (heatmap)
        plt.figure(figsize=(8, 6))
        sns.kdeplot(x=x_hits_normalized, y=y_hits_normalized, cmap="Reds", fill=True)
        
        # Add a circle to represent the target
        circle = plt.Circle((0, 0), radius=1, color='blue', fill=False, linestyle='--', linewidth=2)
        plt.gca().add_patch(circle)

        # Set plot title and labels
        plt.title(f'Heatmap of Hits for {gun} ({aim_style})')
        plt.xlabel('X Position (Normalized)')
        plt.ylabel('Y Position (Normalized)')
        plt.xlim(-2, 2)  # Adjust limits as needed
        plt.ylim(-2, 2)  # Adjust limits as needed
        plt.grid(True)
        plt.show()

def create_box_plot(time_df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=time_df, x='Gun', y='TimeDiff', hue='AimStyle')
    plt.title('Distribution of Time Differences by Gun and Aim Style (0th to 98th percentile)')
    plt.xlabel('Gun')
    plt.ylabel('Time Difference (s)')
    plt.legend(title='Aim Style')

    plt.ylim(time_df['TimeDiff'].quantile(0.00), time_df['TimeDiff'].quantile(0.98))

    plt.show()
