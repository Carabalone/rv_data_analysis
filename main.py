import pandas as pd
import numpy as np
import sys
import os
from scipy.stats import f_oneway, shapiro, kruskal
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import seaborn as sns
import matplotlib.pyplot as plt
from media import *
from statsmodels.stats.multitest import multipletests
from scikit_posthocs import posthoc_dunn
from cliffs_delta import cliffs_delta
import itertools

session = 1

def parse_position(position):
    if pd.isna(position):
        return (None, None, None)
    x, y, z = map(float, position.split())
    return (x, y, z)

def calculate_accuracy_df(df):
# Create a new DataFrame with Gun, AimStyle, and Hit (True/False)
    hit_df = df[df['InitiatedAction'].isin(['Hit', 'Missed target'])].copy()
    hit_df['Hit'] = hit_df['InitiatedAction'] == 'Hit'
    hit_df = hit_df[['Gun', 'AimStyle', 'Hit', 'UniqueTestID']]

    global session

    # Calculate Accuracy
    destroyed_counts = df[df['InitiatedAction'] == 'Destroyed'].groupby(['AimStyle', 'Gun']).size().reset_index(name='destroyed')
    missed_counts = df[df['InitiatedAction'] == 'Missed target'].groupby(['AimStyle', 'Gun']).size().reset_index(name='missed')

    # Combine counts
    accuracy_df = pd.merge(destroyed_counts, missed_counts, on=['AimStyle', 'Gun'], how='outer').fillna(0)

    # Calculate accuracy
    accuracy_df['accuracy'] = accuracy_df['destroyed'] / (accuracy_df['destroyed'] + accuracy_df['missed'])
    accuracy_df['session'] = session

    # print("Average Accuracy by AimStyle and Gun:")
    # print(accuracy_df[['AimStyle', 'Gun', 'accuracy']])
    return hit_df, accuracy_df

def calculate_distance_df(df):
    # Apply the function and create new columns
    df[['x', 'y', 'z']] = df['Position'].apply(parse_position).apply(pd.Series)
    
    # Filter the DataFrame to include only "Hit" and "Destroyed" actions
    hit_destroyed = df[df['InitiatedAction'].isin(['Hit', 'Destroyed'])]
    
    # Group by TestID and Timestamp
    groups = hit_destroyed.groupby(['TestID', 'Timestamp'])
    
    # Initialize a list to hold distances
    distances = []
    
    for name, group in groups:
        # Check if there is exactly one "Hit" and one "Destroyed"
        if len(group) == 2:
            actions = group['InitiatedAction'].tolist()
            if 'Hit' in actions and 'Destroyed' in actions:
                # Get the positions
                hit_row = group[group['InitiatedAction'] == 'Hit']
                destroyed_row = group[group['InitiatedAction'] == 'Destroyed']
                # Extract x, y, z
                x_hit, y_hit, z_hit = hit_row[['x', 'y', 'z']].values[0]
                x_destroyed, y_destroyed, z_destroyed = destroyed_row[['x', 'y', 'z']].values[0]
                # Check for missing values
                if None in [x_hit, y_hit, z_hit, x_destroyed, y_destroyed, z_destroyed]:
                    continue
                # Calculate distance
                distance = np.sqrt((x_hit - x_destroyed)**2 + (y_hit - y_destroyed)**2 + (z_hit - z_destroyed)**2)
                # Append to distances list
                distances.append({
                    'AimStyle': hit_row['AimStyle'].iloc[0],
                    'Gun': hit_row['Gun'].iloc[0],
                    'distance': distance,
                    'pos_hit': (x_hit, y_hit, z_hit),
                    'pos_destroyed': (x_destroyed, y_destroyed, z_destroyed),
                    'UniqueTestID': group['UniqueTestID'] 
                })
    
    # Create a DataFrame for distances
    distance_df = pd.DataFrame(distances)
    
    # Calculate average distance per AimStyle and Gun
    avg_distance = distance_df.groupby(['AimStyle', 'Gun'])['distance'].mean().reset_index()
    
    # print("\nAverage Distance to Bullseye by AimStyle and Gun:")
    # print(avg_distance)
    return distance_df, avg_distance

def calculate_avg_time_to_destroy_df(df):
    # Initialize a list to store time differences
    time_differences = []

    # Iterate over each test (TestID)
    for test_id in df['TestID'].unique():
        # Filter the DataFrame for the current test
        test_df = df[df['TestID'] == test_id]

        # Group by Timestamp and filter pairs of "Hit" and "Destroyed"
        grouped = test_df.groupby('Timestamp')
        pairs = []
        for timestamp, group in grouped:
            if len(group) == 2 and set(group['InitiatedAction']) == {'Hit', 'Destroyed'}:
                pairs.append(group)

        # Sort pairs by ElapsedTime
        pairs.sort(key=lambda x: x['ElapsedTime'].iloc[0])

        # Use the first pair as a baseline (skip it for time calculation)
        for i in range(1, len(pairs)):
            # Calculate the time difference between the current pair and the previous pair
            current_time = pairs[i]['ElapsedTime'].iloc[0]
            previous_time = pairs[i - 1]['ElapsedTime'].iloc[0]
            time_diff = current_time - previous_time

            # Append the time difference along with Gun and AimStyle
            gun = pairs[i]['Gun'].iloc[0]
            aim_style = pairs[i]['AimStyle'].iloc[0]
            time_differences.append({
                'Gun': gun,
                'AimStyle': aim_style,
                'TimeDiff': time_diff,
                'UniqueTestID': pairs[i]['UniqueTestID']
            })

    # Create a DataFrame from the time differences
    time_diff_df = pd.DataFrame(time_differences)

    # Calculate the average time difference per Gun and AimStyle combo
    avg_time_df = time_diff_df.groupby(['Gun', 'AimStyle'])['TimeDiff'].mean().reset_index()

    # print(f"Average Time to Destroy:\n{avg_time_df}")
    return time_diff_df, avg_time_df

def main():
    if len(sys.argv) < 2:
        print("Provide directory containing session files")
        exit(0)

    directory = sys.argv[1]

    # Initialize empty DataFrames to store combined data
    combined_hit_df = pd.DataFrame()
    combined_distance_df = pd.DataFrame()
    combined_time_diff_df = pd.DataFrame()
    combined_accuracy_df = pd.DataFrame()

    # Iterate over all session files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            print(f"Processing {filename}...")

            global session
            session = int(filename.replace(".csv", "").replace("session", ""))

            # Load the session data
            df = pd.read_csv(filepath)

            df['UniqueTestID'] = filename.replace(".csv", "").replace("session", "") + '_' + df['TestID'].astype(str)

            df[['x', 'y', 'z']] = df['Position'].apply(parse_position).apply(pd.Series)

            # Calculate metrics for the current session
            hit_df, accuracy_df = calculate_accuracy_df(df)
            distance_df, avg_distance_df = calculate_distance_df(df)
            time_diff_df, avg_time_df = calculate_avg_time_to_destroy_df(df)

            # Append the data to the combined DataFrames
            combined_accuracy_df = pd.concat([combined_accuracy_df, accuracy_df], ignore_index=True)
            combined_hit_df = pd.concat([combined_hit_df, hit_df], ignore_index=True)
            combined_distance_df = pd.concat([combined_distance_df, distance_df], ignore_index=True)
            combined_time_diff_df = pd.concat([combined_time_diff_df, time_diff_df], ignore_index=True)

    # Aggregate combined data
    combined_accuracy_avg = combined_accuracy_df.groupby(['AimStyle', 'Gun'])['accuracy'].mean().reset_index()
    combined_distance_avg = combined_distance_df.groupby(['AimStyle', 'Gun'])['distance'].mean().reset_index()
    combined_time_avg = combined_time_diff_df.groupby(['AimStyle', 'Gun'])['TimeDiff'].mean().reset_index()

    # create_heatmaps(combined_distance_df)

    # create_box_plot(combined_time_diff_df)

    # Group accuracies by AimStyle
    data = combined_accuracy_df
    line_accuracies = data[data['AimStyle'] == 'LINE']['accuracy']
    sphere_accuracies = data[data['AimStyle'] == 'SPHERE']['accuracy']
    no_aim_accuracies = data[data['AimStyle'] == 'NO_AIM']['accuracy']

    line_accuracies = combined_accuracy_df[combined_accuracy_df['AimStyle'] == 'LINE']['accuracy']

    no_aim_accuracies = combined_accuracy_df[combined_accuracy_df['AimStyle'] == 'NO_AIM']['accuracy']

    sphere_accuracies = combined_accuracy_df[combined_accuracy_df['AimStyle'] == 'SPHERE']['accuracy']
    shapiro_results = {
        'LINE': shapiro(line_accuracies),
        'NO_AIM': shapiro(no_aim_accuracies),
        'SPHERE': shapiro(sphere_accuracies)
    }
    print("Shapiro-Wilk test for accuracies: ")
    for k, v in shapiro_results.items():
        print(f"{k}: {v}")

    print("\n")

# Perform Kruskal-Wallis test
    stat, p_value = kruskal(line_accuracies, sphere_accuracies, no_aim_accuracies)

    print("We can conclude that the data is not normally distributed \n")

    print(f"Kruskal-Wallis H-statistic: {stat}")
    print(f"P-value: {p_value}")

    if p_value < 0.05:
        print("Reject the null hypothesis: At least one aim style is significantly different.")
        posthoc = posthoc_dunn(data, val_col='accuracy', group_col='AimStyle', p_adjust='bonferroni')

        print(f"posthoc dunn:\n{posthoc}")

# Calculate Cliff's delta for all pairwise comparisons
        aim_styles = {
            'LINE': line_accuracies,
            'NO_AIM': no_aim_accuracies,
            'SPHERE': sphere_accuracies
        }

        cliffs_results = {}
        for (aim1, acc1), (aim2, acc2) in itertools.combinations(aim_styles.items(), 2):
            delta, size = cliffs_delta(acc1, acc2)
            cliffs_results[f"{aim1} vs {aim2}"] = (delta, size)

        print("Cliffs Delta results:")
        for k, v in cliffs_results.items():
            print(f"{k}: {v}")
    else:
        print("Fail to reject the null hypothesis: No significant difference between aim styles.")



if __name__ == '__main__':
    main()
