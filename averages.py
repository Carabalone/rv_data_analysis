import pandas as pd
import numpy as np
import sys

def parse_position(position):
    if pd.isna(position):
        return (None, None, None)
    x, y, z = map(float, position.split())
    return (x, y, z)

def calculate_accuracy_df(df):
    # Calculate Accuracy
# Count "Destroyed" and "Missed target" actions
    destroyed_counts = df[df['InitiatedAction'] == 'Destroyed'].groupby(['AimStyle', 'Gun']).size().reset_index(name='destroyed')
    missed_counts = df[df['InitiatedAction'] == 'Missed target'].groupby(['AimStyle', 'Gun']).size().reset_index(name='missed')

# Combine counts
    accuracy_df = pd.merge(destroyed_counts, missed_counts, on=['AimStyle', 'Gun'], how='outer').fillna(0)

# Calculate accuracy
    accuracy_df['accuracy'] = accuracy_df['destroyed'] / (accuracy_df['destroyed'] + accuracy_df['missed'])

    print("Average Accuracy by AimStyle and Gun:")
    print(accuracy_df[['AimStyle', 'Gun', 'accuracy']])

def calculate_distance_df(df):
    # Function to parse Position into x, y, z
    def parse_position(position):
        if pd.isna(position):
            return (None, None, None)
        try:
            x, y, z = map(float, position.split())
            return (x, y, z)
        except ValueError:
            return (None, None, None)
    
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
                    'distance': distance
                })
    
    # Create a DataFrame for distances
    distance_df = pd.DataFrame(distances)
    
    # Calculate average distance per AimStyle and Gun
    avg_distance = distance_df.groupby(['AimStyle', 'Gun'])['distance'].mean().reset_index()
    
    print("\nAverage Distance to Bullseye by AimStyle and Gun:")
    print(avg_distance)
    distance_df.to_csv('tmp/distance_df.csv')
    
    return distance_df

def main():
    if len(sys.argv) < 2:
        print("provide file name")
        exit(0)

    filename = sys.argv[1]

    df = pd.read_csv(filename)

    df[['x', 'y', 'z']] = df['Position'].apply(parse_position).apply(pd.Series)

    calculate_accuracy_df(df)

    calculate_distance_df(df)


    # df['distance'] = df.apply(lambda row: (row.x**2 + row.y**2 + row.z**2)**0.5 if pd.notna(row.x) else None, axis=1)
    #
    # df['ElapsedTime'] = pd.to_numeric(df['ElapsedTime'], errors='coerce')
    #
    # df['relative_time'] = df.groupby('TestID')['ElapsedTime'].transform(lambda x: x - x.min())
    #
    # # Filter rows where action is "Destroyed"
    # destroyed = df[df['InitiatedAction'] == 'Destroyed']
    #
    # # Group by AimStyle, Gun, and TestID, then calculate mean relative_time
    # avg_time_per_task = destroyed.groupby(['AimStyle', 'Gun'])['relative_time'].mean().reset_index()
    #
    # # Rename the columns for clarity
    # avg_time_per_task.rename(columns={'relative_time': 'avg_time_to_destroy'}, inplace=True)
    #
    # print(df[(df['Gun'] == "RIFLE") & (df['AimStyle'] == 'LINE')])
    #
    # print("Average Time per Task by AimStyle and Gun:")
    # print(avg_time_per_task)
    #
    #



if __name__ == '__main__':
    main()

