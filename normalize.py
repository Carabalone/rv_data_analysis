import sys
import pandas as pd
import numpy as np
from funcs import *

def main():
    if (len(sys.argv) < 2):
        print("provide the file name.")
        exit(-1)

    file_name = sys.argv[1]
    
    df = pd.read_csv(file_name)

    df['ActionData'] = df['ActionData'].str.replace('cricular', 'circular')
    df['ActionData'] = df['ActionData'].apply(str)

    # get the target Type
    df['TargetType'] = df['ActionData'].apply(categorize_motion)
    df['ActionData'] = df['ActionData'].apply(normalize_cube)

    # extract positions
    df['Position'] = df['ActionData'].apply(extract_position)

    df.loc[df['InitiatedAction'] == 'Missed target', 'Position'] = np.nan

    # Create 'Gun' and 'AimStyle' columns filled with NaN
    df['Gun'] = np.nan
    df['AimStyle'] = np.nan

# Apply the function only to 'Hit' actions and assign back
    hits = df[df['InitiatedAction'] == 'Hit']
    gun_aim = hits['ActionData'].apply(extract_gun_aim)
    df.loc[df['InitiatedAction'] == 'Hit', ['Gun', 'AimStyle']] = list(gun_aim)

    # Create a dictionary to map Timestamp to Gun and AimStyle
    gun_aim_map = (
        df.loc[df['InitiatedAction'] == 'Hit', ['Timestamp', 'Gun', 'AimStyle']]
        .set_index('Timestamp')
        .to_dict(orient='index')
    )

# Map Gun and AimStyle to Destroyed rows
    df.loc[df['InitiatedAction'] == 'Destroyed', 'Gun'] = (
        df.loc[df['InitiatedAction'] == 'Destroyed', 'Timestamp']
        .map(lambda x: gun_aim_map.get(x, {}).get('Gun', np.nan))
    )

    df.loc[df['InitiatedAction'] == 'Destroyed', 'AimStyle'] = (
        df.loc[df['InitiatedAction'] == 'Destroyed', 'Timestamp']
        .map(lambda x: gun_aim_map.get(x, {}).get('AimStyle', np.nan))
    )

    for idx, row in df.iterrows():
        if row['InitiatedAction'] == 'Missed target':
            try:
                next_destroyed = df.loc[idx + 1:, 'InitiatedAction'].eq('Destroyed').idxmax()
                if next_destroyed != idx:
                    df.at[idx, 'Gun'] = df.at[next_destroyed, 'Gun']
                    df.at[idx, 'AimStyle'] = df.at[next_destroyed, 'AimStyle']
                    df.at[idx, 'TargetType'] = df.at[next_destroyed, 'TargetType']
            except:
                df.at[idx, 'Gun'] = 'UKNOWN'
                df.at[idx, 'AimStyle'] = 'UKNOWN'
                df.at[idx, 'TargetType'] = 'UKNOWN'

# Select relevant columns
    final_df = df[['Timestamp', 'InitiatedAction', 'TargetType', 'Position', 'Gun', 'AimStyle']]

# Rename columns for clarity
    final_df.rename(columns={'InitiatedAction': 'Action'}, inplace=True)

# Drop rows where Position is missing unless it's a "Missed target"
    final_df = df.dropna(subset=['Position'], how='any')

# Append rows where Action is "Missed target" using pd.concat
    final_df = pd.concat([final_df, df[df['InitiatedAction'] == 'Missed target']], ignore_index=True)

    final_df = final_df.sort_values(by='Timestamp')
    final_df = final_df.drop(columns=['ActionData'])


    final_df['TimestampSeconds'] = final_df['Timestamp'].apply(timestamp_to_seconds)

# Calculate time differences between consecutive rows
    final_df['TimeDiff'] = final_df['TimestampSeconds'].diff().fillna(0)

# Define the threshold for a new test (1 minute = 60 seconds)
    threshold = 60

# Assign TestID based on time jumps
    final_df['TestID'] = (final_df['TimeDiff'] > threshold).cumsum()

# Calculate testSeconds for each test
    final_df['ElapsedTime'] = final_df.groupby('TestID')['TimestampSeconds'].transform(lambda x: x - x.min())

# Drop intermediate columns (if needed)
    float_columns = final_df.select_dtypes(include=['float64']).columns
    final_df[float_columns] = final_df[float_columns].round(2)

    analysis_df = final_df[final_df['InitiatedAction'] != 'started session']

    print(analysis_df)

# Calculate accuracy as a ratio of hits to total actions per AimStyle
    accuracy_df = analysis_df.groupby('AimStyle')['InitiatedAction'].agg(['count', lambda x: (x == 'Hit').sum()]).reset_index()
    accuracy_df = accuracy_df.rename(columns={'<lambda_0>': 'Hits'})

# Calculate Accuracy
    accuracy_df['Accuracy'] = accuracy_df['Hits'] / accuracy_df['count']

# Round the Accuracy column to 2 decimal places
    accuracy_df['Accuracy'] = accuracy_df['Accuracy'].round(2)

# Drop intermediate columns (if needed)
    final_df = final_df.drop(columns=['TimestampSeconds', 'TimeDiff'])

# Calculate accuracy_df: Number of Hits and Total Actions per AimStyle
    accuracy_df = (
        final_df[final_df['InitiatedAction'] == 'Hit']
        .groupby('AimStyle')
        .size()
        .reset_index(name='Hits')
    )

# Calculate total actions per AimStyle
    total_actions = final_df.groupby('AimStyle').size().reset_index(name='TotalActions')

# Merge Hits and TotalActions
    accuracy_df = pd.merge(accuracy_df, total_actions, on='AimStyle', how='left')

# Calculate Accuracy
    accuracy_df['Accuracy'] = accuracy_df['Hits'] / accuracy_df['TotalActions']

# Round Accuracy to 2 decimal places
    accuracy_df['Accuracy'] = accuracy_df['Accuracy'].round(2)

# Calculate time_df: Average ElapsedTime per AimStyle
    time_df = (
        final_df[final_df['InitiatedAction'] == 'Hit']
        .groupby('AimStyle')['ElapsedTime']
        .mean()
        .reset_index(name='AvgElapsedTime')
    )

# Round AvgElapsedTime to 2 decimal places
    time_df['AvgElapsedTime'] = time_df['AvgElapsedTime'].round(2)

# Merge accuracy_df and time_df
    result_df = pd.merge(accuracy_df, time_df, on='AimStyle')

    final_df = final_df[['TestID', 'Timestamp', 'ElapsedTime', 'InitiatedAction', 'TargetType', 'Gun', 'AimStyle', 'Position']]

    final_df.to_csv('tmp/' + file_name.replace("raw_data", ""). replace("/",""), index=False)

    name = file_name.replace('.csv','').replace('DataLog_', '').replace("raw_data", "").replace("/", "")
    print(file_name)
    final_df.to_csv('tmp/final_df.csv', index=False)
    #result_df.to_csv('tmp/result_df.csv', index=False)

# Print confirmation
    print("DataFrames saved to /tmp directory")


if __name__ == "__main__":
    main()
