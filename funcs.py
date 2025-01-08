import pandas as pd
import numpy as np
import re




def categorize_motion(action):
    if 'Cube' in action:
        return 'STATIC_MOTION'
    elif 'linear motion' in action:
        return 'LINEAR_MOTION'
    elif 'circular motion' in action:
        return 'CIRCULAR_MOTION'
    else:
        return 'NO_TARGET'

def normalize_cube(row):
    pattern = r'Cube \(\d+\)'

    output_string = str(row)

    if ("Cube" in row):
        output_string = re.sub(pattern, 'STATIC_TARGET', output_string)

    return output_string

def extract_position(action_data):
    if isinstance(action_data, str) and '(' in action_data:
        start = action_data.find('(')
        end = action_data.find(')')
        if start != -1 and end != -1:
            return action_data[start+1:end]
    return np.nan
def match_aim_type(aim_type):
    if (aim_type == '0'):
        return 'NO_AIM'
    elif (aim_type == '1'):
        return 'LINE'
    if (aim_type == '2'):
        return 'SPHERE'

def extract_gun_aim(action_data):
    if isinstance(action_data, str) and 'with' in action_data:
        parts = action_data.split('with ')
        gun_aim = parts[1].split(' using ')

        gun = gun_aim[0].strip().split(" ")[0].upper()
        aim_type = match_aim_type(gun_aim[1].strip().split(' ')[2])

        return gun, aim_type
    return np.nan, np.nan

def timestamp_to_seconds(ts):
    minutes, seconds = map(float, ts.split(':'))
    return minutes * 60 + seconds
