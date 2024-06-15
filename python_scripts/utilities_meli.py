
import pandas as pd
import numpy as np

# Step 1: Clean the warranty column
def clean_warranty(warranty):
    if pd.isnull(warranty):
        return 'missing'
    # Normalize different variations of the same term
    warranty = warranty.lower()
    if 'mes' in warranty:
        if '3' in warranty:
            return '3 meses'
        elif '6' in warranty:
            return '6 meses'
        elif '12' in warranty or '1 año' in warranty:
            return '12 meses'
        else:
            return 'otros'
    elif 'año' in warranty:
        return '12 meses'
    elif 'si' in warranty:
        return 'sí'
    elif 'sin garantía' in warranty:
        return 'sin garantía'
    elif 'missing' in warranty:
        return 'missing'
    else:
        return 'otros'

# Step 1: Define a function to extract the first two words
def extract_first_two_words(title):
    if pd.isnull(title) or title == '':
        return ''
    words = title.split()
    return ' '.join(words[:2])

    # Step 1: Define a function to extract the first word
def extract_first_word(title):
    if pd.isnull(title) or title == '':
        return ''
    return title.split()[0]

def calculate_group_stats(df, group_column, value_column):
    """
    Calculate multiple statistics (mean, min, max, var, median, std, q1, q3) 
    for a given column grouped by another column in a DataFrame.

    Parameters:
    - df: DataFrame containing the data.
    - group_column: Name of the column to group by.
    - value_column: Name of the column for which statistics are calculated.

    Returns:
    - DataFrame with additional columns for each statistic calculated.
    """
    grouped = df.groupby(group_column)[value_column]
    new_columns = {
        'mean_' + value_column + '_' + group_column: grouped.transform('mean'),
        'min_' + value_column + '_' + group_column: grouped.transform('min'),
        'max_' + value_column + '_' + group_column: grouped.transform('max'),
        'var_' + value_column + '_' + group_column: grouped.transform('var'),
        'median_' + value_column + '_' + group_column: grouped.transform('median'),
        'std_' + value_column + '_' + group_column: grouped.transform('std'),
        'q1_' + value_column + '_' + group_column: grouped.transform(lambda x: x.quantile(0.25)),
        'q3_' + value_column + '_' + group_column: grouped.transform(lambda x: x.quantile(0.75))
    }
    
    # Concatenate all new columns to the original DataFrame
    df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
    
    return df


def replace_rare_values(df, column, rare_values):
    """
    Replace rare values in a DataFrame column with a specified value.

    Parameters:
    - df: DataFrame containing the data.
    - column: Name of the column in which to replace values.
    - rare_values: List or tuple of rare values to be replaced.

    Returns:
    - DataFrame with replaced values.
    """
    for value in rare_values:
        df[column] = np.where(df[column] == value, 1, df[column])
    
    return df

def create_transformed_columns(df, column):
    """
    Create new columns in a DataFrame based on transformations (square, square root, log) of a specified column.

    Parameters:
    - df: DataFrame containing the data.
    - column: Name of the column for which transformations are applied.

    Returns:
    - DataFrame with additional columns for square, square root, and log transformations.
    """
    # Square transformation
    df[f'{column}_square'] = df[column] ** 2

    df[f'{column}_cube'] = df[column] ** 3
    
    # Square root transformation
    df[f'{column}_sqrt'] = np.sqrt(df[column])
    
    # Logarithm transformation (handling non-positive values)
    df[f'{column}_log'] = np.log(df[column].replace(0, np.nan))  # Replace 0 with NaN to avoid log(0) errors
    
    return df