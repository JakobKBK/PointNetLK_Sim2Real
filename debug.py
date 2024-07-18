import pandas as pd

def update_paths(df, row_name, old_path, new_path):
    """
    Replaces a substring in all the paths within a specific column of a pandas DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the paths.
        row_name (str): The row in the DataFrame that contains the paths as strings.
        old_path (str): The substring within the paths that needs to be replaced.
        new_path (str): The new substring that will replace the old substring.

    Returns:
        pd.DataFrame: A DataFrame with the paths in the specified column updated.
    """
    # Check if the specified row exists in the DataFrame
    if row_name in df.index:
        for column in df.columns:
            if isinstance(df.at[row_name, column], str):  # Check if the cell contains a string
                old_str = df.at[row_name, column]
                df.at[row_name, column] = df.at[row_name, column].replace(old_path, new_path)
                print(f'Changed path {old_str} to {df.at[row_name, column]}')
    else:
        raise ValueError(f"The row '{row_name}' does not exist in the DataFrame.")

    return df

row_names = ['path', 'voxel_size', 'labels']
df_paths = pd.DataFrame(index=row_names)
df = pd.read_pickle(r'D:\data\train_data\DEMO_EMO\data_df0.pkl')
print(df)
df_paths = pd.concat([df_paths, df.loc[row_names]], axis=1)
df_paths = update_paths(df_paths, 'path', 'C:\\MaH_Kretschmer\\data\\train_data\\DEMO_EMO', r'D:\data\train_data\DEMO_EMO')

