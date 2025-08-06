import pandas as pd
import glob
import os


folder_path = './Data/statistics/journals'  # replace with your folder path


csv_files = glob.glob(os.path.join(folder_path, '*.csv'))


df_list = []
for file in csv_files:
    
    df = pd.read_csv(file)
    df_list.append(df)


merged_df = pd.concat(df_list, ignore_index=True)


output_path = os.path.join(folder_path, 'merged_output.csv')
merged_df.to_csv(output_path, index=False)