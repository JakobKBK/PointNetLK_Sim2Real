import sklearn
import pandas as pd
import pickle
import os

data_directory: str = r'C:\MaH_Kretschmer\data\train_data\DEMO_EMO'


def main():

    # load dictionary for labeling
    with open(f'{data_directory}/dictionary.pkl', 'rb') as f:
        part_dict = pickle.load(f)

    # create lists for labels and filepaths
    labels_lst = []
    paths_lst = []

    # create dataframe storing filepaths and labels
    data_df = pd.DataFrame(columns=['path', 'lbl'])

    for label in part_dict:
        part = part_dict[label]
        part_folder = f'{data_directory}/clouds/{part}'

        files = os.listdir(part_folder)

        for file in files:
            if file.endswith('.ply'):
                path_cloud = os.path.join(part_folder, file)
                labels_lst.append(label)
                paths_lst.append(path_cloud)

    data_df['path'] = paths_lst
    data_df['lbl'] = labels_lst
    data_df.to_pickle(f'{data_directory}/data_df.pkl')


if __name__ == '__main__':
    main()
