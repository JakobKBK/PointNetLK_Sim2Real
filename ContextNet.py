import torch
from sklearn.model_selection import train_test_split
import keras
from keras import layers
import tensorflow as tf
from pyntcloud import PyntCloud
import open3d as o3d
import argparse
import os
import pandas as pd
import numpy as np
import gc


def options(argv=None):

    parser = argparse.ArgumentParser(description='ContextNet')

    # io settings.
    parser.add_argument('--path', default=r'C:\MaH_Kretschmer\data\train_data\DEMO_EMO', type=str,
                        metavar='PATH', help='path to data directory')
    parser.add_argument('--outfile', type=str, default='./logs/220524_train_log',
                        metavar='BASENAME', help='output filename (prefix)')

    # settings for input data
    parser.add_argument('--data_type', default='synthetic', type=str,
                        metavar='DATASET', help='whether data is synthetic or real')
    parser.add_argument('--dictionaryfile', type=str, default=r'C:\MaH_Kretschmer\data\train_data\DEMO_EMO\dictionary.pkl',
                        metavar='PATH', help='path to the categories to be trained')
    parser.add_argument('--num_points', default=2048, type=int,
                        metavar='N', help='points in point-cloud.')
    parser.add_argument('--workers', default=1, type=int,
                        metavar='N', help='number of data loading workers')

    # settings for training.
    parser.add_argument('--batch_size', default=64, type=int,
                        metavar='N', help='mini-batch size')
    parser.add_argument('--max_iter', default=1e3, type=int,
                        metavar='N', help='max iter voxel down')
    parser.add_argument('--max_epochs', default=20, type=int,
                        metavar='N', help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int,
                        metavar='N', help='manual epoch number')
    parser.add_argument('--optimizer', default='Adam', type=str,
                        metavar='METHOD', help='name of an optimizer')
    parser.add_argument('--device', default='cuda:0', type=str,
                        metavar='DEVICE', help='use CUDA if available')
    parser.add_argument('--lr', type=float, default=1e-3,
                        metavar='D', help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=1e-4,
                        metavar='D', help='decay rate of learning rate')
    parser.add_argument('--train_on_single_masks', type=bool, default=True,
                        metavar='BOOL', help='wether to train solely on single part pointclouds')
    parser.add_argument('--path_model', type=str, default=r'C:\MaH_Kretschmer\data\models',
                        metavar='PATH', help='safe path for trained model .keras (prefix) & checkpoints')
    parser.add_argument('--leaky_slope', type=float, default=0.1,
                        metavar='D', help='slope for leaky relu activation function')

    # settings for log
    parser.add_argument('--logfile', default='', type=str,
                        metavar='LOGNAME', help='path to logfile')
    parser.add_argument('--resume', default='', type=str,
                        metavar='PATH', help='path to latest checkpoint')
    parser.add_argument('--pretrained', default='', type=str,
                        metavar='PATH', help='path to pretrained model file')

    args = parser.parse_args(argv)

    return args


def main(ARGS):

    train_df, test_df, n_classes, max_voxel = get_dataset()
    contextnet = get_model(ARGS, n_classes)
    model, history = train(ARGS, train_df, test_df, max_voxel, contextnet)
    model.save(f'{ARGS.path_model}\contextnet_v1.keras')


class CustomDataloader(keras.utils.Sequence):

    def __init__(self, data, dictionary, shuffle, ARGS):
        super().__init__()
        self.data = data.loc['path']
        self.lbls = data.loc['labels']
        self.voxel_size = data.loc['voxel_size']
        self.point_var_x = data.loc['point_variance_x']
        self.point_var_y = data.loc['point_variance_y']
        self.point_var_z = data.loc['point_variance_z']
        self.point_skew_x = data.loc['point_skew_x']
        self.point_skew_y = data.loc['point_skew_y']
        self.point_skew_z = data.loc['point_skew_z']
        self.relative_prop = data.loc['rel_proportion']
        self.density_cloud = data.loc['density']
        self.center_of_mass = data.loc['com']
        self.num_points_set = ARGS.num_points
        self.max_iter = ARGS.max_iter
        self.indexes = np.arange(len(data.columns))
        self.n_classes = len(dictionary)+1  # accounting for the additional 'background' or 'unlabeled' class
        self.shuffle = shuffle
        self.keys_lbl = np.array(list(dictionary.keys()))
        self.label_to_index = {label: index for index, label in enumerate(self.keys_lbl)}
        self.on_epoch_end()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        indexes = self.indexes[idx * ARGS.batch_size:(idx + 1) * ARGS.batch_size]
        data = np.empty((ARGS.batch_size, 36 * self.num_points_set + 11))
        label = np.empty((ARGS.batch_size, self.n_classes), dtype=int)

        for i, idx in enumerate(indexes):

            label_src = self.lbls.iloc[idx]
            path_dp = self.data.iloc[idx]
            voxel = self.voxel_size.iloc[idx]
            var_x = self.point_var_x.iloc[idx]
            var_y = self.point_var_y.iloc[idx]
            var_z = self.point_var_z.iloc[idx]
            skew_x = self.point_skew_x.iloc[idx]
            skew_y = self.point_skew_y.iloc[idx]
            skew_z = self.point_skew_z.iloc[idx]
            proportion = self.relative_prop.iloc[idx]
            density = self.density_cloud.iloc[idx]
            com = self.center_of_mass.iloc[idx]
            cloud_temp = PyntCloud.from_file(path_dp)
            dp = cloud_temp.points
            label_unique = np.unique(label_src)

            if np.random.rand() < 0.05:
                # Generate a random scale factor between 0 and 2
                scale_factor = 2 * np.random.rand()

                # Scale the point cloud
                dp *= scale_factor

            if len(dp) > self.num_points_set:
                dp = self.__voxel_down_sample_to_n(self, dp, voxel)

            if len(dp) < self.num_points_set:
                dp = self.__add_padding(self, dp)

            points = dp.to_numpy()
            fpfh, normals = self.get_fpfh(points, voxel)
            data[i, 0: 3 * self.num_points_set] = normals
            data[i, 3 * self.num_points_set: 36 * self.num_points_set] = fpfh
            data[i, 36 * self.num_points_set] = var_x
            data[i, 36 * self.num_points_set + 1] = var_y
            data[i, 36 * self.num_points_set + 2] = var_z
            data[i, 36 * self.num_points_set + 3] = skew_x
            data[i, 36 * self.num_points_set + 4] = skew_y
            data[i, 36 * self.num_points_set + 5] = skew_z
            data[i, 36 * self.num_points_set + 6] = proportion
            data[i, 36 * self.num_points_set + 7] = density
            data[i, 36 * self.num_points_set + 7:-1] = com
            label[i] = self.to_one_hot_enc_lbls(label_unique)

        return data, label

    def to_one_hot_enc_lbls(self, labels):
        get_index_or_negative_one = lambda label: self.label_to_index.get(label, -1)
        vectorized_function = np.vectorize(get_index_or_negative_one)
        label_indices = vectorized_function(labels)
        one_hot_labels = np.eye(self.n_classes)[label_indices+1]
        return one_hot_labels

    def get_fpfh(self, points, voxel):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        fpfh_o3d = self.preprocess_point_cloud(pcd, voxel)
        fpfh = np.asarray(fpfh_o3d.data)
        normals_points = np.asarray(pcd.normals)

        return fpfh.flatten(), normals_points.flatten()

    def preprocess_point_cloud(self, pcd, voxel_size):

        radius_normal = voxel_size * 2

        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = voxel_size * 5
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_fpfh

    def mask_unknown(self, array):
        mask = ~np.isin(array, list(np.array(list(self.label_to_index.values()))))
        array[mask] = -1

        return array

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indexes)

    @staticmethod
    def __add_padding(self, dp):
        n = self.num_points_set - len(dp)
        random_row = dp.sample(n, replace=True)
        dp = pd.concat([dp, random_row], ignore_index=True)
        return dp

    @staticmethod
    def __align_lbls(label, n_repeats):
        repeated_labels = label.repeat(n_repeats+1)

        return repeated_labels

    @staticmethod
    def __shorten_lbls(label, n_labels):
        shortened_labels = label[:n_labels]
        return shortened_labels


    @staticmethod
    def __voxel_down_sample_to_n(self, dp, voxel):
        initial_voxel = voxel
        current_iter = 0
        n = self.num_points_set
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(dp.values)

        while current_iter < self.max_iter:
            if current_iter == 0:
                voxel = initial_voxel
                if voxel < 1e-6:
                    voxel = 1e-6
                down_pcd = pcd.voxel_down_sample(voxel_size=voxel)
                num_points = len(down_pcd.points)

            if num_points == n:
                points = pd.DataFrame(np.asarray(down_pcd.points), columns=['x', 'y', 'z'])
                return points
            elif num_points > n:
                voxel = voxel + ((0.1*np.abs(n-num_points)*(np.log(0.1*np.abs(n-num_points))))*1e-5)  # adjust the step size as needed
            else:
                voxel = voxel - ((0.1*np.abs(n-num_points)*(np.log(0.1*np.abs(n-num_points))))*1e-5)
                if voxel <= 0:
                    voxel = 1e-4

            down_pcd = pcd.voxel_down_sample(voxel_size=voxel)
            num_points = len(down_pcd.points)
            current_iter += 1

        if num_points > n:
            #force drop if n not met within max iter
            array_points = np.asarray(down_pcd.points)
            num_drop = num_points-n
            indices_to_drop = np.random.choice(num_points, num_drop, replace=False)
            array_points = np.delete(array_points, indices_to_drop, axis=0)
            points = pd.DataFrame(array_points, columns=['x', 'y', 'z'])

        else:
            points = self.__add_padding(self, pd.DataFrame(np.asarray(down_pcd.points), columns=['x', 'y', 'z']))

        return points

class OrthogonalRegularizer(keras.regularizers.Regularizer):
    """Reference: https://keras.io/examples/vision/pointnet/#build-a-model"""

    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.identity = keras.ops.eye(num_features)

    def __call__(self, x):
        x = keras.ops.reshape(x, (-1, self.num_features, self.num_features))
        xxt = keras.ops.tensordot(x, x, axes=(2, 2))
        xxt = keras.ops.reshape(xxt, (-1, self.num_features, self.num_features))
        return keras.ops.sum(self.l2reg * keras.ops.square(xxt - self.identity))

    def get_config(self):
        config = super().get_config()
        config.update({"num_features": self.num_features, "l2reg_strength": self.l2reg})
        return config


def train(ARGS, train_df, test_df, max_voxel, contextnet):
    total_training_examples = len(train_df.columns) + len(test_df.columns)
    steps_per_epoch = total_training_examples // ARGS.batch_size
    total_training_steps = steps_per_epoch * ARGS.max_epochs
    dictionary = pd.read_pickle(ARGS.dictionaryfile)
    print(f"Steps per epoch: {steps_per_epoch}.")
    print(f"Total training steps: {total_training_steps}.")

    train_loader = CustomDataloader(train_df, dictionary, True, ARGS)
    test_loader = CustomDataloader(test_df, dictionary, False, ARGS)


    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=ARGS.lr,
        decay_steps=steps_per_epoch * 5,
        decay_rate=ARGS.decay_rate,
        staircase=True,
    )

    contextnet.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=custom_loss(len(dictionary)+1),
        metrics=["accuracy"]
    )

    checkpoint_filepath = f'{ARGS.path_model}\checkpoint_contextnet_v1.weights.h5'
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
    )

    history = contextnet.fit(
        train_loader,
        validation_data=test_loader,
        batch_size=ARGS.batch_size,
        epochs=ARGS.max_epochs,
        callbacks=[checkpoint_callback],
    )

    contextnet.load_weights(checkpoint_filepath)
    return contextnet, history


def custom_loss(number_of_classes):
    def loss(y_true, y_pred):
        crossentropy_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        return tf.reduce_mean(crossentropy_loss)
    return loss


def conv_block(x, filters, name):
    x = layers.Conv1D(filters, kernel_size=1, padding="valid", name=f"{name}_conv")(x)
    x = layers.BatchNormalization(name=f"{name}_batch_norm")(x)
    return layers.Activation("relu", name=f"{name}_relu")(x)


def mlp_block(x, filters, name):
    x = layers.Dense(filters, name=f"{name}_dense")(x)
    x = layers.BatchNormalization(name=f"{name}_batch_norm")(x)
    return layers.Activation("relu", name=f"{name}_relu")(x)


def get_model(ARGS, n_classes):

    model = keras.Sequential()
    model.add(layers.Input(shape=(36 * ARGS.num_points + 11,)))
    model.add(layers.Dense(8192))
    model.add(layers.LeakyReLU(negative_slope=ARGS.leaky_slope))
    model.add(layers.Dense(4096))
    model.add(layers.LeakyReLU(negative_slope=ARGS.leaky_slope))
    model.add(layers.Dense(2048))
    model.add(layers.LeakyReLU(negative_slope=ARGS.leaky_slope))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(negative_slope=ARGS.leaky_slope))
    model.add(layers.Dense(128))
    model.add(layers.LeakyReLU(negative_slope=ARGS.leaky_slope))
    model.add(layers.Dense(n_classes, activation="softmax"))
    return model


def get_dataset():
    search_string: str = 'data_df'
    row_names = ['path', 'voxel_size', 'labels',  'point_variance_x', 'point_variance_y', 'point_variance_z',
                 'point_skew_x', 'point_skew_y', 'point_skew_z', 'rel_proportion', 'density', 'weight', 'com']
    df_paths = pd.DataFrame(index=row_names)

    for filename in os.listdir(ARGS.path):
        if filename.endswith(".pkl") and search_string in filename:
            file_path = os.path.join(ARGS.path, filename)

            if ARGS.train_on_single_masks:

                try:
                    df = pd.read_pickle(file_path)
                    cols_to_drop = []

                    for col in df.columns:
                        if 'assembly' in df.loc['path', col]:
                            cols_to_drop.append(col)

                    # Drop the columns from the DataFrame
                    df = df.drop(cols_to_drop, axis=1)

                    df_paths = pd.concat([df_paths, df.loc[row_names]], axis=1)
                    del df
                    gc.collect()
                except Exception as e:
                    print(f"Failed to load {filename}: {e}")

            else:

                try:
                    # Load the dataframe from pickle file
                    df = pd.read_pickle(file_path)
                    df_paths = pd.concat([df_paths, df.loc[row_names]], axis=1)
                    del df
                    gc.collect()
                except Exception as e:
                    print(f"Failed to load {filename}: {e}")

    max_voxel = df_paths.loc['voxel_size'].max()
    dictionary = pd.read_pickle(ARGS.dictionaryfile)
    n_classes = len(dictionary) + 1
    train_df, test_df = train_test_split(df_paths.T, test_size=0.2, random_state=42)
    return train_df.T, test_df.T, n_classes, max_voxel


def ensure_unique_column_names(df):
    """
    Ensure that all column names in the DataFrame are unique.
    If duplicates are found, append a suffix to make them unique.

    Args:
    df (pd.DataFrame): The DataFrame whose columns are to be checked for uniqueness.

    Returns:
    pd.DataFrame: A DataFrame with guaranteed unique column names.
    """
    # Convert column names to a list to preserve order
    cols = df.columns.tolist()
    # Dictionary to track occurrences of each column name
    name_count = {}
    # Iterate over column names to detect and modify duplicates
    for idx, column in enumerate(cols):
        if column in name_count:
            # If the column name already exists, increment the count and append it
            name_count[column] += 1
            cols[idx] = f"{column}_{name_count[column]}"
        else:
            # If the column name is new, start counting from 1
            name_count[column] = 1

    # Assign the new column names back to the DataFrame
    df.columns = cols
    return df


def get_unique_lbls(df_row):
    unique_lbls = set()
    for entry in df_row:
        if hasattr(entry, '__iter__') and not isinstance(entry, str):
            unique_lbls.update(entry)
        else:
            # If the entry is not iterable or is a single scalar value, add it directly
            unique_lbls.add(entry)
    return list(unique_lbls)


if __name__ == '__main__':
    ARGS = options()
    main(ARGS)