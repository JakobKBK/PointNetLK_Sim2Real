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
    # io settings.
    parser = argparse.ArgumentParser(description='PoseNet')
    parser.add_argument('--path', default=r'D:\data\train_data\DEMO_EMO', type=str,
                        metavar='PATH', help='path to data directory')
    parser.add_argument('--outfile', type=str, default='./logs/220524_train_log',
                        metavar='BASENAME', help='output filename (prefix)')

    # settings for input data
    parser.add_argument('--data_type', default='synthetic', type=str,
                        metavar='DATASET', help='whether data is synthetic or real')
    parser.add_argument('--dictionaryfile', type=str, default=r'D:\data\train_data\DEMO_EMO\dictionary.pkl',
                        metavar='PATH', help='path to the categories to be trained')
    parser.add_argument('--num_points', default=2048, type=int,
                        metavar='N', help='points in point-cloud.')
    parser.add_argument('--min_points', default=32, type=int,
                        metavar='N', help='minimum number of points in mask to be considered for training.')
    parser.add_argument('--workers', default=4, type=int,
                        metavar='N', help='number of data loading workers')

    # settings for training.
    parser.add_argument('--batch_size', default=64, type=int,
                        metavar='N', help='mini-batch size')
    parser.add_argument('--max_iter', default=1e3, type=int,
                        metavar='N', help='max iter voxel down')
    parser.add_argument('--max_epochs', default=10, type=int,
                        metavar='N', help='number of total epochs to run') #default 200
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
    parser.add_argument('--omit_single_masks', type=bool, default=False,
                        metavar='BOOL', help='wether to omit masked single part pointclouds')
    parser.add_argument('--path_model', type=str, default=r'C:\MaH_Kretschmer\data\models',
                        metavar='PATH', help='safe path for trained model .keras (prefix) & checkpoints')

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
    siam_posenet = get_siamese_model(n_classes, ARGS)
    model, history = train(ARGS, train_df, test_df, max_voxel, siam_posenet)
    model.save(f'{ARGS.path_model}\posenet.keras')


class CustomDataloader(keras.utils.Sequence):

    def __init__(self, data, dictionary, shuffle, ARGS):
        super().__init__()
        self.data = data.loc['path']
        self.lbls = data.loc['labels']
        self.voxel_size = data.loc['voxel_size']
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
        data_a = np.empty((ARGS.batch_size, ARGS.num_points, 3))
        data_b = np.empty((ARGS.batch_size, ARGS.num_points, 3))
        label = np.empty((ARGS.batch_size, 8), dtype=int)
        j = 0
        for i, idx in enumerate(indexes):
            if j < ARGS.batch_size:
                for part_id in self.keys_lbl:
                    label_src = self.lbls.iloc[idx]
                    if np.any(label_src == part_id):
                        path_dp = self.data.iloc[idx]
                        voxel = self.voxel_size.iloc[idx]
                        cloud_temp = PyntCloud.from_file(path_dp)
                        dp = cloud_temp.points
                        dp, label_src = self.get_roi(dp, label_src, part_id)
                        label_src = torch.tensor(label_src.to_numpy(), dtype=torch.float32)

                        if len(dp) > ARGS.min_points:
                            if np.random.rand() < 0.05:
                                # Generate a random scale factor between 0 and 2
                                scale_factor = 2 * np.random.rand()

                                # Scale the point cloud
                                dp *= scale_factor

                            if len(dp) > self.num_points_set:
                                dp = self.__voxel_down_sample_to_n(self, dp, voxel)

                            if len(dp) < self.num_points_set:
                                dp = self.__add_padding(self, dp)

                            if label_src.shape.numel() < dp.shape[0]:
                                n_repeats = dp.shape[0] - label_src.shape.numel()
                                label_src = self.__align_lbls(label_src, n_repeats)

                            if label_src.shape.numel() > dp.shape[0]:
                                label_src = self.__shorten_lbls(label_src, self.num_points_set)

                            label_np = label_src.numpy().astype(np.int32)
                            points = dp.to_numpy()
                            data_a[j,] = points
                            points_transformed, dual_quat = self.transform_points(points)
                            data_b[j,] = points_transformed
                            dual_quat_flat = np.concatenate(dual_quat)
                            label[j] = dual_quat_flat
                            j += 1

                else:
                    return (data_a, data_b), label

        return (data_a, data_b), label

    def transform_points(self, points):

        q, qd, translation_3d = self.random_rotation_translation()
        points_transformed = self.apply_dual_quaternion_to_pointcloud(points, q, qd, translation_3d[1:])

        return points_transformed, (q, translation_3d)

    def random_rotation_translation(self):
        # Random angle between 0 and 2*pi
        angle = np.random.uniform(0, 2 * np.pi)

        # Random unit vector for axis of rotation
        axis = np.random.normal(size=3)
        axis = axis / np.linalg.norm(axis)

        # Quaternion for rotation
        q = np.array([
            np.cos(angle / 2),
            np.sin(angle / 2) * axis[0],
            np.sin(angle / 2) * axis[1],
            np.sin(angle / 2) * axis[2]
        ])

        # Random translation vector
        translation = np.random.uniform(-1, 1, 3)

        # Quaternion for translation (dual part)
        translation_quat = [0] + translation.tolist()
        translation_quat = np.array(translation_quat)
        qd = self.quaternion_multiply(translation_quat, q)
        qd = [0.5 * x for x in qd]

        return q, np.array(qd), translation_quat

    def quaternion_multiply(self, q1, q2):
        """
        Multiplies two quaternions.
        q1, q2 are given as [w, x, y, z].
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        ]

    def apply_dual_quaternion_to_pointcloud(self, cloud, q, qd, translation_3d):
        # Convert quaternion to rotation matrix
        q_mat = self.quaternion_to_rotation_matrix(q)
        transformed_cloud = np.dot(cloud, q_mat.T)

        # Apply translation
        transformed_cloud += translation_3d

        return transformed_cloud

    def quaternion_to_rotation_matrix(self, q):
        """Convert a quaternion into a 3x3 rotation matrix."""
        w, x, y, z = q
        return np.array([
            [1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x ** 2 - 2 * y ** 2]
        ])

    def to_one_hot_enc_lbls(self, labels):
        get_index_or_negative_one = lambda label: self.label_to_index.get(label, -1)
        vectorized_function = np.vectorize(get_index_or_negative_one)
        label_indices = vectorized_function(labels)
        one_hot_labels = np.eye(self.n_classes)[label_indices + 1]
        return one_hot_labels

    def get_roi(self, dp, labels, part_id):

        df = pd.DataFrame(columns=['x', 'y', 'z', 'label'])
        df[['x', 'y', 'z']] = dp[['x', 'y', 'z']]
        df['label'] = labels
        masked_df = df[df['label'] == part_id]
        masked_df = masked_df.iloc[:, :-1]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(masked_df.values)
        o3d_bbox = pcd.get_oriented_bounding_box()
        bbox_points = o3d_bbox.get_box_points()
        tol_param = np.random.randint(1, 5) * np.random.rand() + 0.1
        bbox_coordinates = self.min_max_values_bbox(np.asarray(bbox_points), tol_param)

        roi_adjusted_df = df[
            (df['x'] >= bbox_coordinates[0][0]) & (df['x'] <= bbox_coordinates[0][1]) &
            (df['y'] >= bbox_coordinates[1][0]) & (df['y'] <= bbox_coordinates[1][1]) &
            (df['z'] >= bbox_coordinates[2][0]) & (df['z'] <= bbox_coordinates[2][1])
            ]

        return roi_adjusted_df.iloc[:, :-1], roi_adjusted_df.iloc[:, -1]

    def min_max_values_bbox(self, points, tol_ratio=0.5):
        # Extract x, y, and z coordinates from the points
        x_values = points[:, 0]
        y_values = points[:, 1]
        z_values = points[:, 2]

        # Calculate minimum and maximum values for each dimension
        min_x, max_x = np.min(x_values), np.max(x_values)
        min_y, max_y = np.min(y_values), np.max(y_values)
        min_z, max_z = np.min(z_values), np.max(z_values)

        x_tol = tol_ratio * (max_x - min_x)
        y_tol = tol_ratio * (max_y - min_y)
        z_tol = tol_ratio * (max_z - min_z)

        min_x = min_x - x_tol
        max_x = max_x + x_tol
        min_y = min_y - y_tol
        max_y = max_y + y_tol
        min_z = min_z - z_tol
        max_z = max_z + z_tol

        return [[min_x, max_x], [min_y, max_y], [min_z, max_z]]

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
        config = {"num_features": self.num_features, "l2reg_strength": self.l2reg}
        return config


def train(ARGS, train_df, test_df, max_voxel, siam_posenet):
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

    siam_posenet.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='mean_squared_error',
        metrics=["accuracy"]
    )

    checkpoint_filepath = f'{ARGS.path_model}/checkpoint_posenet.weights.h5'
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        mode="auto",
        save_freq="epoch",
        initial_value_threshold=None,
    )

    history = siam_posenet.fit(
        train_loader,
        validation_data=test_loader,
        batch_size=ARGS.batch_size,
        epochs=ARGS.max_epochs,
        callbacks=[checkpoint_callback],
    )

    siam_posenet.load_weights(checkpoint_filepath)
    return siam_posenet, history


def conv_block(x, filters, name, kernel_size=1):
    x = layers.Conv1D(filters, kernel_size, padding="valid", name=f"{name}_conv")(x)
    x = layers.BatchNormalization(name=f"{name}_batch_norm")(x)
    return layers.Activation("relu", name=f"{name}_relu")(x)


def mlp_block(x, filters, name):
    x = layers.Dense(filters, name=f"{name}_dense")(x)
    x = layers.BatchNormalization(name=f"{name}_batch_norm")(x)
    return layers.Activation("relu", name=f"{name}_relu")(x)


def get_siamese_model(n_classes, ARGS):
    posenet = get_segmentation_model(n_classes, ARGS)

    input_a = layers.Input(shape=(ARGS.num_points, 3))
    input_b = layers.Input(shape=(ARGS.num_points, 3))

    # Both branches share the same PointNet
    processed_a = posenet(input_a)
    processed_b = posenet(input_b)

    reshaped_a = layers.Reshape((512, 512))(processed_a)
    reshaped_b = layers.Reshape((512, 512))(processed_b)

    concatenated = layers.Concatenate(axis=-1)([reshaped_a, reshaped_b])
    features_64 = conv_block(concatenated, filters=64, name="features_64", kernel_size=1)
    features_128_1 = conv_block(features_64, filters=128, name="features_128_1", kernel_size=1)
    features_128_2 = conv_block(features_128_1, filters=128, name="features_128_2", kernel_size=1)

    # x = layers.Conv2D(64, kernel_size=(64, 128), activation='relu', padding='same')(concatenated)
    # x = layers.Dropout(0.5)(x)  # Adding dropout for regularization
    # x = layers.Conv2D(32, kernel_size=(64, 64), activation='relu', padding='same')(x)
    # x = layers.Dropout(0.3)(x)  # Additional dropout layer
    # x = layers.Conv2D(16, kernel_size=(64, 64), activation='relu', padding='same')(x)
    # x = layers.Dropout(0.2)(x)  # Adding dropout for regularization
    # x = layers.Conv2D(8, kernel_size=(64, 64), activation='relu', padding='same')(x)
    # x = layers.Dropout(0.1)(x)  # Additional dropout layer
    # x = layers.Flatten()(x)

    pose_input = layers.Concatenate(name="segmentation_input")(
        [
            features_64,
            features_128_1,
            features_128_2,
        ]
    )
    pose_features = conv_block(
        pose_input, filters=128, name="segmentation_features", kernel_size=1
    )
    flat_features = layers.Flatten()(pose_features)
    predictions = layers.Dense(8, activation='sigmoid')(flat_features)
    model = keras.models.Model(inputs=[input_a, input_b], outputs=predictions)
    return model

def compute_transformation(features):
    """
    Custom function to compute translation and rotation.
    Assume features is a list [processed_a, processed_b] where each is of shape (batch_size, feature_dim).

    For example purposes, let's concatenate them and pass through a small network.
    """
    concatenated = tf.concat(features, axis=-1)
    # Example network to compute the translation and rotation.
    # Adjust this network architecture to suit the actual computation needed.
    x = layers.Dense(128, activation='relu')(concatenated)
    output = layers.Dense(8)(x)  # Output layer for 8 components
    return output


def get_segmentation_model(n_classes, ARGS):
    input_points = keras.Input(shape=(ARGS.num_points, 3))

    # PointNet Classification Network.
    transformed_inputs = transformation_block(
        input_points, num_features=3, name="input_transformation_block"
    )
    features_64 = conv_block(transformed_inputs, filters=64, name="features_64")
    features_128_1 = conv_block(features_64, filters=128, name="features_128_1")
    features_128_2 = conv_block(features_128_1, filters=128, name="features_128_2")
    transformed_features = transformation_block(
        features_128_2, num_features=128, name="transformed_features"
    )
    features_512 = conv_block(transformed_features, filters=512, name="features_512")
    features_2048 = conv_block(features_512, filters=2048, name="pre_maxpool_block")
    global_features = layers.MaxPool1D(pool_size=ARGS.num_points, name="global_features")(
        features_2048
    )
    global_features = keras.ops.tile(global_features, [1, ARGS.num_points, 1])

    # Segmentation head.
    segmentation_input = layers.Concatenate(name="segmentation_input")(
        [
            features_64,
            features_128_1,
            features_128_2,
            transformed_features,
            features_512,
            global_features,
        ]
    )
    segmentation_features = conv_block(
        segmentation_input, filters=128, name="segmentation_features"
    )
    flat_features = layers.Flatten()(segmentation_features)

    return keras.Model(inputs=input_points, outputs=flat_features)


def transformation_net(inputs, num_features, name):
    """
    The `filters` values come from the original paper:
    https://arxiv.org/abs/1612.00593.
    """
    x = conv_block(inputs, filters=64, name=f"{name}_1")
    x = conv_block(x, filters=128, name=f"{name}_2")
    x = conv_block(x, filters=1024, name=f"{name}_3")
    x = layers.GlobalMaxPooling1D()(x)
    x = mlp_block(x, filters=512, name=f"{name}_1_1")
    x = mlp_block(x, filters=256, name=f"{name}_2_1")
    return layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=keras.initializers.Constant(np.eye(num_features).flatten()),
        activity_regularizer=OrthogonalRegularizer(num_features),
        name=f"{name}_final",
    )(x)


def transformation_block(inputs, num_features, name):
    transformed_features = transformation_net(inputs, num_features, name=name)
    transformed_features = layers.Reshape((num_features, num_features))(
        transformed_features
    )
    return layers.Dot(axes=(2, 1), name=f"{name}_mm")([inputs, transformed_features])


def get_dataset():
    search_string: str = 'data_df'
    row_names = ['path', 'voxel_size', 'labels']
    df_paths = pd.DataFrame(index=row_names)

    for filename in os.listdir(ARGS.path):
        if filename.endswith(".pkl") and search_string in filename:
            file_path = os.path.join(ARGS.path, filename)

            if ARGS.omit_single_masks:

                try:
                    df = pd.read_pickle(file_path)
                    cols_to_drop = []

                    for col in df.columns:
                        if 'assembly' not in df.loc['path', col]:
                            cols_to_drop.append(col)

                    # Drop the columns from the DataFrame
                    df = df.drop(cols_to_drop, axis=1)
                    df = update_paths(df, 'path', 'C:\\MaH_Kretschmer\\data\\train_data\\DEMO_EMO', ARGS.path)
                    df_paths = pd.concat([df_paths, df.loc[row_names]], axis=1)
                    del df
                    gc.collect()
                except Exception as e:
                    print(f"Failed to load {filename}: {e}")

            else:

                try:
                    # Load the dataframe from pickle file
                    df = pd.read_pickle(file_path)
                    df = update_paths(df, 'path', 'C:\\MaH_Kretschmer\\data\\train_data\\DEMO_EMO', ARGS.path)
                    df_paths = pd.concat([df_paths, df.loc[row_names]], axis=1)
                    del df
                    gc.collect()
                except Exception as e:
                    print(f"Failed to load {filename}: {e}")

    dictionary = pd.read_pickle(ARGS.dictionaryfile)
    n_classes = len(dictionary)+1
    max_voxel = df_paths.loc['voxel_size'].max()
    df_paths = update_paths(df_paths, 'path', 'C:\\MaH_Kretschmer\\data\\train_data\\DEMO_EMO', ARGS.path)
    train_df, test_df = train_test_split(df_paths.T, test_size=0.2, random_state=42)
    return train_df.T, test_df.T, n_classes, max_voxel


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