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
    parser.add_argument('--batch_size', default=10, type=int,
                        metavar='N', help='mini-batch size')
    parser.add_argument('--max_iter', default=1e3, type=int,
                        metavar='N', help='max iter voxel down')
    parser.add_argument('--max_epochs', default=60, type=int,
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
    segmentnet = get_segmentation_model(n_classes, ARGS)
    model, history = train(ARGS, train_df, test_df, max_voxel, segmentnet)
    model.save(f'{ARGS.path_model}\segmentnet.keras')


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
        data = np.empty((ARGS.batch_size, ARGS.num_points, 3))
        label = np.empty((ARGS.batch_size, ARGS.num_points, self.n_classes), dtype=int)
        for i, idx in enumerate(indexes):
            label_src = self.lbls.iloc[idx]
            path_dp = self.data.iloc[idx]
            voxel = self.voxel_size.iloc[idx]
            cloud_temp = PyntCloud.from_file(path_dp)
            dp = cloud_temp.points
            label_src = torch.tensor(label_src)

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
            data[i,] = points
            label[i] = self.to_one_hot_enc_lbls(label_np)

        return data, label

    def to_one_hot_enc_lbls(self, labels):
        get_index_or_negative_one = lambda label: self.label_to_index.get(label, -1)
        vectorized_function = np.vectorize(get_index_or_negative_one)
        label_indices = vectorized_function(labels)
        one_hot_labels = np.eye(self.n_classes)[label_indices + 1]
        return one_hot_labels

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


def train(ARGS, train_df, test_df, max_voxel, segmentnet):
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

    segmentnet.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=custom_loss(len(dictionary)+1),
        metrics=["accuracy"]
    )

    checkpoint_filepath = f'{ARGS.path_model}\checkpoint.weights.h5'
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
    )

    history = segmentnet.fit(
        train_loader,
        validation_data=test_loader,
        batch_size=ARGS.batch_size,
        epochs=ARGS.max_epochs,
        callbacks=[checkpoint_callback],
    )

    segmentnet.load_weights(checkpoint_filepath)
    return segmentnet, history


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
    outputs = layers.Conv1D(
        n_classes, kernel_size=1, activation="softmax", name="segmentation_head"
    )(segmentation_features)
    return keras.Model(input_points, outputs)


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

            try:
                # Load the dataframe from pickle file
                df = pd.read_pickle(file_path)
                df_paths = pd.concat([df_paths, df.loc[row_names]], axis=1)
                del df
                gc.collect()
            except Exception as e:
                print(f"Failed to load {filename}: {e}")

    dictionary = pd.read_pickle(ARGS.dictionaryfile)
    n_classes = len(dictionary)+1
    max_voxel = df_paths.loc['voxel_size'].max()
    train_df, test_df = train_test_split(df_paths.T, test_size=0.2, random_state=42)
    return train_df.T, test_df.T, n_classes, max_voxel


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
    parser = argparse.ArgumentParser(description='SegmentNet')
    parser.add_argument('--path', default=r'C:\MaH_Kretschmer\data\train_data\DEMO_EMO', type=str,
                        metavar='PATH', help='path to data directory')
    ARGS = options()
    main(ARGS)