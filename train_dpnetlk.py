import trainer
import pandas as pd
import torch
import torchvision
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from pyntcloud import PyntCloud

import data_utils
import trainer


def options(argv=None):
    # io settings.
    parser.add_argument('--outfile', type=str, default='./logs/220524_train_log',
                        metavar='BASENAME', help='output filename (prefix)')

    # settings for input data
    parser.add_argument('--dataset_type', default='modelnet', type=str,
                        metavar='DATASET', help='dataset type')
    parser.add_argument('--data_type', default='synthetic', type=str,
                        metavar='DATASET', help='whether data is synthetic or real')
    parser.add_argument('--dictionaryfile', type=str, default='./dataset/modelnet40_half1.txt',
                        metavar='PATH', help='path to the categories to be trained')
    parser.add_argument('--num_points', default=1000, type=int,
                        metavar='N', help='points in point-cloud.')
    parser.add_argument('--num_random_points', default=100, type=int,
                        metavar='N', help='number of random points to compute Jacobian.')
    parser.add_argument('--mag', default=0.8, type=float,
                        metavar='D', help='max. mag. of twist-vectors (perturbations) on training (default: 0.8)')
    parser.add_argument('--sigma', default=0.00, type=float,
                        metavar='D', help='noise range in the data')
    parser.add_argument('--clip', default=0.00, type=float,
                        metavar='D', help='noise range in the data')
    parser.add_argument('--workers', default=4, type=int,
                        metavar='N', help='number of data loading workers')

    # settings for Embedding
    parser.add_argument('--embedding', default='pointnet',
                        type=str, help='pointnet')
    parser.add_argument('--dim_k', default=1024, type=int,
                        metavar='K', help='dim. of the feature vector')

    # settings for LK
    parser.add_argument('--max_iter', default=10, type=int,
                        metavar='N', help='max-iter on LK.')

    # settings for training.
    parser.add_argument('--batch_size', default=32, type=int,
                        metavar='N', help='mini-batch size')
    parser.add_argument('--max_epochs', default=200, type=int,
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

    # settings for log
    parser.add_argument('--logfile', default='', type=str,
                        metavar='LOGNAME', help='path to logfile')
    parser.add_argument('--resume', default='', type=str,
                        metavar='PATH', help='path to latest checkpoint')
    parser.add_argument('--pretrained', default='', type=str,
                        metavar='PATH', help='path to pretrained model file')

    args = parser.parse_args(argv)

    return args


def train(ARGS, train_df, test_df, dptnetlk):

    train_costum = CustomDataset(train_df)
    test_costum = CustomDataset(test_df)

    train_loader = torch.utils.data.DataLoader(train_costum, batch_size=ARGS.batch_size, shuffle=True,
                                               num_workers=ARGS.workers, drop_last=True)

    test_loader = torch.utils.data.DataLoader(test_costum, batch_size=ARGS.batch_size, shuffle=False,
                                              num_workers=ARGS.workers, drop_last=True)


class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, data):
        self.data = data['path']
        self.labels = data['lbl']

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        label = self.labels.iloc[idx]
        path_dp = self.data.iloc[idx]
        cloud_temp = PyntCloud.from_file(path_dp)
        dp = cloud_temp.points
        dp = torch.tensor(dp.values)
        label = torch.tensor(label)

        return dp, label


def main(ARGS):

    train_df, test_df = get_dataset(ARGS.path)
    dptnetlk = trainer.TrainerAnalyticalPointNetLK(ARGS)
    train(ARGS, train_df, test_df, dptnetlk)


def path_comprehension(path, file_name):
    p = Path(path)
    joined_path = p / file_name
    return joined_path.as_posix()


def get_dataset(path):

    path_data = path_comprehension(path, 'data_df.pkl')
    data_df = pd.read_pickle(path_data)
    path_dictionary = path_comprehension(ARGS.path, 'dictionary.pkl')
    dictionary = pd.read_pickle(path_dictionary)
    cinfo = (list(dictionary.values()), list(dictionary.keys()))

    train_df, test_df = train_test_split(data_df, test_size=0.2, random_state=42, stratify=True)

    transform = torchvision.transforms.Compose([data_utils.OnUnitCube(), data_utils.Resampler(ARGS.num_points)])

    traindata = data_utils.ModelNet(ARGS.dataset_path, train=1, transform=transform, classinfo=cinfo)
    evaldata = data_utils.ModelNet(ARGS.dataset_path, train=0, transform=transform, classinfo=cinfo)

    trainset = data_utils.PointRegistration(traindata, data_utils.RandomTransformSE3(ARGS.mag))
    evalset = data_utils.PointRegistration(evaldata, data_utils.RandomTransformSE3(ARGS.mag))

    return train_df, test_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PointNet-LK')
    parser.add_argument('--path', default=Path.cwd(), type=str,
                        metavar='PATH', help='path to data directory (default: path to script folder)')
    ARGS = options()
    main(ARGS)
