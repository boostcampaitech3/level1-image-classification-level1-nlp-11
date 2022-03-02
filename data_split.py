import argparse
import os
import random
import shutil

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold



def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('An error occurred While attempting to create the directory ' + directory)


class Data_Split:
    def __init__(self, dirname):
        self.dirname = dirname


    def train_val_split(self, train_label, n_splits):
        """
        stratifiedkfold를 활용하여 데이터를 train,val로 나눈다.
        param:
            train_label: train_with_label.csv
            n_splits: kfold_num
        :return:
        """
        X = train_label.index
        y = train_label["label"]
        sf_KFold = StratifiedKFold(n_splits)

        train, val = [], []
        for train_idx, val_idx in sf_KFold.split(X, y):
            train.append(train_label[train_label.index.isin(X[train_idx])].reset_index(drop=True))
            val.append(train_label[train_label.index.isin(X[val_idx])].reset_index(drop=True))
            # print("TRAIN:", train_idx, "TEST:", val_idx) # 확인용
        return train, val


    def load_image(self, df, df_name):
        dir_name = os.path.join(self.dirname, df_name)
        # 빈 directory 생성
        create_folder(dir_name)

        # 위에서 만든 빈 directory에 파일들을 복사
        for idx in range(len(df)):
            path = df["path"][idx]
            name = df["name"][idx]
            shutil. (path, os.path.join(dir_name, name))


if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument("--train_path", default="/opt/ml/pstage1/input/data/train", type=str, help="train_path",)
    args.add_argument("--image_data", default="train_with_label.csv", type=str, help="CSV according to image type(Original, Crop, All)",)
    args.add_argument("--image_dir", default="images_split", type=str, help="Directory according to image type",)
    args.add_argument("--n_splits", default=5, type=int, help="stratified_kfold_num")
    args = args.parse_args()
    np.random.seed(42)
    random.seed(42)

    
    train_label = pd.read_csv(os.path.join(args.train_path, args.image_data))
    run_train = Data_Split(os.path.join(args.train_path, args.image_dir))
    n_splits = args.n_splits
    train_list, val_list = run_train.train_val_split(train_label, n_splits)

    for idx in range(n_splits):
        run_train.load_image(train_list[idx], f"train_{idx}")
        run_train.load_image(val_list[idx], f"val_{idx}")