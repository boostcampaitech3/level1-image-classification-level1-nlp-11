import argparse
import datetime as dt
import os
import time
import random
import cv2
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from data_split import Data_Split
from model.loss import batch_loss, FocalLoss
from model.metric import batch_acc, batch_f1, epoch_mean
from pytz import timezone
from torch.utils.data import DataLoader
from importlib import import_module
from PIL import Image

import wandb


class Mask_Dataset(object):
    def __init__(self, transforms, name, df, path, folder):
        self.transforms = transforms
        self.name = name
        self.path = path
        self.folder = folder
        self.imgs = sorted(os.listdir(os.path.join(self.path, f"{self.folder}/{self.name}")))
        self.df = df

        self.X = df['path']
        self.y = df['label']

    def __getitem__(self, idx):
        img_path = self.df["path"][idx] # idx번째 사진 경로
        target = self.df["label"][idx] # idx번째 정답라벨

        img = Image.open(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms:
            # augmented = self.transforms(image=img)
            # img = augmented["image"]
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)


def collate_fn(batch):
    return tuple(zip(*batch))


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    wandb.init(project="Mask_Image_Classification")

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument("-lr", "--learning_rate", default=0.0001, type=float, help="learning rate for training")
    parser.add_argument("-bs", "--batch_size", default=64, type=int, help="batch size for training")
    parser.add_argument("--epoch", default=10, type=int, help="training epoch size")
    parser.add_argument("--n_splits", default=5, type=int, help="StratifiedKFold size")
    parser.add_argument("--train_path", default="/opt/ml/pstage1/input/data/train", type=str, help="train_directory_path")
    parser.add_argument("--model_save", default="/opt/ml/pstage1/output/model_save", type=str, help="model_save_path")
    parser.add_argument("--normalize_mean",default=(0.485, 0.456, 0.406),type=float,help="Normalize mean value")
    parser.add_argument("--normalize_std",default=(0.229, 0.224, 0.225),type=float,help="Normalize std value")
    # Original:train_with_label.csv, Crop:train_with_crop.csv
    # parser.add_argument("--image_data",default="train_with_all.csv",type=str,help="CSV according to image type(Original, Crop, All)")
    parser.add_argument("--image_data", default="train_with_label.csv", type=str, help="CSV according to image type(Original, Crop, All)")
    # Original:image_all, Crop:image_crop_all
    # parser.add_argument("--image_dir",default="ori_crop_split",type=str,help="Directory according to image type")
    parser.add_argument("--image_dir", default="images_split", type=str, help="Directory according to image type")

    args = parser.parse_args()
    print(args)
    config = wandb.config
    config.learning_rate = args.learning_rate
    seed_everything(42) # 시드 고정

    train_label = pd.read_csv(os.path.join(args.train_path, args.image_data))
    run_split = Data_Split(os.path.join(args.train_path, args.image_dir))
    train_list, val_list = run_split.train_val_split(train_label, args.n_splits)
    print(f'type of train_list: {type(train_list[0])}')
    print(f'train_list:{train_list[0]}')
    num_classes = 18

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device is {device}!")

    data_transforms = transforms.Compose(
        [
            transforms.Resize((256, 192), Image.BILINEAR),
            # transforms.GaussianBlur(3, sigma_limit=(0.1, 2.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=args.normalize_mean, std=args.normalize_std)
        ]
    )

    #모델 저장할때 현재시간을 넣기위해 선언
    now = (dt.datetime.now().astimezone(timezone("Asia/Seoul")).strftime("%Y-%m-%d_%H%M%S"))

    model_save_path = args.model_save
    dirname = os.path.join(model_save_path, f"model_{now}")
    print(f'dirname={dirname}')
    ensure_dir(dirname)

    st_time = time.time()
    for i in range(args.n_splits):
        print(f'{i+1}번째 세트 학습 시작!-------------------------------------------------------------------------')
        # model.py 모듈을 import하고, 해당 모듈의 args.model를 불러오라는 뜻
        model_module = getattr(import_module("model.model"), args.model)  # default: BaseModel
        model = model_module(
            num_classes=num_classes
        ).to(device)
        model = torch.nn.DataParallel(model)

        wandb.watch(model)

        # 분류 학습 때 많이 사용되는 Cross entropy loss를 objective function으로 사용
        criterion = FocalLoss()
        # optimizer를 Adam으로 사용함
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        train_dataset = Mask_Dataset(data_transforms, f"train_{i}", train_list[i], args.train_path, args.image_dir)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=2)

        valid_dataset = Mask_Dataset(data_transforms, f"val_{i}", val_list[i], args.train_path, args.image_dir)
        valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=2)

        dataloaders = {"train": train_dataloader, "val": valid_dataloader}

        # 학습 코드 시작
        best_val_acc = 0.0
        best_val_loss = 9999.0

        for epoch in range(args.epoch):
            for phase in ['train', 'val']:
                n_iter = 0
                running_loss = 0.0
                running_acc = 0.0
                running_f1 = 0.0

                if phase == 'train':
                    model.train()  # 학습모드로 전환 (gradient 계산 및 Dropout이나 Batch-Normalization 동작함)
                elif phase ==  'val':
                    model.eval()  # 평가 모드

                for inputs, (images, labels) in enumerate(dataloaders[phase]):
                    images = torch.stack(list(images), dim=0).to(device)
                    labels = torch.tensor(list(labels)).to(device)

                    optimizer.zero_grad()  # parameter gradient를 업데이트 전 초기화함

                    with torch.set_grad_enabled(phase == 'train'): # train모드일때만 gradient를 업데이트하도록 한다.
                        outputs = model(images)
                        _, preds = torch.max(outputs, 1) # 입력된 데이터가 18개의 클래스에 속할 각각의 확률값이 output으로 출력된다. 그중 max값을 예측값으로 저장
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward() # CrossEntropy를 통해 gradient 계산
                            optimizer.step() # gradient를 통해 모델의 가중치를 갱신

                    running_loss += batch_loss(loss, images)  # 한 Batch에서의 loss 값 저장
                    running_acc += batch_acc(preds, labels.data)  # 한 Batch에서의 Accuracy 값 저장
                    running_f1 += batch_f1(preds.cpu().numpy(), labels.cpu().numpy(), "macro")
                    n_iter += 1
                    # 100번마다 로그찍음
                    if inputs % 100 == 0:
                        wandb.log({"loss": loss})
                        wandb.log({"lr": args.learning_rate})

                # epoch이 종료됨
                data_len = len(dataloaders[phase].dataset) # 15120, 3780
                epoch_loss = epoch_mean(running_loss, data_len)
                epoch_acc = epoch_mean(running_acc, data_len)
                epoch_f1 = epoch_mean(running_f1, n_iter)

                print(f"현재 epoch-{epoch}의 {phase}-데이터 셋에서 평균 Loss : {epoch_loss:.3f}, "
                      f"평균 Accuracy : {epoch_acc:.3f}, F1 Score : {epoch_f1:.3f}")
                if phase ==  'val' and best_val_acc < epoch_acc:  # phase가 test일 때, best accuracy 계산
                    best_val_acc = epoch_acc
                if phase ==  'val' and best_val_loss > epoch_loss:  # phase가 test일 때, best loss 계산
                    best_val_loss = epoch_loss

        torch.save(model, os.path.join(dirname, f"model_{i}.pth"))
        print("Job Finished!")
        print(f"Best accuracy : {best_val_acc}, Worst loss : {best_val_loss}")
    end_time = time.time()
    total_time = (round(end_time - st_time, 2)) // 60
    print(f"총 학습 시간 : {total_time}분.")