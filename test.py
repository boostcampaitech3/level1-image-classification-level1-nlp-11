import argparse
import datetime as dt
import os
import time
import random
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from pytz import timezone
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


class TestDataset(Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])
        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path",default="/opt/ml/pstage1/input/data/eval",type=str,help="eval_dir_path",)
    parser.add_argument("--result_save",default="/opt/ml/pstage1/output/sub",type=str,help="submission_save_path",)
    parser.add_argument("--normalize_mean",default=(0.485, 0.456, 0.406),type=float,help="Normalize mean",)
    parser.add_argument("--normalize_std",default=(0.229, 0.224, 0.225),type=float,help="Normalize std",)
    parser.add_argument("--image_dir", default="images", type=str, help="image dir path")
    # 실행할때마다 train된 모델이 저장된 위치를 입력해주어야함.
    parser.add_argument('--model_path', default="/opt/ml/pstage1/output/model_save/model_2022-03-02_032524", type=str, help="saved model path")

    args = parser.parse_args()
    seed_everything(42)

    test_dir = args.test_path
    submission = pd.read_csv(os.path.join(test_dir, "info.csv"))
    image_dir = os.path.join(test_dir, args.image_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Test Dataset 클래스 객체를 생성하고 DataLoader를 생성.
    image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]

    transforms = transforms.Compose(
        [
            transforms.Resize((256, 192), Image.BILINEAR),
            # transforms.GaussianBlur(3, sigma_limit=(0.1, 2.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=args.normalize_mean, std=args.normalize_std)
        ]
    )

    dataset = TestDataset(image_paths, transforms)
    dataloader = DataLoader(dataset, shuffle=False, num_workers=2)

    model_listdir = os.listdir(args.model_path) # n_splits개의 모델이 저장되어있음
    all_predictions = [[] for _ in range(len(dataloader))]
    for i in model_listdir:
        """
        n_splits개의 모델들을 사용하여 18개의 라벨에 대한 모든 라벨들의 답안을 구한 후 그중 가장 높은 확률을 정답으로 제출
        """
        model = torch.load(os.path.join(args.model_path, i))
        model.eval()

        idx = 0
        st_time = time.time()
        for images in dataloader:
            with torch.no_grad():
                images = images.to(device)
                pred = model(images)
                pred = F.log_softmax(pred, dim=1).cpu().numpy() # 기존 softmax 대신 연산속도를 위해 log_softmax 사용
                if len(all_predictions[idx]) == 0:
                    all_predictions[idx] = pred / len(model_listdir)
                else:
                    all_predictions[idx] += pred / len(model_listdir)

            idx += 1
    # print(all_predictions[0])


    all_predictions = [pre.argmax() for pre in all_predictions] #softmax로 출력된 값들중에서 가장 큰값이 있는 index를 pred값으로 선정
    submission["ans"] = all_predictions

    # 제출할 csv파일을 현재시간을 이름으로하여 저장.
    now = (dt.datetime.now().astimezone(timezone("Asia/Seoul")).strftime("%Y-%m-%d_%H%M%S"))
    submission.to_csv(os.path.join(args.result_save, f"sub_{now}.csv"))
    print("Test Finish!!")