import os

import numpy as np
import pandas as pd

class Make_Label:
    def __init__(self, base_path: str, csv_file: pd.DataFrame, col: list):
        """
        원래의 train.csv 파일에 labeling을 새로 해준다.
        param:
            base_path: 주어진 train.csv의 파일경로
            csv_file: train.csv 파일
            col: 새로 만들 DataFrame의 column
        """
        self.path = base_path
        self.img_path = os.path.join(base_path, "images")
        self.csv_file = csv_file
        self.df = pd.DataFrame(columns = col)

    def labeling(self):
        cnt = 0
        # 이미지 폴더 path
        for idx in range(train_df.shape[0]):
            img_path = os.path.join(self.img_path, train_df.iloc[idx]["path"])
            file_list = os.listdir(img_path)

            for file in file_list:
                if file.rstrip().startswith("._"):  # '._'로 시작하는 파일 거름
                    continue

                file_path = os.path.join(img_path, file)
                # train.csv 파일에서 gender, age를 갖고옴
                self.df.loc[cnt] = train_df.loc[idx][["gender", "age"]]
                self.df.loc[cnt]["path"] = file_path
                self.df.loc[cnt]["name"] = file_path.split("/")[-2] + "_" + file
                self.check_label(self.df, cnt)
                cnt += 1
        print('labeling done!')

        # 새로 만든 DataFrame 저장
        labeling_data_path = "/opt/ml/input/data/train/train_with_label.csv"
        self.df.to_csv(labeling_data_path, index=False)

    # 라벨링표를 참고하여 라벨링
    def check_label(self, df, idx_df):
        mask = df.loc[idx_df]["name"]
        gender = df.loc[idx_df]["gender"]
        age = df.loc[idx_df]["age"]

        if age < 30:
            label = 0
        elif 30 <= age < 60:
            label = 1
        else:
            label = 2

        if gender == "female":
            label += 3

        if "normal" in mask:
            label += 12
        elif "incorrect" in mask:
            label += 6

        df.loc[idx_df]["label"] = label


if __name__ == "__main__":
    # train 데이터 경로
    train_path = "/opt/ml/input/data/train"
    train_df = pd.read_csv(os.path.join(train_path, "train.csv"))

    # labeling 시행
    make_label = Make_Label(train_path, train_df, ["gender", "age", "path", "name", "label"])
    make_label.labeling()

    
    df = pd.read_csv('/opt/ml/input/train/train_with_label.csv')