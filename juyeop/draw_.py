class_list = [[['wear'], ['male'], ['un30']], [['wear'], ['male'], ['3060']], [['wear'], ['male'], ['ov60']],
              [['wear'], ['fema'], ['un30']], [['wear'], ['fema'], ['3060']], [['wear'], ['fema'], ['ov60']],
              [['Inco'], ['male'], ['un30']], [['Inco'], ['male'], ['3060']], [['Inco'], ['male'], ['ov60']],
              [['Inco'], ['fema'], ['un30']], [['Inco'], ['fema'], ['3060']], [['Inco'], ['fema'], ['ov60']],
              [['NoWe'], ['male'], ['un30']], [['NoWe'], ['male'], ['3060']], [['NoWe'], ['male'], ['ov60']],
              [['NoWe'], ['fema'], ['un30']], [['NoWe'], ['fema'], ['3060']], [['NoWe'], ['fema'], ['ov60']]
             ]

def draw_(df):
    plt.figure(figsize = (15, 30))
    row = len(wrong_df) // 3
    # 틀린 번호 찾기
    wrong_number = list()
    for df_path in list(df['path']):
        wrong_number.append(df_path.split('/')[4].split('_')[0])
        
    for i in range(df.shape[0]):
        plt.subplot(row + 1, df.shape[0] // row, i + 1)
        plt.imshow(Image.open(df['path'][i]))
        plt.title(f"target:{class_list[df['target'][i]]}, \n pred: {class_list[df['pred'][i]]} \n id: {wrong_number[i]}", color='r', size=20)
        plt.axis('off')
    print(df)
    

    plt.tight_layout()
    plt.show()