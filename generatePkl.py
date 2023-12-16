import os
import pandas as pd
import pickle

image_data = []
categories = []

# 画像ファイルのディレクトリを指定
image_directory = os.path.expanduser('~/Desktop/image')

for filename in os.listdir(image_directory):
    if filename.endswith('.jpg'):
        # ここでカテゴリ名を正しく抽出する方法を確認してください
        category = filename.split(' ')[0]
        categories.append(category)

        # 画像ファイルを開き、バイト配列に変換
        with open(os.path.join(image_directory, filename), 'rb') as file:
            img_byte_arr = file.read()
            image_data.append(img_byte_arr)

# DataFrameを作成
df = pd.DataFrame({'fashionImage': image_data, 'fashionCategory': categories})
print(df.info())

# pklファイルに保存
with open(os.path.expanduser('~/Desktop/image/fashion_dataset.pkl'), 'wb') as f:
    pickle.dump(df, f)
