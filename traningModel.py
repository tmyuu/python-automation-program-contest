import io
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add


def resize_image(image):
    resize_shape = (32, 32)

    resized_image = Image.open(io.BytesIO(image))
    resized_image = resized_image.resize(resize_shape, Image.LANCZOS)
    resized_image = np.array(resized_image) / 255.0  # 正規化
    return resized_image


def preprocess_image(df, resize_image):
    preprocessed_images = np.array([resize_image(x) for x in df['fashionImage']])

    rotated_90 = np.rot90(preprocessed_images, k=1, axes=(1, 2))
    preprocessed_images = np.concatenate((preprocessed_images, rotated_90), axis=0)

    rotated_180 = np.rot90(preprocessed_images, k=2, axes=(1, 2))
    preprocessed_images = np.concatenate((preprocessed_images, rotated_180), axis=0)

    flipped_horizontally = np.flip(preprocessed_images, axis=2)
    preprocessed_images = np.concatenate((preprocessed_images, flipped_horizontally), axis=0)

    return preprocessed_images


def initialize_cnn(input_shape, fashion_category):
    inputs = Input(shape=input_shape)

    x = Conv2D(16, 3, activation='relu', padding='same')(inputs)
    x = MaxPooling2D(pool_size=2, strides=2)(x)
    skip_connection = Conv2D(64, 1, strides=2, activation='relu', padding='same')(x)

    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=2, strides=2)(x)
    x = Add()([x, skip_connection])

    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=2, strides=2)(x)

    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)

    outputs = Dense(fashion_category)(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model


def solution(x_test_df, train_df):
    # データ前処理
    fashion_categories = list(train_df['fashionCategory'].unique())

    # トレーニングデータのラベル生成
    train_images = preprocess_image(train_df, resize_image)
    train_labels = np.array([fashion_categories.index(x) for x in train_df['fashionCategory']] * 8)

    fashion_categories_classes = len(fashion_categories)
    input_shape = train_images[0].shape

    model = initialize_cnn(input_shape, fashion_categories_classes)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00024),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # モデルのトレーニング
    model.fit(train_images, train_labels, epochs=10, batch_size=32)

    # モデルの保存
    model_save_path = "~/Desktop/model/my_model.keras"  # 保存先のパス
    model.save(model_save_path)

    # テストデータに対する予測
    test_images = preprocess_image(x_test_df, resize_image)
    image_classes = len(x_test_df['fashionImage'])

    test_predictions = model.predict(test_images)
    aggregated_logits = np.zeros(image_classes * fashion_categories_classes, dtype=np.float64).reshape((image_classes, fashion_categories_classes))
    for n in range(8):
        aggregated_logits += test_predictions[image_classes * n  :image_classes * (n + 1)]

    # カテゴリ予測の決定
    predictions = tf.nn.softmax(aggregated_logits).numpy()
    answer = [fashion_categories[x.argmax()] for x in predictions]
    
    return pd.DataFrame({'fashionCategory': answer}, index=x_test_df.index)


def main():
    # データのインポート
    df = pd.read_pickle("~/Desktop/image/fashion_dataset.pkl")
    
    train_df, test_df = train_test_split(df, stratify=df['fashionCategory'], test_size=0.10, random_state=42)

    # テストデータの画像データとラベルデータの分離
    y_test_df = test_df[['fashionCategory']]
    x_test_df = test_df.drop(columns=['fashionCategory'])

    # solution関数を実行
    user_result_df = solution(x_test_df, train_df)

    average_accuracy = 0
    # ユーザーの提出物のフォーマット確認
    if type(y_test_df) == type(user_result_df) and y_test_df.shape == user_result_df.shape:
        # 平均精度の計算
        accuracies = {}
        for failure_type in df['fashionCategory'].unique():
            y_test_df_by_failure_type = y_test_df[y_test_df['fashionCategory'] == failure_type]
            user_result_df_by_failure_type = user_result_df[y_test_df['fashionCategory'] == failure_type]
            matching_rows = (y_test_df_by_failure_type == user_result_df_by_failure_type).all(axis=1).sum()
            accuracies[failure_type] = (matching_rows/(len(y_test_df_by_failure_type)))
        
        average_accuracy = sum(accuracies.values())/len(accuracies)

        # ...

    print(f"平均精度：{average_accuracy*100:.2f}%")

# 実行
main()
