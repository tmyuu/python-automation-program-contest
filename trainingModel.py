import io
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense
from sklearn.model_selection import train_test_split

def resize_image(image):
    image = Image.open(io.BytesIO(image))
    # リサイズ処理
    resize_shape = (32, 32)
    image = image.resize(resize_shape, Image.LANCZOS)

    # グレースケールの場合はRGBに変換
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # NumPy配列に変換
    resized_image = np.asarray(image)

    # ピクセル値を0から1の範囲に正規化（RGB各チャンネル）
    resized_image = resized_image / 255.0

    return resized_image


def scale_image_imgaug(image, scale_range=(0.9, 1.1)):
    import imgaug.augmenters as iaa

    scale = iaa.Affine(scale={"x": scale_range, "y": scale_range})
    scaled_image = scale.augment_image(image)
    return scaled_image


def gaussian_noise_imgaug(image):
    import imgaug.augmenters as iaa

    # imgaugは0から255の範囲の値を期待するため、float64型の画像データを一時的に変換
    image_uint8 = (image * 255).astype(np.uint8)

    # imgaugを使用してガウシアンノイズを追加
    augmenter = iaa.AdditiveGaussianNoise(scale=(0, 10)) # スケールを調整（例: 0から10）
    noisy_image_uint8 = augmenter.augment_image(image_uint8)

    # ノイズが追加された画像を再び0から1の範囲に変換
    noisy_image = noisy_image_uint8.astype(np.float64) / 255.0

    return noisy_image


# def salt_pepper_noise_and_clip(image, salt_pepper=(0.01, 0.05)):
#     import imgaug.augmenters as iaa

#     augmenter = iaa.SaltAndPepper(salt_pepper)
#     noisy_image = augmenter.augment_image(image)

#     # ピクセル値を0〜1の範囲に制限
#     noisy_image_clipped = np.clip(noisy_image, 0, 1)
#     return noisy_image_clipped


def preprocess_image(df, resize_image):
    preprocessed_images = np.array([resize_image(x) for x in df['fashionImage']])

    # # 1. 画像を水平方向に反転
    # flipped_horizontally = np.flip(preprocessed_images, axis=2)
    # preprocessed_images = np.concatenate((preprocessed_images, flipped_horizontally), axis=0)

    # # 2. 画像を90度回転
    # rotated_90 = np.rot90(preprocessed_images, k=1, axes=(1, 2))
    # preprocessed_images = np.concatenate((preprocessed_images, rotated_90), axis=0)

    # # 3. 画像を180度回転
    # rotated_180 = np.rot90(preprocessed_images, k=2, axes=(1, 2))
    # preprocessed_images = np.concatenate((preprocessed_images, rotated_180), axis=0)

    # 4. 画像のスケールを変更
    scaled_images = np.array([scale_image_imgaug(x) for x in preprocessed_images])
    preprocessed_images = np.concatenate((preprocessed_images, scaled_images), axis=0)

    # 5. ノイズを追加
    noisy_images = np.array([gaussian_noise_imgaug(x) for x in preprocessed_images])
    preprocessed_images = np.concatenate((preprocessed_images, noisy_images), axis=0)

    # # 6. 塩胡椒ノイズの追加
    # augmented_images = np.array([salt_pepper_noise_and_clip(x) for x in preprocessed_images])
    # preprocessed_images = np.concatenate((preprocessed_images, augmented_images), axis=0)

    return preprocessed_images


def initialize_vgg16(input_shape, fashion_category):
    # VGG16モデルのロード（トップ層は含まない）
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    # 新しいトップ層を追加
    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    outputs = Dense(fashion_category, activation='softmax')(x)

    # 新しいモデルの作成
    model = Model(inputs=base_model.input, outputs=outputs)

    # VGG16の層のうち、いくつかを凍結（オプション）
    for layer in base_model.layers[:15]:  # 最初の15層を凍結
        layer.trainable = False

    return model


def solution(x_test_df, train_df):
    # データ前処理
    fashion_categories = list(train_df['fashionCategory'].unique())

    # トレーニングデータのラベル生成
    train_images = preprocess_image(train_df, resize_image)
    train_labels = np.array([fashion_categories.index(x) for x in train_df['fashionCategory']] * 4)

    # VGG16用のモデル初期化
    fashion_categories_classes = len(fashion_categories)
    input_shape = (32, 32, 3)

    model = initialize_vgg16(input_shape, fashion_categories_classes)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # モデルのトレーニング
    model.fit(train_images, train_labels, epochs=10, batch_size=32)

    # モデルの保存
    model.save("./my_model.keras")

    # テストデータに対する予測
    test_images = preprocess_image(x_test_df, resize_image)
    image_classes = len(x_test_df['fashionImage'])

    test_predictions = model.predict(test_images)
    aggregated_logits = np.zeros(image_classes * fashion_categories_classes, dtype=np.float64).reshape((image_classes, fashion_categories_classes))
    for n in range(4):
        aggregated_logits += test_predictions[image_classes * n  :image_classes * (n + 1)]

    # カテゴリ予測の決定
    predictions = tf.nn.softmax(aggregated_logits).numpy()
    answer = [fashion_categories[x.argmax()] for x in predictions]
    
    return pd.DataFrame({'fashionCategory': answer}, index=x_test_df.index)


# def plot_confusion_matrix_and_accuracy(y_true, y_pred, classes):
#     import seaborn as sns
#     import matplotlib.pyplot as plt
#     from sklearn.metrics import confusion_matrix

#     # Confusion matrixの計算
#     cm = confusion_matrix(y_true, y_pred, labels=classes)

#     # ヒートマップとしてプロット
#     plt.figure(figsize=(10, 10))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
#     plt.title('Confusion Matrix')
#     plt.ylabel('Actual')
#     plt.xlabel('Predicted')
#     plt.show()

#     # 各クラスごとの正確さと最も間違えやすいクラスを表示
#     print("\nClass Accuracy and Most Common Errors:")
#     for i, class_name in enumerate(classes):
#         accuracy = cm[i, i] / cm[i, :].sum()
#         print(f"{class_name}: Accuracy: {accuracy * 100:.2f}%")
        
#         # 最も間違えやすいクラスを特定
#         error_indices = cm[i, :].argsort()[-2:-1] if accuracy < 1 else []
#         for error_index in error_indices:
#             error_rate = cm[i, error_index] / cm[i, :].sum()
#             error_class = classes[error_index]
#             print(f"    Most common error: Mistaken for {error_class} ({error_rate * 100:.2f}%)")


def main():
    # データのインポート
    df = pd.read_pickle("/Users/mypc/Desktop/zozo_images/zozo_dataset.pkl")
    
    train_df, test_df = train_test_split(df, stratify=df['fashionCategory'], test_size=0.10, random_state=42)

    # テストデータの画像データとラベルデータの分離
    y_test_df = test_df[['fashionCategory']]
    x_test_df = test_df.drop(columns=['fashionCategory'])

    # solution関数を実行
    user_result_df = solution(x_test_df, train_df)
    # plot_confusion_matrix_and_accuracy(y_test_df['fashionCategory'], user_result_df['fashionCategory'], df['fashionCategory'].unique())

    average_accuracy = 0
    # ユーザーの提出物のフォーマット確認
    if type(y_test_df) == type(user_result_df) and y_test_df.shape == user_result_df.shape:
        # 平均精度の計算
        accuracies = {}
        for category_type in df['fashionCategory'].unique():
            y_test_df_by_category_type = y_test_df[y_test_df['fashionCategory'] == category_type]
            user_result_df_by_category_type = user_result_df[y_test_df['fashionCategory'] == category_type]
            matching_rows = (y_test_df_by_category_type == user_result_df_by_category_type).all(axis=1).sum()
            accuracies[category_type] = (matching_rows/(len(y_test_df_by_category_type)))
        
        average_accuracy = sum(accuracies.values())/len(accuracies)

    print(f"平均精度：{average_accuracy*100:.2f}%")

# 実行
main()
