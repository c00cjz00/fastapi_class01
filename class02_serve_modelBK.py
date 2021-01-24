from io import BytesIO
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.imagenet_utils import decode_predictions

# 函式: 讀取讀片
def read_imagefile(file) -> Image.Image:
#def read_imagefile(file):
    # 內存讀取二進制數據, 轉為圖片
    image = Image.open(BytesIO(file))
    return image

# 函式: 預測圖片
#def predict(image: Image.Image):
def predict(image):
    # 模型
    model = tf.keras.applications.MobileNetV2(weights="imagenet")
    
    # 影像處理
    image = np.asarray(image.resize((224, 224)))[..., :3]
    image = np.expand_dims(image, 0)
    image = image / 127.5 - 1.0
    
    # 預測結果輸出
    result = decode_predictions(model.predict(image), 3)[0]
    return result


# Python 直接執行 python serve_model.py
if __name__ == "__main__":
    image = Image.open('cat.png')
    prediction = predict(image)
    print(prediction)
