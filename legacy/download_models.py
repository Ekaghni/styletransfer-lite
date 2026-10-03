# Python script to download models
import urllib.request

# Style Prediction Model
urllib.request.urlretrieve(
    "https://tfhub.dev/google/lite-model/magenta/arbitrary-image-stylization-v1-256/int8/prediction/1?lite-format=tflite",
    "style_predict.tflite"
)

# Style Transform Model  
urllib.request.urlretrieve(
    "https://tfhub.dev/google/lite-model/magenta/arbitrary-image-stylization-v1-256/int8/transfer/1?lite-format=tflite",
    "style_transform.tflite"
)

print("Downloaded both models")