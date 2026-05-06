import tensorflow as tf

print("[INFO] Loading model...")

model = tf.keras.models.load_model(
    "models/lstm_model.keras"
)

print("[INFO] Creating converter...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# -------------------------------------------------
# FIX FOR LSTM / TensorList Ops
# -------------------------------------------------

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

converter._experimental_lower_tensor_list_ops = False

# -------------------------------------------------
# OPTIONAL OPTIMIZATION
# -------------------------------------------------

converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]

print("[INFO] Converting model...")

tflite_model = converter.convert()

print("[INFO] Saving model.tflite")

with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("[SUCCESS] TFLite model created successfully")