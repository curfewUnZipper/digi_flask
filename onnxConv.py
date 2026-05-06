import tensorflow as tf
import tf2onnx

print("[INFO] Loading model...")

model = tf.keras.models.load_model(
    "models/lstm_model.keras"
)

# -------------------------------------------------
# CREATE INPUT SIGNATURE
# -------------------------------------------------

input_signature = [
    tf.TensorSpec(
        [None, 20, 5],
        tf.float32,
        name="input"
    )
]

# -------------------------------------------------
# WRAP MODEL
# -------------------------------------------------

@tf.function(input_signature=input_signature)
def model_fn(x):
    return model(x)

print("[INFO] Converting to ONNX...")

onnx_model, _ = tf2onnx.convert.from_function(
    model_fn,
    input_signature=input_signature,
    opset=13
)

# -------------------------------------------------
# SAVE
# -------------------------------------------------

with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("[SUCCESS] model.onnx saved")