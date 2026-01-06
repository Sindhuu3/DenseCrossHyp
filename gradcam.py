import tensorflow as tf
import numpy as np

def grad_cam_densenet(
    model,
    img_array,
    layer_name="conv5_block16_concat"
):
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_out = conv_out[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap)
    return heatmap.numpy()

def detect_orientation(heatmap):
    h, w = heatmap.shape
    scores = {
        "Left": heatmap[:, :w//2].sum(),
        "Right": heatmap[:, w//2:].sum(),
        "Top": heatmap[:h//2, :].sum(),
        "Bottom": heatmap[h//2:, :].sum()
    }
    return max(scores, key=scores.get)
