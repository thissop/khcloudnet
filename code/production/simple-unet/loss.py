import tensorflow as tf

@tf.function
def tversky(y_true, y_pred, alpha=0.7, smooth=1):
    y_t = tf.where(tf.equal(y_true, 1), 1.0, 0.0)
    y_w = tf.where(tf.equal(y_true, 10), 2.0, 1.0)

    tp = tf.reduce_sum(y_pred * y_t)
    fp = (1 - alpha) * tf.reduce_sum((y_pred * y_w) * (1 - y_t))
    fn = alpha * tf.reduce_sum(((1 - y_pred) * y_w) * y_t)

    numerator = tp
    denominator = tp + fp + fn

    score = (numerator + smooth) / (denominator + smooth)
    return 1 - score

@tf.function
def focal_tversky(y_true, y_pred, alpha=0.7):
    loss = tversky(y_true, y_pred, alpha)
    return tf.pow(loss, 0.75)

@tf.function
def accuracy(y_true, y_pred):
    y_t = tf.where(tf.equal(y_true, 1), 1.0, 0.0)

    tp = tf.reduce_sum(y_pred * y_t)
    fp = tf.reduce_sum(y_pred * (1 - y_t))
    fn = tf.reduce_sum((1 - y_pred) * y_t)
    tn = tf.reduce_sum((1 - y_pred) * (1 - y_t))
    return (tp + tn) / (tp + tn + fp + fn + tf.keras.backend.epsilon())

@tf.function
def dice_coef(y_true, y_pred, smooth=1e-7):
    y_t = tf.where(tf.equal(y_true, 1), 1.0, 0.0)

    tp = tf.reduce_sum(y_pred * y_t)
    fp = tf.reduce_sum(y_pred * (1 - y_t))
    fn = tf.reduce_sum((1 - y_pred) * y_t)

    two_tp = 2.0 * tp
    return (two_tp + smooth) / (two_tp + fp + fn + smooth)
