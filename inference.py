import tensorflow as tf
m = tf.saved_model.load("./introduction_to_deep_learning_with_jax/lunar_lander_model")
out = m.signatures["serving_default"](x=tf.ones((2, 8), dtype=tf.float32))
print(out)
