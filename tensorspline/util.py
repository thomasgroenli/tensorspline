import tensorflow as tf

def grid(dims):
    return tf.stack(tf.meshgrid(*[tf.linspace(0.,1.,x) for x in dims],indexing='ij'),axis=-1)