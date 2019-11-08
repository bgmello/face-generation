import tensorflow as tf

def generator_loss(d_logits_fake, d_model_fake):
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,labels=tf.ones_like(d_model_fake))
    g_loss=tf.reduce_mean(cross_entropy)
    return g_loss

def discriminator_loss_real(d_logits_real, d_model_real):
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,labels=tf.ones_like(d_model_real))
    d_loss_real=tf.reduce_mean(cross_entropy)
    return d_loss_real

def discriminator_loss_fake(d_logits_fake, d_model_fake):
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,labels=tf.zeros_like(d_model_fake))
    d_loss_fake=tf.reduce_mean(cross_entropy)
    return d_loss_fake