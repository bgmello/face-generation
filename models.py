import tensorflow as tf

class Discriminator(tf.keras.Model):
    
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=2, strides=2, padding='same')
        self.relu1 = tf.keras.layers.LeakyReLU(alpha=0.1)
        
        self.conv2 = tf.keras.layers.Conv2D(filters=256, kernel_size=5, strides=2, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.LeakyReLU(alpha=0.1)
        
        self.fc = tf.keras.layers.Reshape(target_shape=(-1,7*7*256))
        self.logits = tf.keras.layers.Dense(1)
        self.outputs = tf.keras.layers.Activation('sigmoid')
        
    def call(self, x):
        """
        Create the discriminator network
        :param images: Tensor of input image(s)
        :param reuse: Boolean if the weights should be reused
        :return: Tuple of (tensor output of the discriminator, tensor logits of the discriminator)
        """
        # TODO: Implement Function
        x1 = self.relu1(self.conv1(x))
        x2 = self.relu2(self.bn2(self.conv2(x1)))
        logits = self.logits(self.fc(x2))
        output = self.outputs(logits)
        return (output,logits)


class Generator(tf.keras.Model):
    
    def __init__(self, out_channel_dim):
        super(Generator, self).__init__()
        
        self.fc1 = tf.keras.layers.Dense(2*2*512)
        self.rsp1 = tf.keras.layers.Reshape(target_shape=(2,2, 512))
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.LeakyReLU(0.1)
        
        self.convt2 = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=5, strides=2, padding='valid')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.LeakyReLU(0.1)
        
        self.convt3 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=5, strides=2, padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.relu3 = tf.keras.layers.LeakyReLU(0.1)
        
        self.logits = tf.keras.layers.Conv2DTranspose(filters=out_channel_dim, kernel_size=5, 
                                                      strides=2, padding='same')
        
        self.outputs = tf.keras.layers.Activation('tanh')
    
    def call(self, x):
        """
        Create the generator network
        :param z: Input z
        :param out_channel_dim: The number of channels in the output image
        :param is_train: Boolean if generator is being used for training
        :return: The tensor output of the generator
        """
        # TODO: Implement Function
        
        x1 = self.relu1(self.bn1(self.rsp1(self.fc1(x))))
        x2 = self.relu2(self.bn2(self.convt2(x1)))
        
        x3 = self.relu3(self.bn3(self.convt3(x2)))
        
        logits = self.logits(x3)
        
        output = self.outputs(logits)
        
        return output