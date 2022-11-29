import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, MaxPooling2D, BatchNormalization
from SpectralNormalizationKeras import DenseSN, ConvSN2D

class Discriminator(keras.Model):

    def __init__(self):
        super(Discriminator, self).__init__()

        # [b, 64, 64, 1] => [b, 1]

        self.conv1 = ConvSN2D(32, 5, 2, 'same',kernel_initializer='glorot_uniform')
        self.bn1 = layers.BatchNormalization()
        self.max1 = layers.MaxPooling2D(2, 2, 'same')

        self.conv2 = ConvSN2D(64, 3, 2, 'same',kernel_initializer='glorot_uniform')
        self.bn2 = layers.BatchNormalization()
        self.max2 = layers.MaxPooling2D(2, 2, 'same')

        self.conv3 = ConvSN2D(128, 3, 2, 'same',kernel_initializer='glorot_uniform')
        self.bn3 = layers.BatchNormalization()
        self.max3 = layers.MaxPool2D(2, 2, 'same')

        self.conv4 = ConvSN2D(256, 3, 2, 'same',kernel_initializer='glorot_uniform')
        self.max4 = MaxPooling2D(2, 2, 'same')
        # [b, h, w ,c] => [b, -1]
        self.flatten = layers.Flatten()
        #self.fc1 = layers.Dense(256, kernel_constraint=spectral_normalization)
        self.fc2 = DenseSN(1)


    def call(self, inputs, training=None):

        x = tf.nn.leaky_relu(self.max1(self.conv1(inputs)))
        x = tf.nn.leaky_relu(self.max2(self.conv2(x)))
        x = tf.nn.leaky_relu(self.max3(self.conv3(x)))
        x = tf.nn.leaky_relu(self.max4(self.conv4(x)))

        # [b, h, w, c] => [b, -1]
        # [b, -1] => [b, 1]
        logit = self.fc2(x)

        return logit

class Generator(keras.Model):

    def __init__(self, ):
        self.num_classes = 7
        self.latent_dim = 100
        super(Generator, self).__init__()
        self.embedding = layers.Embedding(self.num_classes, self.latent_dim)  #将（200，1）的标签转化为（200，1，100）
        self.flatten = layers.Flatten()                                       #打平层变成（200，100）
        # z: [b, 100] => [b, 3*3*512] => [b, 3, 3, 512] => [b, 64, 64, 3]
        self.fc = Dense(4*4*512)

        self.conv1 = Conv2DTranspose(128, 5, 2, 'same')
        self.bn1 = BatchNormalization()

        self.conv2 = Conv2DTranspose(64, 5, 2, 'same')
        self.bn2 = BatchNormalization()

        self.conv3 = Conv2DTranspose(32, 5, 2, 'same')
        self.bn3 = BatchNormalization()

        self.conv4 = Conv2DTranspose(1, 3, 2, 'same')



    def call(self, inputs, training=None):
        # [z, 100] => [z, 3*3*512]
        label_embedding = self.flatten(inputs[1])
        model_input = tf.concat([inputs[0], label_embedding],axis=1)
        x = tf.nn.leaky_relu(self.fc(model_input))
        x = tf.reshape(x, [-1, 4, 4, 512])
        x = tf.nn.leaky_relu(self.bn1(self.conv1(x), training=training))
        x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=training))
        x = tf.nn.leaky_relu(self.bn3(self.conv3(x), training=training))
        x = tf.tanh(self.conv4(x))  # 输出为-1到1

        return x

class Classifier(keras.Model):
    def __init__(self):
        super(Classifier, self).__init__()

        self.conv1 = Conv2D(32, 3, 2, 'same')
        self.bn1 = BatchNormalization()
        self.max1 = MaxPooling2D(2 , 2 ,'same')

        self.conv2 = Conv2D(64, 3, 2, 'same')
        self.bn2 = BatchNormalization()
        self.max2 = MaxPooling2D(2, 2, 'same')

        self.conv3 = Conv2D(128, 3, 2, 'same')
        self.bn3 = BatchNormalization()
        self.max3 = MaxPooling2D(2, 2, 'same')

        self.conv4 = Conv2D(256, 3, 2, 'same')
        self.bn4 = BatchNormalization()
        self.max4 = MaxPooling2D(2, 2, 'same')

        self.flatten = layers.Flatten()
        self.fc1 = Dense(256)
        self.fc2 = Dense(128)
        self.fc3 = Dense(7)

    def call(self, inputs, training=None):
        x = tf.nn.relu(self.max1(self.bn1(self.conv1(inputs), training=training)))
        x = tf.nn.relu(self.max2(self.bn2(self.conv2(x), training=training)))
        x = tf.nn.relu(self.max3(self.bn3(self.conv3(x), training=training)))
        x = tf.nn.relu(self.max4(self.bn4(self.conv4(x), training=training)))

        # [b, h, w, c] => [b, -1]
        x = self.flatten(x)
        # [b, -1] => [b, 1]
        x = tf.nn.relu(self.fc1(x))
        x = tf.nn.relu(self.fc2(x))
        x = tf.nn.softmax(self.fc3(x))

        return x