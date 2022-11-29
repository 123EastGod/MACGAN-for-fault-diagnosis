import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os, glob, random, csv
from PIL import Image
from tqdm import tqdm
from MACGAN_model import Discriminator, Generator, Classifier

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# generate gray images
def preprocess(img):
    img = np.round((img * 127.5 + 127.5 ))
    # img = img.astype(np.uint8)
    return img


# loss function of MACGAN
def d_loss_fn(generator, discriminator, classifier, input, img, label, is_training):
    g = generator(input, is_training)
    d_fake_logits = discriminator(g, is_training)
    d_real_logits = discriminator(img, is_training)
    d_loss_real = -tf.reduce_mean(d_real_logits)
    d_loss_fake = tf.reduce_mean(d_fake_logits)
    loss = d_loss_fake + d_loss_real
    return loss

def g_loss_fn(generator, discriminator, classifier, input,img, label, is_training):
    g = generator(input, is_training)
    d_fake_logits = discriminator(g, is_training)
    c_real = classifier(img, is_training)
    c_real_loss = tf.reduce_mean(tf.losses.categorical_crossentropy(label, c_real))
    c_fake = classifier(g, is_training)
    c_fake_loss = tf.reduce_mean(tf.losses.categorical_crossentropy(input[1], c_fake))
    loss = -tf.reduce_mean(d_fake_logits) + 0.5*c_fake_loss + 0.5*c_real_loss
    return loss

def c_loss_fn(generator, classifier,input ,imgs, label, is_training):
    c_real = classifier(imgs, is_training)
    real_loss = tf.reduce_mean(tf.losses.categorical_crossentropy(label, c_real))
    g = generator(input, is_training)
    c_fake = classifier(g, is_training)
    fake_loss = tf.reduce_mean(tf.losses.categorical_crossentropy(input[1], c_fake))
    loss = real_loss + 0.1*fake_loss
    return loss

def read_image(img_name):
    im = Image.open(img_name).convert('L')
    data = np.array(im)
    data = (data.astype(np.float32) - 127.5)/127.5
    return data

def dataset(path, csvfile, class_number):
    classnames = os.listdir(path)
    name2label = {}
    classnum = 0
    for name in classnames:
        if classnum == class_number:
            break
        if not os.path.isdir(os.path.join(path, name)):
            continue
        name2label[name] = len(name2label.keys())
        classnum = classnum+1
    images, labels = [], []
    for name in name2label.keys():
        images += glob.glob(os.path.join(path, name, '*.jpg'))
    with open(os.path.join(path, csvfile), mode='w', newline='') as f:  # w模式会清空之前的文件
        writer = csv.writer(f)
        for img in images:
            name = img.split(os.sep)[-2]  # -2为读取上一级目录的名称，即为读取标签名
            label = name2label[name]
            writer.writerow([img, label])
    with open(os.path.join(path, csvfile)) as f:
        reader = csv.reader(f)
        for row in reader:
            img, label = row
            label = int(label)
            labels.append(label)
    return images, labels
# --------------------------------------------------------------------------------------

# Data load
def entiredataset(X_train, Y_train):
    x_train = []
    for i in X_train:
        x = read_image(i)
        x_train.append(x)
    x_train = np.asarray(x_train)
    x_train = np.expand_dims(x_train, axis=3)
    y_train = np.asarray(Y_train)
    y_train = y_train.reshape(-1, 1)
    return x_train, y_train

def train_sets(x_train,y_train, sample_num,class_num):
    a = np.zeros(shape=(sample_num*class_num,64,64,1))
    b = np.zeros(shape=(sample_num*class_num,1),dtype='int32')
    for i in range(class_num):
        a[i*sample_num:i*sample_num+sample_num] = x_train[i*500:i*500+sample_num]
        b[i*sample_num:i*sample_num+sample_num] = y_train[i*500:i*500+sample_num]
    return a,b

def extract_of_data(x_train, y_train, begin, class_num,sample_num ):
    X_train, Y_train  = x_train[begin*sample_num:(begin+class_num)*sample_num], y_train[begin*sample_num:(begin+class_num)*sample_num]
    return X_train, Y_train
# --------------------------------------------------------------------------------------

# training
def main():
    # config
    tf.random.set_seed(233)
    np.random.seed(233)
    generator = Generator()
    generator.build(input_shape=[(None, 100), (None, 7)])
    classifier = Classifier()
    classifier.build(input_shape=(None, 64, 64, 1))
    discriminator = Discriminator()
    discriminator.build(input_shape=(None, 64, 64, 1))
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5)
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5)
    c_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001, beta_1=0.9, beta_2=0.999)
    batch_size = 32
    sample_num = 50
    class_num = 7   # Number of Health states
    path = r'gray_images\\'
    csvfile = '2022-11-28-dataset'
    epoch = 30000

    # data load
    x_train, y_train = dataset(path, csvfile, class_num)
    x_train, y_train = entiredataset(x_train, y_train)
    x_train, y_train = train_sets(x_train, y_train, sample_num, class_num)
    begin = 0
    x_train, y_train = extract_of_data(x_train, y_train, begin, class_num,sample_num)
    print(x_train.shape, y_train.shape, 'number of training set')
    for i in range(class_num):
        print(y_train[i*sample_num])
    print( 'check training set')
    loss = {'d': [], 'g': [], 'c': []}
    is_training = True
    g_input = {}

    # load pre-training model
    checkpoint_dir = 'parameter'+'./training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=g_optimizer,
                                     discriminator_optimizer=d_optimizer,
                                     classifier_optimizer=c_optimizer,
                                     generator=generator,
                                     discriminator=discriminator,
                                     classifier=classifier)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    for epoch in tqdm(range(epoch)):

        # train Discriminator
        for _ in range(5):
            noise = tf.random.normal([batch_size, 100])
            idx1 = np.random.randint(0, x_train.shape[0], batch_size)
            imgs, label = x_train[idx1], y_train[idx1]
            label = tf.squeeze(tf.one_hot(label, 7))
            sample_label = np.random.randint(begin, begin + class_num, (batch_size, 1))
            sample_label = tf.squeeze(tf.one_hot(sample_label, 7))
            g_input[0] = noise
            g_input[1] = sample_label
            with tf.GradientTape() as tape:
                d_loss = d_loss_fn(generator, discriminator, classifier, g_input, imgs, label, is_training)
            d_grads = tape.gradient(d_loss, discriminator.trainable_variables)
            d_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))

        # training Classifier
        noise = tf.random.normal([batch_size, 100])
        idx1 = np.random.randint(0, x_train.shape[0], batch_size)
        imgs, label = x_train[idx1], y_train[idx1]
        label = tf.squeeze(tf.one_hot(label, 7))
        sample_label = np.random.randint(begin, begin + class_num, (batch_size, 1))
        sample_label = tf.squeeze(tf.one_hot(sample_label, 7))
        g_input[0] = noise
        g_input[1] = sample_label
        with tf.GradientTape() as tape:
            c_loss = c_loss_fn(generator, classifier, g_input, imgs, label, is_training)
        c_grads = tape.gradient(c_loss, classifier.trainable_variables)
        c_optimizer.apply_gradients(zip(c_grads, classifier.trainable_variables))

        # training Generator
        sample_label = np.random.randint(begin, begin + class_num, (batch_size, 1))
        sample_label = tf.squeeze(tf.one_hot(sample_label, 7))
        g_input[0] = tf.random.normal([batch_size, 100])
        g_input[1] = sample_label
        with tf.GradientTape() as tape:
            g_loss = g_loss_fn(generator, discriminator, classifier, g_input, imgs, label, is_training)
        g_grads = tape.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))

        # record loss curve of every five training epoch
        if epoch % 5 == 0:
            print(epoch, 'd-loss:', float(d_loss), 'g-loss:', float(g_loss), 'c_loss', float(c_loss))
        loss['d'].append(d_loss)      
        loss['g'].append(g_loss)
        loss['c'].append(c_loss)

        # save parameters of model
        if epoch % 500 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        # generate images of every 200 training epoch
        if epoch % 200 == 0:
            generate_num = 100
            batch = generate_num * class_num
            noise = np.random.normal(0, 1, (batch, 100))
            sampled_labels = np.zeros(shape=(batch , 1),dtype='int32')
            for i in range(begin, begin + class_num):
                sampled_labels[(i - begin) * generate_num:(i - begin) * generate_num + generate_num] = i
            sampled_labels = tf.squeeze(tf.one_hot(sampled_labels, 7))
            fake_image = generator([noise, sampled_labels], training=False)
            preprocesed = preprocess(fake_image)
            preprocesed = preprocesed.reshape(batch, 64, 64)
            for b in range(preprocesed.shape[0]):
                img = preprocesed[b, :, :]
                im = Image.fromarray(img)
                im.convert('L').save('generated images\\' + str(int(b/generate_num)+begin)  + '_' + str(b-int(b/generate_num)*generate_num) + '.jpg', format='jpeg')

    # Plot the loss function
    fig, ax = plt.subplots(figsize=(10,7),facecolor='white')
    font2 = {'weight':'bold', 'size':23}
    plt.xlabel('Iteration times', font2)
    plt.ylabel('Loss', font2)
    plt.plot(loss['g'], label='Generator')
    plt.plot(loss['d'], label='Discriminator')
    plt.plot(loss['c'], label='Classifier')  # 画出损失函数的图
    plt.tick_params(labelsize=20)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontweight('bold') for label in labels]
    legend_properties = {'size':20,'weight':'bold'}
    plt.legend(prop=legend_properties)
    plt.subplots_adjust(bottom=0.15)
    plt.savefig('D:\\Study_Code\\AEI论文代码\\CRWU GAN\\1118\\' + '7类' + '.svg', dpi=600, bbox_inches = 'tight')
    plt.show()
    plt.clf()


if __name__ == '__main__':
    main()


