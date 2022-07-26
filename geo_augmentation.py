from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


def geo_aug(train_images,test_images):
    datagen = ImageDataGenerator(featurewise_center=False,samplewise_center=False, featurewise_std_normalization=False,samplewise_std_normalization=False, zca_whitening=False,
                                 rotation_range=15, zoom_range = 0.07,width_shift_range=0.04, shear_range = 0.03, height_shift_range=0.05,  horizontal_flip=False,  vertical_flip=False)

    datagen.fit(train_images)

    imgs=[]
    for i in range (0,16):
      imgs.append(datagen.flow(test_images)[i][0].reshape(28,28))

    _, axs = plt.subplots(4, 4, figsize=(8, 8))
    axs = axs.flatten()
    for img, ax in zip(imgs, axs):
        ax.imshow(img,cmap='gray')
    plt.show()
