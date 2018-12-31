from generator import get_files, Img2ImgGenerator
import os

# BDD100K = "H:/Dataset/bdd100k"
BDD100K = "/mnt/hdd/Dataset/bdd100k"

def get_generators(batch_size):
    x_train_files = []
    y_train_files = get_files(BDD100K + '/drivable_maps/labels/train/*.png')

    x_valid_files = []
    y_valid_files = get_files(BDD100K + '/drivable_maps/labels/val/*.png')

    for file in y_train_files:
        id = file.replace('\\', '/').split('/')[-1].split('_')[0]
        x_file = BDD100K + '/images/100k/train/' + id + '.jpg'

        if not os.path.exists(x_file):
            print('Not exist ', x_file)
            y_train_files.remove(file)
            continue

        x_train_files.append(x_file)

    for file in y_valid_files:
        id = file.replace('\\', '/').split('/')[-1].split('_')[0]
        x_file = BDD100K + '/images/100k/val/' + id + '.jpg'

        if not os.path.exists(x_file):
            print('Not exist ', x_file)
            y_valid_files.remove(file)
            continue

        x_valid_files.append(x_file)

    print('train: %d -> %d' % (len(x_train_files), len(y_train_files)))
    print('valid: %d -> %d' % (len(x_valid_files), len(y_valid_files)))

    train_gen = Img2ImgGenerator(x_train_files, y_train_files, batch_size, x_shape=(288, 512), y_shape=(288, 512))
    valid_gen = Img2ImgGenerator(x_valid_files, y_valid_files, batch_size, x_shape=(288, 512), y_shape=(288, 512))

    return train_gen, valid_gen

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    train, valid = get_generators(1)

    for x, y in train:
        plt.imshow(x[0])
        plt.show()
        plt.imshow(y[0])
        plt.show()
        input()
