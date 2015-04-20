# Credits to Kyle Kastner
import os
import zipfile
import fnmatch
import matplotlib.image as mpimg
import numpy as np
from scipy.misc import imresize
import tables


def load_color(random_seed=123522):
    # Check if dataset is in the data directory.
    data_path = os.path.join(os.path.split(__file__)[0], "data")
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    dataset = 'train.zip'
    data_file = os.path.join(data_path, dataset)
    if os.path.isfile(data_file):
        dataset = data_file

    if (not os.path.isfile(data_file)):
        try:
            import urllib
            urllib.urlretrieve('http://google.com')
        except AttributeError:
            import urllib.request as urllib
        url = 'https://dl.dropboxusercontent.com/u/15378192/train.zip'
        print('Downloading data from %s' % url)
        urllib.urlretrieve(url, data_file)

    data_dir = os.path.join(data_path, 'cvd')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        zf = zipfile.ZipFile(data_file)
        zf.extractall(data_dir)

    data_file = os.path.join(data_path, 'cvd_color.hdf5')
    label_file = os.path.join(data_path, 'cvd_color_labels.npy')
    if not os.path.exists(data_file):
        print('... loading data')
        cat_matches = []
        dog_matches = []
        for root, dirname, filenames in os.walk(data_dir):
            for filename in fnmatch.filter(filenames, 'cat*'):
                cat_matches.append(os.path.join(root, filename))
            for filename in fnmatch.filter(filenames, 'dog*'):
                dog_matches.append(os.path.join(root, filename))

        sort_key = lambda x: int(x.split('.')[-2])
        cat_matches = sorted(cat_matches, key=sort_key)
        dog_matches = sorted(dog_matches, key=sort_key)

        def square(x):
            resize_shape = (260, 260)
            slice_size = (256, 256)
            slice_left = (resize_shape[0] - slice_size[0]) / 2
            slice_upper = (resize_shape[1] - slice_size[1]) / 2
            return imresize(x, resize_shape, interp='nearest')[
                slice_left:slice_left + slice_size[0],
                slice_upper:slice_upper + slice_size[1]].transpose(
                    2, 0, 1).astype('uint8')

        matches = cat_matches + dog_matches
        matches = np.array(matches)
        random_state = np.random.RandomState(random_seed)
        idx = random_state.permutation(len(matches))
        c = [0] * len(cat_matches)
        d = [1] * len(dog_matches)
        y = np.array(c + d).astype('uint8')
        matches = matches[idx]
        y = y[idx]

        compression_filter = tables.Filters(complevel=5, complib='blosc')
        h5_file = tables.openFile(data_file, mode='w')
        example = square(mpimg.imread(matches[0]))
        image_storage = h5_file.createEArray(h5_file.root, 'images',
                                             tables.Int8Atom(),
                                             shape=(0,) + example.shape,
                                             filters=compression_filter)
        for n, f in enumerate(matches):
            print("Processing image %i of %i" % (n, len(matches)))
            x = square(mpimg.imread(f)).astype('uint8')
            image_storage.append(x[None])
        h5_file.close()
        np.save(label_file, y)
    h5_file = tables.openFile(data_file, mode='r')
    x_s = h5_file.root.images
    y_s = np.load(label_file)
    return (x_s, y_s)
