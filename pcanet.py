import copy
import random
import numpy as np


def im_to_col_mean_removal(in_img, patch_size, stride=[1, 1], remove_mean=True):
    image_shape = in_img.shape
    if len(image_shape) == 3:
        rows, cols, z = image_shape
    else:
        in_img = in_img.reshape(in_img.shape[0], in_img.shape[1], 1)
        rows, cols, z = in_img.shape

    im_rows = len(range(0, rows - patch_size[0] + 1, stride[0]))
    im_cols = len(range(0, cols - patch_size[1] + 1, stride[1]))
    im = np.zeros((patch_size[0] * patch_size[1], im_rows * im_cols * z))
    idx = 0
    for chl in range(z):
        for i in range(0, rows - patch_size[0] + 1, stride[0]):
            for j in range(0, cols - patch_size[1] + 1, stride[1]):
                iim = in_img[i:(i + patch_size[0]), j:(j + patch_size[1]), chl]
                iim = iim.reshape(patch_size[0] * patch_size[1], 1)
                iim = np.asarray(iim)
                im[:, idx] = iim.T
                idx += 1
        pass
    im = im.reshape(patch_size[0] * patch_size[1], im_rows * im_cols * z)
    if remove_mean:
        im_mean = np.mean(im, axis=0)
        im = im - im_mean
    return im


def pca_output(in_img, in_img_idx, patch_size, num_filters, v):
    out_img = []
    mag = (patch_size - 1) // 2
    for i in range(len(in_img)):
        img_x, img_y, num_channels = in_img[i].shape
        img = np.zeros((img_x + patch_size - 1, img_y + patch_size - 1, num_channels))
        img[mag:(mag + in_img[i].shape[0]), mag:(mag + in_img[i].shape[1]), :] = in_img[i]
        im = im_to_col_mean_removal(img, [patch_size, patch_size], remove_mean=True)
        for j in range(num_filters):
            out_img.append(v[:, j].T.dot(im).reshape(img_x, img_y, 1))
        pass

    out_img_idx = np.kron(in_img_idx, np.ones((1, num_filters))[0])
    return out_img, out_img_idx


def pac_filter_bank(in_img, patch_size, num_filter):
    num_r_samples = np.min([len(in_img), 81259])
    rand_idx = range(len(in_img))
    list_ind = list(rand_idx)
    random.shuffle(list_ind)
    rand_idx = list_ind[:num_r_samples]

    num_channels = in_img[0].shape[2]
    r_x = np.zeros((num_channels * pow(patch_size, 2), num_channels * pow(patch_size, 2)), dtype="float32")

    im = []
    for i in rand_idx:
        im = im_to_col_mean_removal(in_img[i], [patch_size, patch_size], remove_mean=True)
        r_x = r_x + im.dot(im.T)
    r_x /= (num_r_samples * im.shape[1])
    
    d, eig = np.linalg.eig(np.mat(r_x))
    ind = np.argsort(d)
    ind = ind[:-(num_filter + 1):-1]
    v = np.asarray(eig[:, ind])
    return v


def pca_net_fea_ext(pca_net, in_img, v):
    assert len(pca_net.num_filters) == pca_net.num_stages, 'Length(pca_net.num_filters)~=pca_net.num_stages'

    out_img = copy.deepcopy(in_img)
    img_idx = range(0, len(in_img), 1)
    for stage in range(pca_net.num_stages):
        out_img, img_idx = pca_output(out_img, img_idx, pca_net.patch_size[stage], pca_net.num_filters[stage], v[stage])

    return hashing_hist(pca_net, img_idx, out_img)


def pca_net_train(in_img, pca_net, idx_ext, display_number):
    assert len(pca_net.num_filters) == pca_net.num_stages, 'Length(pca_net.num_filters)~=pca_net.num_stages'

    v = []
    out_img = copy.deepcopy(in_img)
    img_idx = range(0, len(in_img), 1)

    for stage in range(pca_net.num_stages):
        print('Computing PCA filter bank and its outputs at stage %d ...' % stage)
        v.append(pac_filter_bank(out_img, pca_net.patch_size[stage], pca_net.num_filters[stage]))
        if stage != pca_net.num_stages - 1:
            out_img, img_idx = pca_output(out_img, img_idx, pca_net.patch_size[stage], pca_net.num_filters[stage], v[stage])
        pass

    f = []
    if idx_ext == 1:
        f = []
        for idx in range(len(in_img)):
            if np.mod(idx, display_number) == 0:
                print('Extracting pca_net feature of the  %d th training sample...' % idx)
            out_img_index, = np.where(np.array(img_idx) == idx)
            out_img_tmp = []
            for i in out_img_index:
                out_img_tmp.append(out_img[i])
            out_img_i, img_idx_i = pca_output(out_img_tmp, np.zeros((len(out_img_index), 1), dtype='int'), pca_net.patch_size[-1], pca_net.num_filters[-1], v[-1])
            f_idx = hashing_hist(pca_net, img_idx_i, out_img_i)
            f.append(f_idx[0])
        pass

    return f, v


def hashing_hist(pca_net, img_idx, out_img):
    num_images = int(np.max(img_idx)) + 1
    map_weights = 2 ** np.array(range(pca_net.num_filters[-1] - 1, -1, -1))
    patch_step = (1 - pca_net.blk_overlap_ratio) * np.array(pca_net.hist_block_size)
    patch_step = [int(round(n, 0)) for n in patch_step]

    f = []
    bins = []
    hist_size = 2 ** pca_net.num_filters[-1]
    num_os = 0.0
    for idx in range(num_images):
        idx_span = np.where(np.array(img_idx) == idx)
        idx_span = idx_span[0]
        num_os = len(idx_span) // pca_net.num_filters[-1]
        for i in range(num_os):
            t = np.zeros(out_img[0].shape)
            for j in range(pca_net.num_filters[-1]):
                sign_map = np.sign(out_img[pca_net.num_filters[-1] * i + j])
                sign_map[sign_map <= 0] = 0
                t += map_weights[j] * sign_map
            tt = im_to_col_mean_removal(t, pca_net.hist_block_size, patch_step, remove_mean=False)
            bins = tt.shape[1]
            for k in range(bins):
                b_hist_temp, bins_temp = np.histogram(tt[:, k], range(hist_size + 1))
                b_hist_temp = np.asarray(b_hist_temp)
                f.append(b_hist_temp * (hist_size / sum(b_hist_temp)))
            pass
        pass

    return np.array(f).reshape(num_images, bins * num_os * hist_size)
