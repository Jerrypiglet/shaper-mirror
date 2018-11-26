import torch

from shaper.nn.functional import pdist
from shaper.nn.functional import encode_one_hot, smooth_cross_entropy


def test_pdist():
    import time
    import numpy as np
    import scipy.spatial.distance as sdist

    batch_size = 16
    num_points = 1024
    channels = 64

    features = np.random.rand(batch_size, channels, num_points)
    features_tensor = torch.from_numpy(features)
    if torch.cuda.is_available():
        features_tensor = features_tensor.cuda()

    # check pairwise distance
    distance = np.stack([sdist.squareform(np.square(sdist.pdist(feat.T))) for feat in features])

    distance_tensor = pdist(features_tensor)  # warm up
    assert np.allclose(distance, distance_tensor.cpu().numpy())

    with torch.no_grad():
        end = time.time()
        for _ in range(10):
            pdist(features_tensor)
        print("pdist benchmark:")
        print("Time: {:.6f}s".format((time.time() - end) / 50),
              "Memory: {}MB".format(torch.cuda.memory_allocated() / 1024 ** 2))


def test_smooth_cross_entropy():
    import numpy as np

    num_samples = 2
    num_classes = 10
    label_smoothing = 0.1

    # numpy version
    target_np = np.random.randint(0, num_classes, [num_samples])
    one_hot_np = np.zeros([num_samples, num_classes])
    one_hot_np[np.arange(num_samples), target_np] = 1.0
    smooth_one_hot = one_hot_np * (1.0 - label_smoothing) + np.ones_like(one_hot_np) * label_smoothing / num_classes
    prob_np = np.ones([num_samples, num_classes]) / num_classes
    cross_entropy_np = - (smooth_one_hot * np.log(prob_np)).sum(1).mean()

    target = torch.from_numpy(target_np)
    prob = torch.from_numpy(prob_np)

    one_hot = encode_one_hot(target, num_classes)
    assert np.allclose(one_hot_np, one_hot.numpy())

    cross_entropy = smooth_cross_entropy(prob, target, label_smoothing)
    assert np.allclose(cross_entropy_np, cross_entropy.numpy())
