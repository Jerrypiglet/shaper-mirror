import torch
import torch.nn.functional as F


def encode_one_hot(target, num_classes):
    """Encode integer labels into one-hot vectors

    Args:
        target (torch.Tensor): (N,)
        num_classes (int): the number of classes

    Returns:
        torch.FloatTensor: (N, C)

    """
    one_hot = target.new_zeros(target.size(0), num_classes)
    one_hot = one_hot.scatter(1, target.unsqueeze(1), 1)
    return one_hot.float()


def smooth_cross_entropy(input, target, label_smoothing):
    """Cross entropy with label smoothing

    Args:
        input (torch.Tensor): (N, C)
        target (torch.Tensor): (N,)
        label_smoothing (float):

    Returns:
        loss (torch.Tensor): scalar

    """
    assert input.dim() == 2 and target.dim() == 1
    assert isinstance(label_smoothing, float)
    batch_size, num_classes = input.shape
    one_hot = torch.zeros_like(input).scatter(1, target.unsqueeze(1), 1)
    smooth_one_hot = one_hot * (1 - label_smoothing) + torch.ones_like(input) * (label_smoothing / num_classes)
    log_prob = F.log_softmax(input, dim=1)
    loss = F.kl_div(log_prob, smooth_one_hot, reduction="none").sum(1).mean()
    return loss


def test_smooth_cross_entropy():
    num_samples = 4
    num_classes = 10
    target = torch.randint(num_classes, (num_samples,)).to(torch.int64)
    print("target:", target)
    print("one_hot:",encode_one_hot(target, num_classes))
    uniform_prob = torch.ones([num_samples, num_classes]) / num_classes
    print("smooth CE:", smooth_cross_entropy(uniform_prob, target, 1.0))


def test_smooth_cross_entropy_2():
    num_samples = 3
    num_classes = 6
    # target = torch.randint(num_classes, (num_samples,)).to(torch.int64)
    target = torch.arange(num_samples)
    print("target:", target)
    print("one_hot:", encode_one_hot(target, num_classes))
    uniform_prob = torch.arange(num_samples - 2, num_classes - 2)
    uniform_prob = encode_one_hot(uniform_prob, num_classes)
    print("logits: ", uniform_prob)
    label_smooth = 0.2
    print('label smooth: ', label_smooth)
    print("smooth CE:", smooth_cross_entropy(uniform_prob, target, label_smooth))

if __name__ == '__main__':
    test_smooth_cross_entropy_2()
