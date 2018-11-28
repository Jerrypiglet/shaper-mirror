import torch

from shaper.models.dgcnn_utils.functions import pdist, \
    construct_edge_feature_index, construct_edge_feature_gather, construct_edge_feature


def warmup():
    a = torch.rand(2, 3)
    b = torch.rand(2, 3)
    for _ in range(10):
        c = a + b


def generate_data(batch_size=16, num_points=1024, in_channels=64, k=20):
    with torch.no_grad():
        features_tensor = torch.rand(batch_size, in_channels, num_points).cuda()
        distance = pdist(features_tensor)
        _, knn_inds = torch.topk(distance, k, largest=False)
        return features_tensor, knn_inds


def test_construct_edge_feature():
    features_tensor, knn_inds = generate_data()

    def forward_backward(fn):
        torch.cuda.empty_cache()
        features = torch.zeros_like(features_tensor).copy_(features_tensor)
        features.requires_grad = True
        edge_feature = fn(features, knn_inds)
        edge_feature.backward(torch.ones_like(edge_feature), retain_graph=True, create_graph=False)
        return edge_feature.cpu(), features.grad.cpu()

    o_index, g_index = forward_backward(construct_edge_feature_index)
    o_gather, g_gather = forward_backward(construct_edge_feature_gather)
    o_knn, g_knn = forward_backward(construct_edge_feature)

    # forward
    assert o_index.allclose(o_gather)
    assert o_gather.allclose(o_knn)

    # backward
    assert g_index.allclose(g_gather)
    assert g_gather.allclose(g_knn)
