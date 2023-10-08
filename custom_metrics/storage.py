BYTES_TO_KB_CONVERSION_FACTOR = 1024
BYTES_TO_MB_CONVERSION_FACTOR = (BYTES_TO_KB_CONVERSION_FACTOR) ** 2

DECISION_TREE_NUMBER_OF_REPRESENTATION_VECTORS = 5

def pytorch_compute_model_size_mb(model):
    """Compute a pytorch model size in megabytes"""
    # Reference
    # https://discuss.pytorch.org/t/finding-model-size/130275
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_param_buffer_mb = (param_size + buffer_size) / BYTES_TO_MB_CONVERSION_FACTOR

    return size_param_buffer_mb


def sklearn_random_forest_compute_model_size_mb(model):
    """Compute a sklearn random forest model size in megabytes"""
    # Reference
    # https://stackoverflow.com/questions/51139875/sklearn-randomforestregressor-number-of-trainable-parameters

    # Each binary tree is described by left & righ children, feature, threshold, and value for each node.
    n_params = sum(tree.tree_.node_count for tree in model.estimators_) * DECISION_TREE_NUMBER_OF_REPRESENTATION_VECTORS

    model_size_mb = (n_params * np.dtype(np.float32).itemsize ) / BYTES_TO_MB_CONVERSION_FACTOR

    return model_size_mb
