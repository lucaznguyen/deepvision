def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def flops_placeholder(model):
    """
    Real FLOPs counting (fvcore or ptflops) requires extra deps.
    For now we expose parameter count which is often enough.
    """
    return count_trainable_params(model)
