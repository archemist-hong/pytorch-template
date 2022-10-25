def count_parameters(model, requires_grad):
    return sum(p.numel() for p in model.parameters() if p.requires_grad == requires_grad)