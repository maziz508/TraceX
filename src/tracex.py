import torch

def tracex_output(model, x, eps=1e-2):
    """
    x: tensor (D,) single input
    returns phi: tensor (D,)
    """
    D = x.shape[0]
    phi = torch.zeros_like(x)
    f0 = model(x.unsqueeze(0)).item()
    for i in range(D):
        x_p = x.clone(); x_m = x.clone()
        x_p[i] += eps; x_m[i] -= eps
        f_p = model(x_p.unsqueeze(0)).item()
        f_m = model(x_m.unsqueeze(0)).item()
        phi[i] = (f_p - f_m) / (2*eps)
    return phi

def tracex_layerwise(model, x, eps=1e-2):
    """
    Returns dict[layer_name -> delta tensor (D,)]
    Requires model.register_forward_hook on each sublayer.
    """
    activations = {}
    deltas = {}

    def make_hook(name):
        def hook(m, inp, out):
            activations[name] = out.detach().clone()
        return hook

    # attach hooks to each submodule you care about
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) or 'encoder' in name:
            m.register_forward_hook(make_hook(name))

    # baseline forward
    _ = model(x.unsqueeze(0))
    base = activations.copy()

    # for each feature
    for i in range(x.shape[0]):
        x_p = x.clone(); x_m = x.clone()
        x_p[i] += eps; x_m[i] -= eps
        activations.clear()
        _ = model(x_p.unsqueeze(0))
        act_p = activations.copy()
        activations.clear()
        _ = model(x_m.unsqueeze(0))
        act_m = activations.copy()

        for name in base:
            delta = (act_p[name] - act_m[name]).abs().mean().item() / (2*eps)
            deltas.setdefault(name, []).append(delta)

    # convert lists → tensors
    for name in deltas:
        deltas[name] = torch.tensor(deltas[name])
    return deltas
