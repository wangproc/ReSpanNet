"""Load model tensors with an exact state-dictionary match."""

import inspect
import torch


def read_weights(path):
    options = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        options["weights_only"] = True
    state = torch.load(path, **options)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    if not isinstance(state, dict) or not state:
        raise ValueError("Expected a nonempty tensor state dictionary")
    if not all(isinstance(k, str) and torch.is_tensor(v) for k, v in state.items()):
        raise ValueError("Expected a model state dictionary containing tensors")
    return state


def load_weights(model, path):
    model.load_state_dict(read_weights(path), strict=True)
    return model


def initialize_backbone(backbone, path):
    state = read_weights(path)
    if any(key.startswith('backbone.') for key in state):
        state = {key[len('backbone.'):]: value for key, value in state.items()
                 if key.startswith('backbone.')}
    target = backbone.state_dict()
    features = {key for key in target if not key.startswith('output_norms.')}
    if not features <= state.keys():
        raise ValueError('Classification checkpoint does not match the backbone')
    target.update({key: state[key] for key in features})
    backbone.load_state_dict(target, strict=True)
