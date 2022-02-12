import collections
import torch


def build_linear_relu(in_features, out_features):
    return torch.nn.Sequential(
        torch.nn.Linear(in_features, out_features),
        torch.nn.ReLU()
    )


def freeze(module):
    module.requires_grad_(False)
    module.eval()


def unfreeze(module):
    module.requires_grad_(True)
    module.train()


def extract_descendent_state_dict(state_dict, descendent_name):
    descendent_name_prefix = descendent_name + "."
    descendent_state_dict_items = [
        ( key[len(descendent_name_prefix):], value )
        for key, value in state_dict.items()
        if key.startswith(descendent_name_prefix)
    ]
    descendent_state_dict_metadata_items = [
        ( key[len(descendent_name):].lstrip("."), value )
        for key, value in state_dict._metadata.items()
        if key.startswith(descendent_name)
    ]

    descendent_state_dict = collections.OrderedDict(
        descendent_state_dict_items
    )
    descendent_state_dict._metadata = collections.OrderedDict(
        descendent_state_dict_metadata_items
    )
    return descendent_state_dict

