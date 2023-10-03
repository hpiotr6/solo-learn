from torch import nn


def simclr_proj(features_dim, proj_hidden_dim, proj_output_dim):
    return nn.Sequential(
        nn.Linear(features_dim, proj_hidden_dim),
        nn.ReLU(),
        nn.Linear(proj_hidden_dim, proj_output_dim),
    )


def barlow_proj(features_dim, proj_hidden_dim, proj_output_dim):
    return nn.Sequential(
        nn.Linear(features_dim, proj_hidden_dim),
        nn.BatchNorm1d(proj_hidden_dim),
        nn.ReLU(),
        nn.Linear(proj_hidden_dim, proj_hidden_dim),
        nn.BatchNorm1d(proj_hidden_dim),
        nn.ReLU(),
        nn.Linear(proj_hidden_dim, proj_output_dim),
    )
