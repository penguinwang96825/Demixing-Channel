import torch
import torch.nn as nn


class Sub(nn.Module):

    def __init__(self, embed_dim):
        super(Sub, self).__init__()
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, mixtures, itf_speaker):
        # mixtures = [batch, length, embed_dim]
        # itf_speaker = [batch, length, embed_dim]
        x = torch.sub(mixtures, itf_speaker)
        x = self.fc(x)
        return x


class Mul(nn.Module):
    """
    Parameters
    ----------
    mixtures: torch.tensor
        [batch, length, embed_dim]
    itf_speaker: torch.tensor
        [batch, length, embed_dim]
    """
    def __init__(self, embed_dim):
        super(Mul, self).__init__()
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, mixtures, itf_speaker):
        x = torch.mul(mixtures, itf_speaker)
        # x = mixtures * itf_speaker
        x = self.fc(x)
        return x


class Concat1(nn.Module):
    """
    Parameters
    ----------
    mixtures: torch.tensor
        [batch, length, embed_dim]
    itf_speaker: torch.tensor
        [batch, length, embed_dim]
    """
    def __init__(self, embed_dim):
        super(Concat1, self).__init__()
        self.fc = nn.Linear(2*embed_dim, embed_dim)

    def forward(self, mixtures, itf_speaker):
        x = torch.cat((mixtures, itf_speaker), axis=-1)
        x = self.fc(x)
        return x


class Concat2(nn.Module):
    """
    Parameters
    ----------
    mixtures: torch.tensor
        [batch, length, embed_dim]
    itf_speaker: torch.tensor
        [batch, length, embed_dim]
    """
    def __init__(self, embed_dim):
        super(Concat2, self).__init__()
        self.fc1 = nn.Linear(2*embed_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, embed_dim)

    def forward(self, mixtures, itf_speaker):
        x = torch.cat((mixtures, itf_speaker), axis=-1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x


class ShareConcat(nn.Module):
    """
    Parameters
    ----------
    mixtures: torch.tensor
        [batch, length, embed_dim]
    itf_speaker: torch.tensor
        [batch, length, embed_dim]
    """
    def __init__(self, embed_dim):
        super(ShareConcat, self).__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.fc2 = nn.Linear(2*embed_dim, embed_dim)

    def forward(self, mixtures, itf_speaker):
        mixtures = nn.ReLU()(self.fc1(mixtures))
        itf_speaker = nn.ReLU()(self.fc1(itf_speaker))
        x = torch.cat((mixtures, itf_speaker), axis=-1)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x


class SeparateConcat(nn.Module):
    """
    Parameters
    ----------
    mixtures: torch.tensor
        [batch, length, embed_dim]
    itf_speaker: torch.tensor
        [batch, length, embed_dim]
    """
    def __init__(self, embed_dim):
        super(SeparateConcat, self).__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, embed_dim)
        self.fc3 = nn.Linear(2*embed_dim, embed_dim)

    def forward(self, mixtures, itf_speaker):
        mixtures = nn.ReLU()(self.fc1(mixtures))
        itf_speaker = nn.ReLU()(self.fc2(itf_speaker))
        x = torch.cat((mixtures, itf_speaker), axis=-1)
        x = nn.ReLU()(x)
        x = self.fc3(x)
        return x