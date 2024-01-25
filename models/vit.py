import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

def ViT_pretrained(n_classes):
    # Load the pretrained ViT model
    model = timm.create_model("vit_base_patch16_224", pretrained=True)

    # Replace the classifier head with a new one (adjust the in_features accordingly)
    num_ftrs = model.head.in_features
    model.head = torch.nn.Linear(num_ftrs, n_classes)

    return model



class SupConViT(nn.Module):
    """ViT backbone + projection head for n_classes"""
    def __init__(self, n_classes, head='mlp'):
        super(SupConViT, self).__init__()

        # Load the pretrained ViT model and remove its classification head
        self.encoder = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.encoder.head = nn.Identity()

        dim_in = self.encoder.num_features  # Get the feature dimension from ViT

        # Define the projection head
        if head == 'linear':
            self.head = nn.Linear(dim_in, n_classes)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, n_classes)
            )
        elif head == 'None':
            self.head = None
        else:
            raise NotImplementedError('head not supported: {}'.format(head))

    def forward(self, x):
        # Extract features using the ViT encoder
        feat = self.encoder(x)

        # Apply the projection head if it exists
        if self.head:
            feat = self.head(feat)
        else:
            feat = F.normalize(feat, dim=1)
        return feat

    def features(self, x):
        # Directly use the ViT encoder for extracting features
        return self.encoder(x)
