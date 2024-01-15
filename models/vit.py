import timm
import torch

def ViT_pretrained(n_classes):
    # Load the pretrained ViT model
    model = timm.create_model("vit_base_patch16_224", pretrained=True)

    # Replace the classifier head with a new one (adjust the in_features accordingly)
    num_ftrs = model.head.in_features
    model.head = torch.nn.Linear(num_ftrs, n_classes)

    return model