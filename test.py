import copy
import torch
import torch.nn as nn
import cnn

num_classes = 26
model_loaded = cnn.CNN(
    (28, 28),
    [(3, 1, 1, 32), (3, 1, 1, 32), (3, 1, 1, 64), (3, 1, 1, 64)],
    [0, 2, 0, 2],
    [num_classes],
    1
)
model_loaded.load_state_dict(torch.load(
    "model/cnn_model.pth", map_location="cpu"))
# --- Mapping functions ---
def w_to_G(w):
    """Weight -> Conductance mapping (0,2)."""
    return 2.0 / (1.0 + torch.exp(-w))

def G_to_w(G):
    """Conductance -> Weight mapping (inverse)."""
    return -torch.log((2.0 / G) - 1.0)

def map_model_to_memristor(model: nn.Module):
    """
    Clone a model and replace its weights with memristor-mapped equivalents.
    """
    model_copy = copy.deepcopy(model)

    with torch.no_grad():
        for name, param in model_copy.named_parameters():
            if "weight" in name:  # only map weight matrices, not biases
                G = w_to_G(param.data)
                w_back = G_to_w(G)
                param.copy_(w_back)

    return model_copy


# --- Test the mapping ---
def compare_models(model, mapped_model, sample_input):
    with torch.no_grad():
        out_orig = model(sample_input)
        out_mapped = mapped_model(sample_input)

    print("Original output logits:", out_orig[0].numpy())
    print("Mapped output logits:  ", out_mapped[0].numpy())

    pred_orig = torch.argmax(out_orig, dim=1).item()
    pred_mapped = torch.argmax(out_mapped, dim=1).item()

    print(f"Original prediction: {pred_orig}, Mapped prediction: {pred_mapped}")

    return pred_orig, pred_mapped


# --- Example usage ---
if __name__ == "__main__":
    # make a random "fake" 28x28 input (like an image)
    sample = torch.randn(1, 1, 28, 28)

    # map the model
    mapped_model = map_model_to_memristor(model_loaded)

    # compare predictions
    compare_models(model_loaded, mapped_model, sample)
