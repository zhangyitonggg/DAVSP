import torch
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate average hidden states from a saved model.")
    parser.add_argument("--path", type=str, required=True, help="Path to the saved model file.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the average hidden states.")
    args = parser.parse_args()
        
    path = args.path
    save_path = args.save_path

    res = torch.load(path, map_location='cpu')

    layer_sum = {}
    layer_count = {}

    for sample_name, layer_dict in res.items():
        for layer_name, hidden in layer_dict.items():
            hidden = hidden.float()
            if not torch.isfinite(hidden).all():
                print(f"error: {sample_name} - {layer_name}")
                continue
            if layer_name not in layer_sum:
                layer_sum[layer_name] = hidden.clone()
                layer_count[layer_name] = 1
            else:
                layer_sum[layer_name] += hidden
                layer_count[layer_name] += 1

    layer_avg = {layer: layer_sum[layer] / layer_count[layer] for layer in layer_sum}
    torch.save(layer_avg, save_path)

    for layer_name, avg_hidden in layer_avg.items():
        print(f"{layer_name}: mean={avg_hidden.mean():.4f}, std={avg_hidden.std():.4f}, norm={avg_hidden.norm():.4f}")
