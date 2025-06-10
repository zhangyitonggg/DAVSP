import torch
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--harmless_path', type=str, default='')
    parser.add_argument('--harmful_path', type=str, default='')
    parser.add_argument('--save_path', type=str, default='')
    args = parser.parse_args()

    harmless_path = args.harmless_path
    harmful_path = args.harmful_path
    save_path = args.save_path


    harmless_avg = torch.load(harmless_path, map_location='cpu')
    harmful_avg = torch.load(harmful_path, map_location='cpu')

    diff_result = {}
    for layer_name in harmful_avg.keys():
        h1 = harmful_avg[layer_name].float()
        h2 = harmless_avg[layer_name].float()
        diff_result[layer_name] = h1 - h2

    torch.save(diff_result, save_path)

    for layer_name, diff in diff_result.items():
        print(f"{layer_name}: L2 norm = {diff.norm():.4f}")
