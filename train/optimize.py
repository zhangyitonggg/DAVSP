from utils import *

from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
import torch
import os
import json
from tqdm import tqdm

from PIL import Image
from torchvision.transforms import ToTensor, Resize, Compose
import argparse
import torch.nn.functional as F

from torchvision.transforms import Resize

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import transforms

import random
from itertools import cycle

from torch.utils.data import ConcatDataset


class TrainDataset(Dataset):
    def __init__(self, json_path, image_processor):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.keys = list(self.data.keys())
        self.image_processor = image_processor
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        item = self.data[key]

        image = Image.open(item["image_path"]).convert("RGB")
        image = pad_to_square(image)
        image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'].squeeze(0)  # [3, H, W]


        question = item["question"]
        answer = item["answer"] if "answer" in item else ""
        return {
            "image": image,
            "question": question,
            "answer": answer,
            "id": key
        }


def make_input_ids(query: str):
    qs = DEFAULT_IMAGE_TOKEN + "\n" + apply_safety_prompt(query)

    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return tokenizer_image_token(
        prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).to(device)  # shape [seq_len]


def create_border_mask(height, width, padding=30):
    mask = torch.zeros(1, 1, height, width)
    mask[:, :, :padding, :] = 1
    mask[:, :, -padding:, :] = 1
    mask[:, :, :, :padding] = 1
    mask[:, :, :, -padding:] = 1
    return mask


def embed_image_into_noise(noise: torch.Tensor, images: torch.Tensor, padding: int) -> torch.Tensor:
    B, _, _, _ = images.shape
    _, _, H, W = noise.shape
    inner_h = H - 2 * padding
    inner_w = W - 2 * padding
    # device = images.device

    resize_fn = Resize((inner_h, inner_w))
    resized_images = resize_fn(images)  # [B, 3, inner_h, inner_w]

    # input_images = noise.clone().expand(B, -1, -1, -1)  # [B, 3, H, W]
    input_images = noise.repeat(B,1,1,1).clone()
    input_images[:, :, padding:padding+inner_h, padding:padding+inner_w] = resized_images

    return input_images


def compute_align_loss(model, raw_images, perturbed_images):
    with torch.no_grad():
        # clean_feat = model.model.vision_tower(raw_images).last_hidden_state  # [B, N, D]
        clean_feat = model.model.vision_tower(raw_images) # [B, N, D]
        # print(type(clean_feat), clean_feat.shape)
        clean_repr = clean_feat.mean(dim=1)  # [B, D]

    perturbed_feat = model.model.vision_tower(perturbed_images)
    perturbed_repr = perturbed_feat.mean(dim=1)  # [B, D]

    diff = perturbed_repr - clean_repr  # [B, D]
    align_loss = torch.norm(diff, dim=-1).mean()  

    return align_loss


def compute_output_loss(model, tokenizer, questions, answers, images, max_len=1024):
    device = images.device
    B = len(questions)
    losses = []

    for i in range(B):
        qs = DEFAULT_IMAGE_TOKEN + "\n" + apply_safety_prompt(questions[i])
        answer = answers[i]

        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_prompt_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').to(device)
        target_ids = tokenizer(answer, return_tensors='pt', add_special_tokens=False)['input_ids'][0].to(device)

        input_full_ids = torch.cat([input_prompt_ids, target_ids], dim=0)
        labels = torch.full_like(input_full_ids, -100)
        labels[-target_ids.shape[0]:] = target_ids

        padding = max_len - input_full_ids.shape[0]
        if padding > 0:
            input_full_ids = torch.cat([input_full_ids, torch.full((padding,), 0, dtype=torch.long, device=device)])
            labels = torch.cat([labels, torch.full((padding,), -100, dtype=torch.long, device=device)])
        else:
            input_full_ids = input_full_ids[:max_len]
            labels = labels[:max_len]

        input_ids = input_full_ids.unsqueeze(0)
        labels = labels.unsqueeze(0)
        image = images[i].unsqueeze(0)

        outputs = model(input_ids=input_ids, images=image, labels=labels, use_cache=False)
        
        losses.append(outputs.loss)


    return torch.stack(losses).mean()


def compute_projection_loss(model, tokenizer, questions, images, vector_norm_dict, is_harmful):
    device = images.device
    losses = []

    for i in range(len(questions)):
        input_ids = make_input_ids(questions[i]).to(device)  # [T]
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        image = images[i].unsqueeze(0)  # [1, 3, H, W]

        outputs = model(
            input_ids=input_ids,
            images=image,
            output_hidden_states=True,
            use_cache=False
        )

        hidden_states = outputs.hidden_states  # list of [1, T, D]

        per_sample_losses = []


        layers = [20]

        for layer_idx in layers:
            token_vec = hidden_states[layer_idx][:, -1, :]  # [B, D]
            direction = vector_norm_dict[f"layer_{layer_idx}"]  # [D]
            direction = direction.to(token_vec.device)

            proj = torch.matmul(token_vec, direction)  # [B]

            print(f"isHarmful {is_harmful}  Layer {layer_idx} projection: {proj.item()}")

            if is_harmful:
                max_value = args.top_threshold 
                per_sample_losses.append(torch.clamp(max_value - proj.mean(), min=0.0))
            else:
                min_value = args.bottom_threshold
                per_sample_losses.append(torch.clamp(proj.mean() - min_value, min=0.0))

        losses.append(torch.stack(per_sample_losses).mean())

    return torch.stack(losses).mean()


def compute_loss(isHarmful, model, tokenizer, image_processor, raw_images, images, questions, answers, vector_norm_dict, args):
    if args.align_weight > 0:
        align_loss = compute_align_loss(model, raw_images, images)
    else:
        align_loss = torch.tensor(0.0).to(images.device)

    if args.output_weight > 0 and isHarmful == False:
        output_loss = compute_output_loss(model, tokenizer, questions, answers, images, max_len=args.max_len)
    elif args.output_weight > 0 and isHarmful == True:
        p = random.random()
        if p < 0.05:
           output_loss = compute_output_loss(model, tokenizer, questions, answers, images, max_len=args.max_len)
        else:
           output_loss = torch.tensor(0.0).to(images.device)
    else:
        output_loss = torch.tensor(0.0).to(images.device)

    if args.proj_weight > 0:
        projection_loss = compute_projection_loss(model, tokenizer, questions, images, vector_norm_dict, isHarmful)
    else:
        projection_loss = torch.tensor(0.0).to(images.device)

    total_loss = torch.tensor(0.0).to(images.device)
    if args.proj_weight > 0:
        total_loss += args.proj_weight * projection_loss
    if args.align_weight > 0:
        total_loss += args.align_weight * align_loss
    if args.output_weight > 0:
        total_loss += args.output_weight * output_loss

    return total_loss, align_loss, output_loss, projection_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--harmful_dir", type=str, help="Path to the harmful dataset JSON file.")
    parser.add_argument("--benign_dir", type=str, help="Path to the benign dataset JSON file.")
    parser.add_argument("--padding", type=int, default=30, help="padding size for the border mask.")
    parser.add_argument("--proj_weight", type=float, default=1, help="projection loss weight.")
    parser.add_argument("--output_weight", type=float, default=0.1, help="output loss weight.")
    parser.add_argument("--align_weight", type=float, default=0.0, help="alignment loss weight.")
    parser.add_argument("--vector_path", type=str, help="Path to the harmfulness vector file.")
    parser.add_argument("--top_threshold", type=float, default=27, help="Top threshold for harmfulness.")
    parser.add_argument("--bottom_threshold", type=float, default=-2, help="Bottom threshold for harmfulness.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training.")
    parser.add_argument("--steps", type=int, default=1200, help="Number of training steps.")
    parser.add_argument("--max_len", type=int, default=1024, help="Max length for input.")
    parser.add_argument("--model_path", type=str, help="Path to the model.")
    parser.add_argument("--save_dir", type=str, help="Path to the output directory.")
    parser.add_argument("--alpha", type=str, default="1/255", help="step size for optimization.")    
    args = parser.parse_args()

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "args.txt"), "w") as f:
        f.write(str(args))

    noise_dir = f"{save_dir}/noise"
    os.makedirs(noise_dir, exist_ok=True)

    loss_path = os.path.join(save_dir, "loss.txt")
    with open(loss_path, "w") as f:
        f.write("")
    
    tokenizer, model, image_processor = init_model(args.model_path)
    model.eval()
    model.requires_grad_(False)
    model.model.vision_tower.requires_grad_(False)

    device = next(model.parameters()).device

    raw_vector_dict = torch.load(args.vector_path)  # e.g. {"1": [D], "2": [D], ...}
    vector_norm_dict = {k: (v / v.norm()).half() for k, v in raw_vector_dict.items()}

    H, W = 336, 336
    C = 3

    noise = torch.zeros((1, C, H, W), dtype=torch.float16, device=device, requires_grad=True)

    mask = create_border_mask(H, W, padding=args.padding).to(device) # 1, 1, H, W
    mask = mask.expand_as(noise)  

    alpha = eval(args.alpha)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
    norm_alpha = alpha / std  # (1, 3, 1, 1)

    HarmfulDataset = TrainDataset(args.harmful_dir, image_processor)
    Harmfulloader = DataLoader(HarmfulDataset, batch_size=args.batch_size, shuffle=True)    
    BenignDataset = TrainDataset(args.benign_dir, image_processor)
    Benignloader = DataLoader(BenignDataset, batch_size=args.batch_size, shuffle=True)
    harmful_iter = cycle(Harmfulloader)
    benign_iter = cycle(Benignloader)

    count = 0
    for step in range(args.steps):
        p = random.random()
        if p < 0.5:
            batch = next(harmful_iter)
            isHarmful = True
        else:
            batch = next(benign_iter)
            isHarmful = False
    
        noise.requires_grad_(True)

        raw_images = batch["image"].to(device)
        questions = batch["question"]
        answers = batch["answer"]

        images = embed_image_into_noise(noise, raw_images, args.padding)

        loss, align_loss, output_loss, projection_loss = compute_loss(isHarmful, model, tokenizer, image_processor, raw_images, images, questions, answers, vector_norm_dict, args)
        loss.backward()

        with torch.no_grad():
            noise -= norm_alpha * noise.grad.sign() * mask
            noise.grad.zero_()

        count += images.shape[0] 
        with open(loss_path, "a") as f:
            f.write(f"Step {step}, Count {count}, isHarmful={isHarmful}, loss={loss.item():.7f}, output_loss={output_loss.item():.7f}, projection_loss={projection_loss.item():.7f}\n")

        del loss, align_loss, output_loss, projection_loss 
        torch.cuda.empty_cache()  

        if (step + 1) % 50 == 0:
            with open(os.path.join(noise_dir, f"noise-{step + 1}.pt"), "wb") as f:
                torch.save(noise, f)

    print("Done!")