import argparse
from io import BytesIO
import json
import os
from typing import Optional

import numpy as np
from PIL import Image
import requests
import torch
import torch.optim as optim
from torchvision import transforms, models


def prepare_image(
    img: Image.Image,
    max_size: int = 400,
    shape: Optional[int] = None
) -> torch.Tensor:
    """Transform an image to tensor"""
    size = max(max(img.size), max_size)
    if shape is not None:
        size = shape
    in_transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]
    )

    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = in_transform(img)[:3, :, :].unsqueeze(0)

    return image


def load_image_from_file(img_path: str) -> Image.Image:
    return Image.open(img_path).convert('RGB')


def img_convert(tensor: torch.Tensor) -> np.array:
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array(
        (0.485, 0.456, 0.406)
    )
    image = image.clip(0, 1)
    image = Image.fromarray((image * 255).astype(np.uint8))
    return image


def get_model() -> torch.nn.modules:
    vgg = models.vgg19(pretrained=True).features
    for param in vgg.parameters():
        param.requires_grad_(False)
    return vgg


def get_features(
    image: torch.Tensor, model: torch.nn.modules, layers: dict = None
) -> dict:
    """ Run an image forward through a model and get the features for
        a set of layers. Default layers are for VGGNet matching Gatys et al (2016)
    """
    if layers is None:
        layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2',
            '28': 'conv5_1'
        }

    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features


def gram_matrix(tensor: torch.Tensor) -> dict:
    """ Calculate the Gram Matrix of a given tensor
        Gram Matrix: https://en.wikipedia.org/wiki/Gramian_matrix
    """
    batch_size, d, h, w = tensor.size()
    tmp_tensor = tensor.view(batch_size * d, h * w)
    gram = torch.mm(tmp_tensor, tmp_tensor.t())
    return gram


def train(config: dict) -> None:
    optimizer = optim.Adam([config["target_image"]], lr=0.003)
    for ii in range(1, config["steps"] + 1):
        target_features = get_features(config["target_image"], config["model"])
        content_loss = torch.mean(
            (
                target_features['conv4_2'] -
                config["content_features"]['conv4_2']
            )**2
        )
        style_loss = 0
        for layer in config["style_weights"]:
            target_feature = target_features[layer]
            _, d, h, w = target_feature.shape
            target_gram = gram_matrix(target_feature)
            style_gram = config["style_grams"][layer]
            layer_style_loss = config["style_weights"][layer] * torch.mean(
                (target_gram - style_gram)**2
            )
            style_loss += layer_style_loss / (d * h * w) / (d * h * w)
        total_loss = (
            content_loss * config["content_weight"] +
            style_loss * config["style_weight"]
        )
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        if ii % config["show_every"] == 0:
            img_converted = img_convert(config["target_image"])
            config["result_images"].append(img_convert(config["target_image"]))
            if config["save_to_file"]:
                path_to_save = os.path.join(
                    config["result_path"], f"result_{ii}.jpg"
                )
                img_converted.save(path_to_save)
    config["result_images"].append(img_convert(config["target_image"]))


def get_config(path="config.json") -> dict:
    with open(path, "r") as f:
        config = json.load(f)

    config["device"] = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    config["model"] = get_model().to(config["device"])
    config["result_images"] = []
    return config


def prepare_config_to_predict(
    config: dict, content_image: Image.Image, style_image: Image.Image
) -> dict:
    config["content_image"] = prepare_image(content_image).to(config["device"])
    config["style_image"] = prepare_image(
        style_image, shape=config["content_image"].shape[-2:]
    ).to(config["device"])
    config["target_image"] = config["content_image"].clone(
    ).requires_grad_(True).to(config["device"])

    config["content_features"] = get_features(
        config["content_image"], config["model"]
    )
    config["style_features"] = get_features(
        config["style_image"], config["model"]
    )
    config["style_grams"] = {
        layer: gram_matrix(config["style_features"][layer])
        for layer in config["style_features"]
    }
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="config.json", help="path to config.json file"
    )
    args = parser.parse_args()
    config = get_config(args.config)
    config = prepare_config_to_predict(
        config, load_image_from_file(config["content_image_path"]),
        load_image_from_file(config["style_image_path"])
    )
    config["save_to_file"] = True
    train(config)


if __name__ == "__main__":
    main()