#!/usr/bin/env python3
from argparse import ArgumentParser
from unet_spatial_protocol import prepare_persistent_split


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare_persistent_split(args.images, args.masks, args.output, args.seed)
