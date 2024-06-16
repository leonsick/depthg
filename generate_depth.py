# The purpose of this script is to generate the depth map for each of the images in a dataset
# To start with, we will use Depth-M12-N to generate the depth maps

import argparse
import sys
from pathlib import Path

import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, ToPILImage
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

from src.zoedepth.builder import build_model
from src.zoedepth.utils.config import get_config
from src.data import *
from src.utils import ToTargetTensor

class DatasetPathReturn(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

        if type(dataset) == Cityscapes:
            self.dataset.filepaths = [self.dataset.images[i] for i in range(len(self.dataset))]
        elif type(dataset) == PascalVOC:
            self.dataset.filepaths = [self.dataset.images[i] for i in range(len(self.dataset))]

        # Assert that self.dataset has the attribute filepaths and throw an error if it doesn't
        assert hasattr(self.dataset, "filepaths"), "Dataset must have attribute filepaths"

    def __getitem__(self, index):
        return self.dataset[index], self.dataset.filepaths[index]

    def __len__(self):
        return len(self.dataset)

class ImageFolderPathReturn(ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder

    """
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)


        return (sample, target), path

def get_args_parser():
    parser = argparse.ArgumentParser("Depth", add_help=False)

    parser.add_argument("--model", default="zoedepth", help="select the model, zoedepth or midas")
    parser.add_argument("--data_dir", default="", help="path to dataset")
    parser.add_argument("--dataset", default="imagefolder", choices=["cocostuff", "potsdam", "cityscapes", "imagefolder", "nyuv2", "pascalvoc"], help="path to dataset")
    parser.add_argument("--split", default="val", help="Dataset split for which to compute depth")
    parser.add_argument("--output_dir", default="", help="path where to save")
    parser.add_argument("--save_features", action="store_true", help="save features in output_dir")

    return parser

def main(args):
    print(args)

    # Make sure to use timm version timm==0.6.7
    # Otherwise use timm==0.9.0

    # Also use torch==2.0.1 or manually fix bug in torchub download

    # device selection
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = 1

    if args.model == "zoedepth":
        # ZoeD_N
        conf = get_config("zoedepth", "infer")
        model_zoe_n = build_model(conf)

        zoe = model_zoe_n.to(DEVICE)
        zoe.eval()

        print(type(zoe))
        print(type(zoe.core))



        transform = Compose([
            ToTensor()
        ])

    elif args.model == "midas":
        # Midas
        model_type = "DPT_Large"  # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
        midas = torch.hub.load("intel-isl/MiDaS", model_type)
        midas.to(DEVICE)
        midas.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        """if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            transform = midas_transforms.dpt_transform

        else:
            transform = midas_transforms.small_transform"""
        transform = Compose([
            ToTensor()
        ])

    else:
        raise ValueError("Model not supported, please choose between zoedepth and midas")

    print(transform)

    # Standard ImageNet normalization
    # Resize((256, 256)),


    # Define dataloader for images
    if args.dataset == "imagefolder":
        dataset = ImageFolderPathReturn(args.data_dir, transform=transform)
        #dataset = ImageFolderPathReturn(dataset)
        print("Image Folder dataset loaded with length: ", len(dataset))
    elif args.dataset == "potsdam":
        dataset = Potsdam(args.data_dir, image_set=args.split, transform=transform, target_transform=ToTargetTensor(), coarse_labels=False)
        dataset = DatasetPathReturn(dataset)
    elif args.dataset == "cityscapes":
        dataset = Cityscapes(args.data_dir, transform=transform, target_transform=ToTargetTensor())
        dataset = DatasetPathReturn(dataset)
    elif args.dataset == "cocostuff":
        dataset = Coco(args.data_dir, transform=transform, target_transform=ToTargetTensor(), image_set=args.split, coarse_labels=False, exclude_things=False)
        dataset = DatasetPathReturn(dataset)
        print("COCO dataset loaded with length: ", len(dataset))
    elif args.dataset == "nyuv2":
        dataset = NYUv2(args.data_dir, transform=transform, target_transform=ToTargetTensor(), image_set=args.split)
        dataset = DatasetPathReturn(dataset)
    elif args.dataset == "pascalvoc":
        dataset = PascalVOC.PascalVOCGenerator(args.data_dir, transform=transform, target_transform=ToTargetTensor(), image_set=args.split)
        dataset = DatasetPathReturn(dataset)
    else:
        raise NotImplementedError("Dataset not supported")

    # Exclude classes that contain the word "subset" in the folder name

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

    prev_feat = 0

    # Depth inference
    for data, paths in tqdm(dataloader):
        # .infer() allows zoe to take batched input of size (N, 3, H, W)
        # The output is a list of depth maps of size (N, 1, H, W)

        images = data[0]

        images = images.to(DEVICE)

        if args.model == "zoedepth":
            depth_tensor_batched, feats = zoe.infer(images, return_feats=True)
        else:
            depth_tensor_batched = midas(images)

        #out = [zoe.core.core_out[k] for k in zoe.core.layer_names]

        ks = list(zoe.core.core_out.keys())

        #print(ks)
        #for k in ks:
        #    print(k, zoe.core.core_out[k].size())


        """
        ['l4_rn', 'r4', 'r3', 'r2', 'r1', 'out_conv']
        l4_rn torch.Size([1, 256, 12, 16])
        r4 torch.Size([1, 256, 24, 32])
        r3 torch.Size([1, 256, 48, 64])
        r2 torch.Size([1, 256, 96, 128])
        r1 torch.Size([1, 256, 192, 256])
        out_conv torch.Size([1, 32, 384, 512])

        """

        # Save depth maps parallelized in output_dir
        for j, depth_tensor in enumerate(depth_tensor_batched):
            #print(depth_tensor.size())
            #print(feats.size())
            if args.model == "midas":
                # Min-max normalization
                depth_tensor = (depth_tensor - depth_tensor.min()) / (depth_tensor.max() - depth_tensor.min())

                # Invert the depth map
                depth_tensor = 1 - depth_tensor

            path = Path(paths[j])

            # Get only the filename
            filename = path.stem

            # Get the direct folder name
            folder = path.parent.name

            # Add the folder and zoedepth to the filename of the path
            folder_path = Path(args.output_dir) / folder

            # If the folder does not exist, create it
            if not folder_path.exists():
                folder_path.mkdir(parents=True, exist_ok=True)

            #path = Path(args.output_dir) / folder / (filename + f"_{args.model}.pt")

            # Save the depth map as .pt file
            #torch.save(depth_tensor.detach().cpu(), path)

            if args.save_features:
                # Save intermediate features
                for k in ks:
                    path = Path(args.output_dir) / folder / (filename + f"_{k}.pt")
                    torch.save(zoe.core.core_out[k].squeeze(0).detach().cpu(), path)

                #print(zoe.core.core_out["out_conv"].squeeze(0).detach().cpu().shape)

                prev_feat = zoe.core.core_out["out_conv"].squeeze(0).detach().cpu()



            # Save the depth map as .png file
            depth = depth_tensor.detach().cpu().numpy()
            depth = np.squeeze(depth)

            if not args.model == "midas":
                depth = (depth - depth.min()) / (depth.max() - depth.min())

            depth = (depth * 255).astype(np.uint8)
            depth = Image.fromarray(depth)
            depth.save(path.with_suffix(".png"))

            # sys.exit(0)

            # Load the depth map and convert to numpy array
            # depth = torch.load(path)
            # depth = np.array(depth)
            # colored = colorize(depth, cmap="gray_r")

            # Show the depth map
            # plt.imshow(colored)
            # plt.show()





if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)


