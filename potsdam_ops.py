from PIL import Image
from scipy.io import loadmat
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image, to_tensor
import argparse
import os
import torch


def convert_mat(args):
    # List files in folder
    files = os.listdir(args.folder)

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Loop through files
    for file in tqdm(files):
        # Load mat file
        try:
            img = loadmat(os.path.join(args.folder, file))["img"]
        except:
            print("Error loading file: ", file)
            continue

        img = to_pil_image(torch.from_numpy(img).permute(2, 0, 1)[:3])

        # Save as png
        img.save(os.path.join(args.output_dir, file[:-4] + ".png"))

    print("Done!")


def convert_coco_depth_map(args):
    # python potsdam_ops.py --task convert_coco_depth_map --folder /Users/leonsick/Downloads/depth_samples/
    # List files in folder
    files = os.listdir(args.folder)

    # In the folder, make a new directory called "processed"
    if not os.path.exists(os.path.join(args.folder, "processed")):
        os.makedirs(os.path.join(args.folder, "processed"))

    output_dir = os.path.join(args.folder, "processed")

    # Loop through files
    for file in tqdm(files):
        # Load image with PIL
        img = Image.open(os.path.join(args.folder, file))

        # Resize to 224, 224 and convert to tensor
        img = to_tensor(img.resize((224, 224)))

        if file.__contains__("kbr"):
            img = torch.mean(img, dim=0).unsqueeze(0)

            img = (img - img.min()) / (img.max() - img.min())

            img = 1 - img

        elif file.__contains__("midas"):
            img = 1 - img
        elif file.__contains__("zoedepth"):
            img = 1 - img

        img = to_pil_image(img)

        # Save as png
        img.save(os.path.join(output_dir, file[:-4] + ".png"))

    print("Done!")


def match_images(args):
    # Compare all images in folder with all images in comp_folder
    # List files in folder with ext .mat
    files = os.listdir(args.folder)
    files = [file for file in files if file.endswith(".mat")]

    # List files in comp_folder with ext .mat
    comp_files = os.listdir(args.comp_folder)
    comp_files = [file for file in comp_files if file.endswith(".mat")]

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for file in tqdm(files):
        anchor_img = loadmat(os.path.join(args.folder, file))["img"]
        anchor_img = to_tensor(to_pil_image(torch.from_numpy(anchor_img).permute(2, 0, 1)[:3]))

        for comp_file in comp_files:
            comp_img = loadmat(os.path.join(args.comp_folder, comp_file))["img"]
            comp_img = to_tensor(to_pil_image(torch.from_numpy(comp_img).permute(2, 0, 1)[:3]))

            if torch.all(torch.eq(anchor_img, comp_img)):
                print(file, comp_file)

                # Get rename file path
                rename_file_path = os.path.join(args.rename_folder, comp_file[:-4] + ".png")

                # Get output file path
                output_file_path = os.path.join(args.output_dir, file[:-4] + ".png")

                # Copy file
                os.system(f"cp {rename_file_path} {output_file_path}")

                # Rename file
                # os.rename(rename_file_path, output_file_path)
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", default="convert_mat", help="task to perform")
    parser.add_argument("--folder", default="", help="path to mat file")
    parser.add_argument("--comp_folder", default="", help="path where to compare")
    parser.add_argument("--rename_folder", default="", help="path where to rename")
    parser.add_argument("--output_dir", default="", help="path where to save")

    args = parser.parse_args()

    if args.task == "convert_mat":
        convert_mat(args)
    elif args.task == "match_images":
        match_images(args)
    elif args.task == "convert_coco_depth_map":
        convert_coco_depth_map(args)
