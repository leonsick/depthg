from modules import *
import os, warnings
from data import ContrastiveSegDataset
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
# from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader
from torchvision.transforms.functional import five_crop, get_image_size, crop
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset

from pytorch_lightning import seed_everything

from zoedepth.builder import build_model
from zoedepth.utils.config import get_config


def _random_crops(img, size, seed, n):
    """Crop the given image into four corners and the central crop.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    .. Note::
        This transform returns a tuseple of images and there may be a
        mismatch in the number of inputs and targets your ``Dataset`` returns.

    Args:
        img (PIL Image or Tensor): Image to be cropped.
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).

    Returns:
       tuple: tuple (tl, tr, bl, br, center)
                Corresponding top left, top right, bottom left, bottom right and center crop.
    """
    if isinstance(size, int):
        size = (int(size), int(size))
    elif isinstance(size, (tuple, list)) and len(size) == 1:
        size = (size[0], size[0])

    if len(size) != 2:
        raise ValueError("Please provide only two dimensions (h, w) for size.")

    image_width, image_height = get_image_size(img)
    crop_height, crop_width = size
    if crop_width > image_width or crop_height > image_height:
        msg = "Requested crop size {} is bigger than input size {}"
        raise ValueError(msg.format(size, (image_height, image_width)))

    images = []
    for i in range(n):
        seed1 = hash((seed, i, 0))
        seed2 = hash((seed, i, 1))
        crop_height, crop_width = int(crop_height), int(crop_width)

        top = seed1 % (image_height - crop_height)
        left = seed2 % (image_width - crop_width)
        images.append(crop(img, top, left, crop_height, crop_width))

    return images


class RandomCropComputer(Dataset):

    def _get_size(self, img):
        if len(img.shape) == 3:
            return [int(img.shape[1] * self.crop_ratio), int(img.shape[2] * self.crop_ratio)]
        elif len(img.shape) == 2:
            return [int(img.shape[0] * self.crop_ratio), int(img.shape[1] * self.crop_ratio)]
        else:
            raise ValueError("Bad image shape {}".format(img.shape))

    def random_crops(self, i, img, depth=None):
        return _random_crops(img, self._get_size(img), i, 5)

    def five_crops(self, i, img):
        return five_crop(img, self._get_size(img))

    def __init__(self, cfg, dataset_name, img_set, crop_type, crop_ratio, generate_deth: bool = False, depth_type: str = "zoedepth"):
        self.data_dir = cfg.data_dir
        self.crop_ratio = crop_ratio
        self.dataset_name = dataset_name

        if depth_type == "gt" and self.dataset_name == "nyuv2":
            self.save_dir = join(
                cfg.data_dir, "cropped", "{}_{}_crop_{}".format(dataset_name, crop_type, crop_ratio))
        elif depth_type == "zoedepth" and self.dataset_name != "nyuv2":
            self.save_dir = join(
                cfg.data_dir, "cropped", "{}_{}_crop_{}".format(dataset_name, crop_type, crop_ratio))
        elif depth_type == "zoedepth" and self.dataset_name == "nyuv2":
            self.save_dir = join(
                cfg.data_dir, "cropped", "{}_{}_crop_{}_{}".format(dataset_name, crop_type, crop_ratio, depth_type))
        else:
            self.save_dir = join(
                cfg.data_dir, "cropped", "{}_{}_crop_{}_{}".format(dataset_name, crop_type, crop_ratio, depth_type))

        self.img_set = img_set
        self.cfg = cfg

        self.img_dir = join(self.save_dir, "img", img_set)
        self.label_dir = join(self.save_dir, "label", img_set)
        self.depth_dir = join(self.save_dir, "depth", img_set)
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.label_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)

        print(f"depth_dir: {self.depth_dir}")
        print(f"depth_type: {depth_type}")
        print(f"generate_depth: {generate_deth}")
        print(f"save_dir: {self.save_dir}")

        self.generate_depth = generate_deth

        self.zoe = None

        if self.generate_depth:
            # load model
            """conf = get_config("zoedepth", "infer")
            model_zoe_n = build_model(conf)

            # load model on GPU if available
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            self.zoe = model_zoe_n.to(DEVICE)
            self.zoe.eval()"""
            pass

        if crop_type == "random":
            cropper = lambda i, x: self.random_crops(i, x)
            if self.generate_depth:
                raise NotImplementedError("Random crop with depth generation is not implemented yet")
        elif crop_type == "five":
            cropper = lambda i, x: self.five_crops(i, x)
        else:
            raise ValueError('Unknown crop type {}'.format(crop_type))

        print("Crop type: {}".format(crop_type))

        self.depth_type = depth_type

        self.dataset = ContrastiveSegDataset(
            cfg.data_dir,
            dataset_name,
            None,  # This was originally None
            img_set,
            T.ToTensor(),
            ToTargetTensor(),
            cfg=cfg,
            num_neighbors=cfg.num_neighbors,
            pos_labels=False,
            pos_images=False,
            mask=False,
            aug_geometric_transform=None,
            aug_photometric_transform=None,
            extra_transform=cropper,
            return_depth=self.generate_depth,
            depth_type=depth_type,
        )

    def __getitem__(self, item):
        batch = self.dataset[item]

        imgs = batch['img']
        labels = batch['label']

        if self.generate_depth:
            depths = batch['depth']
            #print(len(depths))
            #print(depths[0])
            #print(depths[0].size())

            for crop_num, (img, label, depth) in enumerate(zip(imgs, labels, depths)):
                img_num = item * 5 + crop_num
                img_arr = img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                # Not sure if this works
                depth = depth.float()#.unsqueeze(0)

                #print(f"depths.min(): {depths.min()}, depths.max(): {depths.max()}")

                # print(depths.type())
                #print(depths.size())

                depth_arr = depth.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu',
                                                                                         torch.uint8).numpy().squeeze(
                    -1)

                label_arr = (label + 1).unsqueeze(0).permute(1, 2, 0).to('cpu', torch.uint8).numpy().squeeze(-1)

                img_as_pil = Image.fromarray(img_arr)

                depth_as_pil = Image.fromarray(depth_arr, mode='L')

                if os.path.exists(join(self.img_dir, "{}.jpg".format(img_num))) and os.path.exists(join(self.depth_dir, "{}_zoedepth.png".format(img_num))):
                    warnings.warn(f"Image and depth {img_num} already exists at path {self.img_dir} and {self.depth_dir}")
                    #return True

                img_as_pil.save(join(self.img_dir, "{}.jpg".format(img_num)), 'JPEG')
                # img_as_pil.save(join(self.depth_dir, "{}_zoedepth.png".format(img_num)), 'JPEG')

                Image.fromarray(label_arr).save(join(self.label_dir, "{}.png".format(img_num)), 'PNG')

                depth_as_pil.save(join(self.depth_dir, "{}_{}.png".format(img_num, self.depth_type)), 'PNG')


        else:
            for crop_num, (img, label) in enumerate(zip(imgs, labels)):
                img_num = item * 5 + crop_num
                img_arr = img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                label_arr = (label + 1).unsqueeze(0).permute(1, 2, 0).to('cpu', torch.uint8).numpy().squeeze(-1)
                try:
                    img_as_pil = Image.fromarray(img_arr)
                except:
                    warnings.warn("Could not convert image to PIL format")
                    return True

                if os.path.exists(join(self.img_dir, "{}.jpg".format(img_num))):
                    return True

                img_as_pil.save(join(self.img_dir, "{}.jpg".format(img_num)), 'JPEG')

                Image.fromarray(label_arr).save(join(self.label_dir, "{}.png".format(img_num)), 'PNG')

                """if self.generate_depth:
                    # Generate and save depth values with ZoeDepth
                    depth = self.zoe.infer_pil(img_as_pil, output_type="tensor")
    
                    torch.save(depth.detach().cpu(), join(self.depth_dir, "{}.pt".format(img_num)))
    
                    #print(depth.size())
                    #print("_______________________")
    
                    # sys.exit(0)"""

        return True

    def __len__(self):
        return len(self.dataset)


@hydra.main(config_path="configs", config_name="local_config.yml")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    seed_everything(seed=0) #workers=True

    # dataset_names = ["cityscapes", "cocostuff27", "potsdam"]
    # img_sets = ["train", "val"]
    # crop_types = ["five","random"]
    # crop_ratios = [.5, .7]

    # dataset_names = ["cityscapes"]
    # dataset_names = ["directory"]
    #dataset_names = ["cocostuff27"]
    dataset_names = ["pascalvoc"]
    #img_sets = ["train", "test"] # For NYUv2 use ["train", "test"], otherwise use ["train", "val"]
    img_sets = ["train", "val"]
    crop_types = ["five"]
    crop_ratios = [.5]
    depth_types = ["zoedepth"] # ["gt", "zoedepth", "midas", "kbr"]

    for crop_ratio in crop_ratios:
        for crop_type in crop_types:
            for dataset_name in dataset_names:
                for depth_type in depth_types:
                    for img_set in img_sets:
                        dataset = RandomCropComputer(cfg, dataset_name, img_set, crop_type, crop_ratio,
                                                     generate_deth=cfg.generate_depth, depth_type=depth_type)
                        loader = DataLoader(dataset, 1, shuffle=False, num_workers=0,
                                            collate_fn=lambda l: l)  # cfg.num_workers
                        for _ in tqdm(loader):
                            pass


if __name__ == "__main__":
    prep_args()
    my_app()
