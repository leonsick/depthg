from data import ContrastiveSegDataset
from modules import *
import os
from os.path import join
import hydra
import numpy as np
import torch.multiprocessing
import torch.multiprocessing
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from tqdm import tqdm


def get_feats(model, loader):
    all_feats = []
    for pack in tqdm(loader):
        img = pack["img"]
        feats = F.normalize(model.forward(img.to(("cuda" if torch.cuda.is_available() else "cpu"))).mean([2, 3]), dim=1)
        all_feats.append(feats.to("cpu", non_blocking=True))
    return torch.cat(all_feats, dim=0).contiguous()


@hydra.main(config_path="configs", config_name="local_config.yml")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    #data_dir = cfg.data_dir
    dataset_dir = cfg.data_dir
    log_dir = join(cfg.output_root, "logs")
    #data_dir = join(cfg.output_root, "data")
    data_dir = cfg.data_dir
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(join(data_dir, "nns"), exist_ok=True)

    seed_everything(seed=0)

    print(data_dir)
    print(cfg.data_dir)

    image_sets = ["train", "val"]
    dataset_names = ["cocostuff27", "cityscapes", "potsdam"]
    crop_types = ["five", None]

    # Uncomment these lines to run on custom datasets
    #dataset_names = ["directory"]
    #crop_types = [None]

    #res = 224 # for dinov2 models, increase to 392 for small and
    res = 392
    n_batches = 64 #16

    device = ("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.arch == "dino":
        from modules import DinoFeaturizer, LambdaLayer
        no_ap_model = torch.nn.Sequential(
            DinoFeaturizer(20, cfg),  # dim doesent matter
            LambdaLayer(lambda p: p[0]),
        ).to(device)
    else:
        cut_model = load_model(cfg.model_type, join(cfg.output_root, "data")).to(device)
        no_ap_model = nn.Sequential(*list(cut_model.children())[:-1]).to(device)
    par_model = torch.nn.DataParallel(no_ap_model)

    for crop_type in crop_types:
        for image_set in image_sets:
            for dataset_name in dataset_names:
                nice_dataset_name = cfg.dir_dataset_name if dataset_name == "directory" else dataset_name

                feature_cache_file = join(data_dir, "nns", "nns_{}_{}_{}_{}_{}.npz".format(
                    cfg.model_type, nice_dataset_name, image_set, crop_type, res))
                print(f"Searched for {feature_cache_file}")

                if os.path.exists(feature_cache_file):
                    print("Found {}".format(feature_cache_file))
                else:
                    print("{} not found, computing".format(feature_cache_file))
                    # data_dir=data_dir
                    dataset = ContrastiveSegDataset(
                        data_dir=dataset_dir,
                        dataset_name=dataset_name,
                        crop_type=crop_type,
                        image_set=image_set,
                        transform=get_transform(res, False, "center"),
                        target_transform=get_transform(res, True, "center"),
                        cfg=cfg,
                    )

                    batch_size = 128 if cfg.model_type == "vit_small" else 64

                    loader = DataLoader(dataset, batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=False)

                    with torch.no_grad():
                        # Get features of len(dataset) x feature_dim
                        normed_feats = get_feats(par_model, loader)
                        all_nns = []
                        # Define step size
                        step = normed_feats.shape[0] // n_batches
                        print(normed_feats.shape)
                        for i in tqdm(range(0, normed_feats.shape[0], step)):
                            torch.cuda.empty_cache() if torch.cuda.is_available() else None
                            # From all feats, get only a slice of size step
                            batch_feats = normed_feats[i:i + step, :]
                            # Compute pairwise similarities between batch_feats and all feats
                            pairwise_sims = torch.einsum("nf,mf->nm", batch_feats, normed_feats)
                            # Get top 30 nearest neighbors and append to all_nns list
                            all_nns.append(torch.topk(pairwise_sims, 30)[1].cpu())
                            del pairwise_sims
                        # Once donce, concatenate all_nns into one tensor
                        # This tensor stores the indices of the nearest neighbors for each image on an image level,
                        # not on sampled patch level
                        nearest_neighbors = torch.cat(all_nns, dim=0)

                        np.savez_compressed(feature_cache_file, nns=nearest_neighbors.numpy())
                        print("Saved NNs", cfg.model_type, nice_dataset_name, image_set)


if __name__ == "__main__":
    prep_args()
    my_app()
