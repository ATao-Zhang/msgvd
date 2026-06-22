from argparse import ArgumentParser
from typing import cast

from commode_utils.common import print_config
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything

from src.datas.datamodules import XFGDataModule
from src.models.ep_loc_module import EPLocModule
from src.models.vd import DeepWuKong
from src.train import train
from src.utils import filter_warnings


def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--backbone-checkpoint",
                            required=True,
                            type=str,
                            help="Trained MSAVD/DeepWuKong checkpoint path")
    arg_parser.add_argument("-c",
                            "--config",
                            default="configs/dwk.yaml",
                            type=str,
                            help="Path to YAML configuration file")
    arg_parser.add_argument("--data-folder",
                            default=None,
                            type=str,
                            help="Override config.data_folder, e.g. a SARD-Loc root")
    arg_parser.add_argument("--batch-size",
                            default=None,
                            type=int,
                            help="Override training batch size")
    arg_parser.add_argument("--unfreeze-backbone",
                            action="store_true",
                            help="Jointly fine-tune the MSAVD backbone with EP-LocNet")
    return arg_parser


def train_ep_locnet(backbone_checkpoint: str,
                    config_path: str,
                    data_folder: str = None,
                    batch_size: int = None,
                    unfreeze_backbone: bool = False):
    filter_warnings()
    config = cast(DictConfig, OmegaConf.load(config_path))
    if data_folder is not None:
        config.data_folder = data_folder
    if batch_size is not None:
        config.hyper_parameters.batch_size = batch_size
        config.hyper_parameters.test_batch_size = batch_size
    print_config(config, ["gnn", "classifier", "localization", "hyper_parameters"])
    seed_everything(config.seed, workers=True)

    backbone = DeepWuKong.load_from_checkpoint(backbone_checkpoint)
    vocab = backbone.hparams["vocab"]
    data_module = XFGDataModule(config, vocab)
    model = EPLocModule(config=config,
                        vocab=vocab,
                        backbone_checkpoint=backbone_checkpoint,
                        backbone=backbone,
                        freeze_backbone=not unfreeze_backbone,
                        top_k=config.localization.top_k)
    train(model, data_module, config)


if __name__ == "__main__":
    __arg_parser = configure_arg_parser()
    __args = __arg_parser.parse_args()
    train_ep_locnet(__args.backbone_checkpoint,
                    __args.config,
                    __args.data_folder,
                    __args.batch_size,
                    __args.unfreeze_backbone)
