from argparse import ArgumentParser
import json
from typing import Optional

import torch
from pytorch_lightning import Trainer, seed_everything

from src.datas.datamodules import XFGDataModule
from src.models.ep_loc_module import EPLocModule
from src.utils import filter_warnings


def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("checkpoint", type=str,
                            help="Trained EP-LocNet checkpoint path")
    arg_parser.add_argument("--backbone-checkpoint",
                            default=None,
                            type=str,
                            help="Override the MSAVD backbone checkpoint stored in the EP-LocNet checkpoint")
    arg_parser.add_argument("--data-folder",
                            default=None,
                            type=str,
                            help="Override config.data_folder")
    arg_parser.add_argument("--batch-size",
                            default=None,
                            type=int,
                            help="Override test batch size")
    arg_parser.add_argument("--top-k",
                            default=None,
                            type=int,
                            help="Override the number of ranked source lines to output")
    arg_parser.add_argument("--predictions-output",
                            default=None,
                            type=str,
                            help="Optional JSONL file for Top-k line predictions")
    return arg_parser


def load_model(checkpoint_path: str,
               backbone_checkpoint: Optional[str] = None,
               top_k: Optional[int] = None) -> EPLocModule:
    load_kwargs = {}
    if backbone_checkpoint is not None:
        load_kwargs["backbone_checkpoint"] = backbone_checkpoint
    model = EPLocModule.load_from_checkpoint(checkpoint_path, **load_kwargs)
    if top_k is not None:
        model.top_k = top_k
    return model


def write_predictions(model: EPLocModule,
                      data_module: XFGDataModule,
                      output_path: str):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    with open(output_path, "w") as output_file:
        sample_offset = 0
        for batch in data_module.test_dataloader():
            batch.move_to_device(device)
            with torch.no_grad():
                output = model(batch.graphs)
                topk_results = model.topk(output)
                detection_probs = torch.softmax(output.detection_logits,
                                                dim=-1)[:, 1]
            for local_idx, ranked_lines in enumerate(topk_results):
                record = {
                    "sample_index": sample_offset + local_idx,
                    "vulnerability_probability": float(detection_probs[local_idx].item()),
                    "topk_lines": [
                        {"line": line, "score": score}
                        for line, score in ranked_lines
                    ],
                }
                output_file.write(json.dumps(record) + "\n")
            sample_offset += len(topk_results)


def test(checkpoint_path: str,
         backbone_checkpoint: str = None,
         data_folder: str = None,
         batch_size: int = None,
         top_k: int = None,
         predictions_output: str = None):
    filter_warnings()
    model = load_model(checkpoint_path, backbone_checkpoint, top_k)
    config = model.config
    vocabulary = model.vocab
    if data_folder is not None:
        config.data_folder = data_folder
    if batch_size is not None:
        config.hyper_parameters.test_batch_size = batch_size
        config.hyper_parameters.batch_size = batch_size
    data_module = XFGDataModule(config, vocabulary)
    seed_everything(config.seed)
    gpu = 1 if torch.cuda.is_available() else None
    trainer = Trainer(gpus=gpu)
    trainer.test(model, datamodule=data_module)
    if predictions_output is not None:
        write_predictions(model, data_module, predictions_output)


if __name__ == "__main__":
    __arg_parser = configure_arg_parser()
    __args = __arg_parser.parse_args()
    test(__args.checkpoint,
         __args.backbone_checkpoint,
         __args.data_folder,
         __args.batch_size,
         __args.top_k,
         __args.predictions_output)
