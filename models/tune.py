import os
import json
import logging
import argparse

from modules.models.loss import E2ELoss
# from modules.models.model import OCRModel
from modules.models import model
from modules.logger.logger import Logger
from modules.data.loader import ICDARDataLoader
from modules.trainer.trainer import Trainer
from modules.models.metric import icdar_metric
import wandb
import optuna

logging.basicConfig(level=logging.DEBUG, format='')

def search_param(trial, config):
    if 'args' in config['optuna']:
        if 'epochs' in config['optuna']['args']:
            epoch = trial.suggest_categorical("epoch", config['optuna']['args']['epochs'])
            config['trainer']['epochs'] = epoch
        if 'optimizers' in config['optuna']['args']:
            optim = trial.suggest_categorical("optimizer", config['optuna']['args']['optimizers'])
            config['optimizer_type'] = optim['optimizer_type']
            config['optimizer'] = optim['args']
    return config

# def main(config, resume):
def objective(trial, config, resume):
    train_logger = Logger()
    
    # load data
    train_dataloader = ICDARDataLoader(config).train()
    val_dataloader = ICDARDataLoader(config).val() if config['validation']['validation_split'] > 0 else None

    # IF use optuna - get config
    if 'optuna' in config:
        config = search_param(trial, config)
        print(config)
    
    # initial model
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in config['gpus']])
    ocr_model = getattr(model, config['model'])(config)
    
    wandb_log = None
    dir_name = f"{config['optuna']['study_name']}_#{trial.number:03}_{config['trainer']['epochs']}"
    if 'wandb' in config:
        wandb_log = wandb.init(
            project=config['wandb']['project'],
            config=config
        )
        wandb.run.name = dir_name
        wandb.run.save()

    loss = E2ELoss()
    trainer = Trainer(ocr_model, loss, icdar_metric, resume, config, dir_name, train_dataloader, val_dataloader, train_logger, wandb_logger=wandb_log != None)
    log = trainer.train()
    print(log)
    if wandb_log:
        wandb.finish()
    
    if "val_hmean" in log:
        return log["val_hmean"], log["val_precious"], log["val_recall"]
    return log["hmean"], log["precious"], log["recall"]


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='MLT-OCR')
    parser.add_argument('-c', '--config', default='./optuna_config/config.json', type=str, help='path to config file')
    parser.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    parser.add_argument("--storage", default="postgresql://postgres:0000@localhost:6006/postgres", type=str, help="Optuna database storage path.")
    args = parser.parse_args()
    config = json.load(open(args.config))
    if args.resume:
        logger.warning('Warning: --config overridden by --resume')

    # main(config, args.resume)
    rdb_storage = optuna.storages.RDBStorage(url=args.storage)
    study = optuna.create_study(
        directions=["maximize", "maximize", "maximize"],
        study_name=config['optuna']['study_name'],
        # sampler=sampler,
        storage=rdb_storage,
        load_if_exists=True,
    )
    study.optimize(lambda trial: objective(trial, config, resume=args.resume), n_trials=config['optuna']['trial'])