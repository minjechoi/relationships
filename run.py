import argparse
from pl_model import RelationshipClassifier
from pl_dataloader import RelationshipClassificationDataModule
from pytorch_lightning import Trainer
import os
from os.path import join
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import loggers as pl_loggers
import json
import pandas as pd
from typing import Optional, Tuple, Union

def train(args):
    # load logger
    dict_args = vars(args)
    print(os.getpid())
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=args.default_root_dir)

    # load dataset
    dm = RelationshipClassificationDataModule(**dict_args)

    # create callbacks
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_f1",
        mode="max",
        dirpath=args.default_root_dir,
        save_weights_only=True,
        filename="checkpoint-{epoch}-{val_f1:.3f}-{val_acc:.3f}-{val_loss:.3f}",
        every_n_epochs=1,
    )

    early_stop_callback = EarlyStopping(
        monitor='val_f1',
        min_delta=0.0,
        patience=3,
        verbose=False,
        mode='max'
    )

    # load model
    model = RelationshipClassifier(**dict_args)

    trainer = Trainer.from_argparse_args(args, logger=tb_logger,
                callbacks=[checkpoint_callback, early_stop_callback])

    # trainer = Trainer(
    #     max_epochs=hparams.max_epochs,
    #     max_steps=hparams.max_steps,
    #     accelerator=accelerator,
    #     devices=available_gpus,
    #     # profiler='simple',
    #     default_root_dir=hparams.save_path,
    #     log_every_n_steps=hparams.logging_steps,
    #     # limit_train_batches=hparams.limit_train_batches,
    #     limit_val_batches=hparams.limit_val_batches,
    #     # limit_test_batches=hparams.limit_test_batches,
    #     check_val_every_n_epoch=1,
    #     # check_val_every_n_epoch=check_val_every_n_epoch,
    #     # val_check_interval=val_check_interval,
    #     callbacks=[checkpoint_callback,early_stop_callback],
    #     logger=tb_logger,
    #     precision=16,
    #     # strategy='ddp',
    # )
    trainer.fit(model, datamodule=dm)

    if args.do_eval:
        result = trainer.test(model,
                  datamodule=dm, ckpt_path='best')[0]
        with open(join(args.default_root_dir,'results.json'),'w') as f:
            f.write(json.dumps(result))
    return

def eval(args):
    dict_args = vars(args)
    dm = RelationshipClassificationDataModule(**dict_args)
    model = RelationshipClassifier(**dict_args)
    trainer = Trainer.from_argparse_args(args)
    result = trainer.test(model,
                          datamodule=dm, ckpt_path=args.ckpt_path)[0]
    with open(args.save_file, 'w') as f:
        f.write(json.dumps(result))
    return

def predict(args):
    dict_args = vars(args)
    dm = RelationshipClassificationDataModule(**dict_args)
    model = RelationshipClassifier(**dict_args)
    trainer = Trainer.from_argparse_args(args)
    results = trainer.predict(model,
                          datamodule=dm, ckpt_path=args.ckpt_path)
    y_score, y_pred = [], []
    for res in results:
        y_score.extend(res['scores'])
        y_pred.extend(res['labels'])
    relationships = ['social','romance','family','organizational','parasocial']
    y_pred = [relationships[i] for i in y_pred]
    results = list(zip(y_score,y_pred))
    df=pd.DataFrame(results,columns=['score','prediction'])
    df.to_csv(join(args.save_file),sep='\t',index=False)
    return

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = RelationshipClassifier.add_model_specific_args(parser)
    parser = RelationshipClassificationDataModule.add_model_specific_args(parser)
    parser.add_argument("--setting", type=str, default=None)
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--do_test", action='store_true')
    parser.add_argument("--do_predict", action='store_true')
    parser.add_argument("--ckpt_path", type=str,
        default='/shared/0/projects/relationships/working-dir/relationship-prediction-new/results/without_network_3plus/checkpoint-epoch=2-val_f1=0.703-val_acc=0.717-val_loss=0.854.ckpt')
    parser.add_argument("--save_file",type=str,default=None)
    args = parser.parse_args()
    if args.do_train:
        train(args)
    if args.do_test:
        eval(args)
    if args.do_predict:
        predict(args)

    """
    Examples on how to apply different training / test / prediction settings
    """
    if args.setting=='all':
        args.default_root_dir='/shared/0/projects/relationships/working-dir/relationship-prediction-new/results/all_features_3plus'
        args.train_data_file='/shared/0/projects/relationships/working-dir/relationship-prediction-new/data/processed-data/train_3plus.json.gz'
        args.test_data_file='/shared/0/projects/relationships/working-dir/relationship-prediction-new/data/processed-data/test_3plus.json.gz'
        args.val_data_file='/shared/0/projects/relationships/working-dir/relationship-prediction-new/data/processed-data/val_3plus.json.gz'
        args.max_epochs=10
        args.balance_training_set=True
        args.use_public_mention = True
        args.use_direct_mention = True
        args.use_retweet = True
        args.use_bio = True
        args.use_activity = True
        args.use_count = True
        args.use_network = True
        args.precision = 16
        args.accelerator = 'gpu'
        train(args)

    if args.setting == 'without_network':
        args.default_root_dir = '/shared/0/projects/relationships/working-dir/relationship-prediction-new/results/without_network_3plus'
        args.train_data_file = '/shared/0/projects/relationships/working-dir/relationship-prediction-new/data/processed-data/train_3plus.json.gz'
        args.test_data_file = '/shared/0/projects/relationships/working-dir/relationship-prediction-new/data/processed-data/test_3plus.json.gz'
        args.val_data_file = '/shared/0/projects/relationships/working-dir/relationship-prediction-new/data/processed-data/val_3plus.json.gz'
        args.max_epochs = 10
        args.balance_training_set = True
        args.use_public_mention = True
        args.use_direct_mention = True
        args.use_retweet = True
        args.use_bio = True
        args.use_activity = True
        args.use_count = True
        args.use_network = False
        args.precision = 16
        args.accelerator = 'gpu'
        train(args)

        # takes 20 mins just to go through the validation set once
        # 1.25iter per second -> 1hr leads to 4500 iters
        # 16hrs to go through one epoch -> maybe better to do validations like every 10 hrs?

    if args.setting=='test':
        args.save_file='/shared/0/projects/relationships/working-dir/relationship-prediction-new/results/without_network_3plus/results.json'
        args.train_data_file='/shared/0/projects/relationships/working-dir/relationship-prediction-new/data/processed-data/test_3plus.json.gz'
        args.test_data_file='/shared/0/projects/relationships/working-dir/relationship-prediction-new/data/processed-data/test_3plus.json.gz'
        args.val_data_file='/shared/0/projects/relationships/working-dir/relationship-prediction-new/data/processed-data/val_3plus.json.gz'
        args.ckpt_path='/shared/0/projects/relationships/working-dir/relationship-prediction-new/results/without_network_3plus/checkpoint-epoch=2-val_f1=0.703-val_acc=0.717-val_loss=0.854.ckpt'
        args.use_public_mention = True
        args.use_direct_mention = True
        args.use_retweet = True
        args.use_bio = True
        args.use_activity = True
        args.use_count = True
        args.use_network = False
        args.precision = 16
        args.accelerator = 'gpu'
        args.test_batch_size = 64
        eval(args)

    if args.setting=='predict':
        args.predict_data_file='/shared/0/projects/relationships/working-dir/relationship-prediction-new/data/processed-data/test_3plus.json.gz' # change this to your custom file
        args.save_file='/shared/0/projects/relationships/working-dir/relationship-prediction-new/results/without_network_3plus/predictions.df.tsv' # file for storing the predictions
        args.ckpt_path='/shared/0/projects/relationships/working-dir/relationship-prediction-new/results/without_network_3plus/checkpoint-epoch=2-val_f1=0.703-val_acc=0.717-val_loss=0.854.ckpt' # best to set this fixed unless you're using another model
        args.use_public_mention = True
        args.use_direct_mention = True
        args.use_retweet = True
        args.use_bio = True
        args.use_activity = True
        args.use_count = True
        args.use_network = False
        args.precision = 16
        args.accelerator = 'gpu'
        args.predict_batch_size = 16
        # args.limit_predict_batches = 4 # for debugging purposes
        predict(args)

"""
CUDA_VISIBLE_DEVICES=1 python run.py --setting all --accelerator gpu
CUDA_VISIBLE_DEVICES=2 python run.py --setting without_network --accelerator gpu

CUDA_VISIBLE_DEVICES=2 python run.py --setting test --accelerator gpu
CUDA_VISIBLE_DEVICES=0 python run.py --setting predict --accelerator gpu

CUDA_VISIBLE_DEVICES=0 python run.py \
    --do_predict --predict_batch_size=16 --accelerator gpu \
    --use_public_mention --use_direct_mention --use_retweet \
    --use_bio --use_activity --use_count \
    --predict_data_file /shared/0/projects/relationships/working-dir/relationship-prediction-new/data/processed-data/test_3plus.json.gz \
    --save_file=/shared/0/projects/relationships/working-dir/relationship-prediction-new/results/without_network_3plus/predictions.df.tsv \
    --ckpt_path=/shared/0/projects/relationships/working-dir/relationship-prediction-new/results/without_network_3plus/checkpoint-epoch=2-val_f1=0.703-val_acc=0.717-val_loss=0.854.ckpt
"""