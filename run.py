import argparse
import os
import torch
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import ConcatenatedClassifier
from data_loader import CustomDataset
from transformers import (
    AdamW,set_seed
)
import gc
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, precision_score, recall_score


def arg_parser():
    parser = argparse.ArgumentParser()

    # directory hyperparameters
    parser.add_argument(
        "--train_data_file", default=None, type=str, help="The file containing training data"
    )
    parser.add_argument(
        "--val_data_file", default=None, type=str, help="The file containing validation data"
    )
    parser.add_argument(
        "--test_data_file", default=None, type=str, help="The file containing test data"
    )
    parser.add_argument(
        "--infer_data_file", default=None, type=str, help="The file containing infer data"
    )
    parser.add_argument(
        "--output_dir", type=str,required=True,help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--classifier_config_dir", default='data/', type=str, help="File directory containing config.json for the classifier"
    )
    parser.add_argument(
        "--model_dir", default=None, type=str,
        help="Path to the actual pickle for for the trained classifier model"
    )

    # script-related hyperparameters
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_test", action="store_true", help="Whether to run evaluation on the test set.")
    parser.add_argument("--do_infer", action="store_true", help="Whether to run inference on an unknown set of dyads")

    # model & data hyperparameters
    parser.add_argument("--use_pm", action="store_true", help="whether to consider public mentions in training")
    parser.add_argument("--use_dm", action="store_true", help="whether to consider directed mentions in training")
    parser.add_argument("--use_rt", action="store_true", help="whether to consider retweets in training")
    parser.add_argument("--use_bio", action="store_true", help="whether to consider bio information of the two users in training (loads an additional model for processing description text)")
    parser.add_argument("--use_name", action="store_true", help="whether to consider the username information of two users in training (loads an additional model for character embeddings)")
    parser.add_argument("--use_count", action="store_true", help="whether to consider the ratio of tweets/retweets")
    parser.add_argument("--use_network", action="store_true", help="whether to consider network features")

    parser.add_argument("--min_samples", default=0, type=int, help='Minimum number of samples to be considered into the training set')
    parser.add_argument("--max_samples", default=20, type=int, help='Maximum number of samples to be considered into the training set, sample if exceeds')
    parser.add_argument("--n_clf_layers", default=6, type=int, help='Custom number of layers to be used in the classifier')

    parser.add_argument("--checkpoint_interval", default=10000, type=int,help="Interval for saving checkpoints")
    parser.add_argument("--valid_interval", default=10000, type=int,help="Interval for running validation")

    parser.add_argument("--batch_size", default=4, type=int,help="Batch size")
    parser.add_argument("--lr", default=1e-5, type=float,help="Learning rate for the model")
    parser.add_argument("--n_epochs", default=5, type=int,help="Number of epochs to run")

    parser.add_argument("--eps", default=1e-8, type=float,help="Hyperparameter for optimizer in training phase")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,help="Hyperparameter for optimizer in training phase")
    parser.add_argument("--num_warmup_steps", default=1000, help="Hyperparameter for optimizer in training phase")

    # other hyperparameters
    parser.add_argument("--start_from", default=None, type=int, help="Checkpoint to start from")
    parser.add_argument("--gpu_id", default=None, type=int,help="GPU id to use (optional)")
    parser.add_argument("--seed", default=42, type=int,help="Seed to use for reproducibility")

    args = parser.parse_args()

    return args

def main():
    print('pid: ',os.getpid())

    args = arg_parser()
    set_seed(args.seed)

    if args.do_train:
        train_bert(args)
    if args.do_infer:
        infer_bert(args)
    return

def train_bert(args):

    assert os.path.exists(str(args.train_data_file)),"The argument --train_data_file should be a valid file"
    assert os.path.exists(str(args.val_data_file)),"The argument --val_train_data_file should be a valid file"
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # load model
    cc = ConcatenatedClassifier(classifier_config_dir=args.classifier_config_dir, device=device, task_type='train', n_clf_layers=args.n_clf_layers,
                                use_dm=args.use_dm, use_pm=args.use_pm, use_rt=args.use_rt, use_bio=args.use_bio, use_name=args.use_name,
                                use_network=args.use_network, use_count=args.use_count)
    cc.to(device)

    # load data
    train_data = []
    with open(args.train_data_file) as f:
        for line in f:
            train_data.append(json.loads(line))
    val_data = []
    with open(args.val_data_file) as f:
        for line in f:
            val_data.append(json.loads(line))

    train_dataset = CustomDataset(data_list=train_data, task_type='train',
             use_pm=args.use_pm, use_dm=args.use_dm, use_rt=args.use_rt, use_bio=args.use_bio,
             use_network=args.use_network, use_count=args.use_count, use_name=args.use_name,
             max_samples=args.max_samples)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=train_dataset.collate_fn)
    val_dataset = CustomDataset(data_list=val_data, task_type='test',
             use_pm=args.use_pm, use_dm=args.use_dm, use_rt=args.use_rt, use_bio=args.use_bio,
             use_network=args.use_network, use_count=args.use_count, use_name=args.use_name,
             max_samples=args.max_samples)
    val_data_loader = DataLoader(val_dataset, batch_size=args.batch_size*4, shuffle=False, num_workers=4, collate_fn=train_dataset.collate_fn)

    print("Starting training")
    save_dir = args.output_dir # create save directory to store models and training loss
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # hyperparameters
    eps = args.eps
    max_grad_norm = args.max_grad_norm
    num_training_steps = len(train_data_loader) * args.n_epochs
    num_warmup_steps = args.num_warmup_steps

    optimizer = AdamW(cc.parameters(), lr=args.lr, eps=eps)  # To reproduce BertAdam specific behavior set correct_bias=False
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                    num_training_steps=num_training_steps)  # PyTorch scheduler

    # training progress
    total_steps = 0

    for epoch in range(args.n_epochs):
        # Reset the total loss for this epoch.
        total_loss = 0
        # For each batch of training data...
        pbar = tqdm(train_data_loader)
        for step,batch in enumerate(pbar):
            # validation step
            if total_steps % args.valid_interval == 0 and not total_steps == 0:
                gc.collect()
                torch.cuda.empty_cache()

                y_true, y_pred = [],[]
                with torch.no_grad():
                    cc.eval()
                    valid_loss = 0
                    total_samples = 0
                    for batch in val_data_loader:
                        y_true.extend(batch['text'][0].tolist())
                        loss,logits = cc(batch)
                        y_pred.extend(logits.argmax(1).tolist())
                        valid_loss+=loss.item()*len(logits)
                        total_samples+=len(logits)
                    valid_loss/=total_samples
                # save valid loss
                with open(os.path.join(save_dir, 'valid-loss.txt'), 'a') as f:
                    f.write('\t'.join([str(x) for x in [total_steps, round(valid_loss, 4)]]) + '\n')
                # compute metrics and save them separately
                for name,fun in [('f1',f1_score),('precision',precision_score),('recall',recall_score)]:
                    with open(os.path.join(save_dir,'valid-%s.txt'%name),'a') as f:
                        f.write('Step %d\n'%total_steps)
                        macro = fun(y_true=y_true,y_pred=y_pred,average='macro')
                        classwise = fun(y_true=y_true,y_pred=y_pred,average=None)
                        f.write('%1.3f\n'%macro)
                        f.write('\t'.join([str(round(x,3)) for x in classwise])+'\n')

                # clear any memory and gpu
                gc.collect()
                torch.cuda.empty_cache()

                cc.train()

            if total_steps % args.checkpoint_interval == 0 and not total_steps == 0:
                # save models
                torch.save(obj=cc.state_dict(), f=os.path.join(save_dir, '%d-steps-cc.pth' % (total_steps)))
                # collect garbage
                torch.cuda.empty_cache()
                gc.collect()

            optimizer.zero_grad()
            # compute for one step
            loss,logits = cc(batch)
            # gradient descent and update
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cc.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            pbar.set_description("[%d] Loss at %d/%dth batch: %1.3f" %(int(os.getpid()),step+1,len(train_data_loader),loss.item()))
            with open(os.path.join(save_dir,'training-loss.txt'),'a') as f:
                f.write('\t'.join([str(x) for x in [step,round(loss.item(),4)]])+'\n')

            total_steps += 1

        print("Epoch %d complete! %d steps"%(epoch,total_steps))

    return



def infer_bert(args):
    import pandas as pd
    assert os.path.exists(str(args.infer_data_file)),"The argument --infer_data_file should be a valid file"
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # load model
    cc = ConcatenatedClassifier(classifier_config_dir=args.classifier_config_dir, device=device, task_type='infer', n_clf_layers=args.n_clf_layers,
                                use_dm=args.use_dm, use_pm=args.use_pm, use_rt=args.use_rt, use_bio=args.use_bio, use_name=args.use_name,
                                use_network=args.use_network, use_count=args.use_count)
    cc.load_state_dict(torch.load(args.model_dir))
    cc.to(device)

    # load data
    data = []
    with open(args.infer_data_file) as f:
        for line in f:
            data.append(json.loads(line))

    dataset = CustomDataset(data_list=data, task_type='infer',
             use_pm=args.use_pm, use_dm=args.use_dm, use_rt=args.use_rt, use_bio=args.use_bio,
             use_network=args.use_network, use_count=args.use_count, use_name=args.use_name,
             max_samples=args.max_samples)
    data_loader = DataLoader(dataset, batch_size=args.batch_size*4, shuffle=False, num_workers=4, collate_fn=dataset.collate_fn)

    print("Starting evaluation")
    save_dir = args.output_dir # create save directory to store output files
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    pbar = tqdm(data_loader)
    y_true, y_pred = [], []
    with torch.no_grad():
        cc.eval()
        for batch in pbar:
            logits = cc(batch)
            y_pred.extend(logits.argmax(1).tolist())

    # save
    data_name = args.infer_data_file.split('/')[-1]
    out = []
    for (aid,bid),pred in zip(dataset.dyad_ids,y_pred):
        out.append((aid,bid,['social','romance','family','organizational','parasocial'][pred]))
    df = pd.DataFrame(out, columns=['user_id_a', 'user_id_b','predicted-relationship'])
    save_file = os.path.join(args.output_dir, '%s-predictions.tsv' % data_name)
    df.to_csv(save_file,sep='\t',index=False)
    print("Saved inferred outputs to %s"%save_file)
    return



if __name__=='__main__':
    main()

"""
CUDA_VISIBLE_DEVICES=0 python run.py \
 --do_infer --use_dm --use_pm --use_rt --use_name --use_bio --use_network --use_count \
 --infer_data_file=data/sample_outputs.json \
 --output_dir=data \
 --model_dir=data/full-model.pth \
 --checkpoint_interval=50000 --valid_interval=50000 --batch_size=2 \
 --classifier_config_dir=data/bert-config.json
"""

"""
CUDA_VISIBLE_DEVICES=0 python run.py \
 --do_train --use_dm --use_pm --use_rt --use_name --use_bio --use_network --use_count \
 --train_data_file=data/train-data-camera-ready.json \
 --val_data_file=data/train-data-camera-ready.json \
 --output_dir=data \
 --checkpoint_interval=50000 --valid_interval=50000 --batch_size=2 \
 --classifier_config_dir=data/bert-config.json
"""