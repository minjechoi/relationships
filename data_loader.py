from pytorch_lightning import LightningDataModule
import os
from os.path import join
import gzip
import json
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import resample
from transformers import AutoTokenizer
import pandas as pd
import numpy as np

class DefaultDataset(Dataset):
    def __init__(self, data):
        self.data = data
        return

    def __getitem__(self, idx):
        obj = self.data[idx]
        return obj

    def __len__(self):
        return len(self.data)

class RelationshipClassificationDataModule(LightningDataModule):
    def __init__(self,
        # paths to directory
        # train_data_file='/shared/0/projects/relationships/working-dir/relationship-prediction-new/data/processed-data/train_3plus.json.gz',
        # test_data_file='/shared/0/projects/relationships/working-dir/relationship-prediction-new/data/processed-data/test_3plus.json.gz',
        # val_data_file='/shared/0/projects/relationships/working-dir/relationship-prediction-new/data/processed-data/val_3plus.json.gz',
        # predict_data_file='/shared/0/projects/relationships/working-dir/relationship-prediction-new/data/processed-data/val_3plus.json.gz',
        train_data_file,
        test_data_file,
        val_data_file,
        predict_data_file,
        # paths to components of model
        # language_model_name_or_path='cardiffnlp/twitter-roberta-base-sentiment',
        # language_model_cache_dir='/shared/0/projects/relationships/.cache/huggingface/transformers',
        # feature_bin_file_dir='/shared/0/projects/relationships/working-dir/relationship-prediction-new/data/features/feature_bins.csv',
        language_model_name_or_path,
        language_model_cache_dir,
        feature_bin_file_dir,

        # feature settings
        use_public_mention,
        use_direct_mention,
        use_retweet,
        use_bio,
        use_activity,
        use_count,
        use_network,

        # settings related to dataset
        balance_training_set,
        train_batch_size,
        eval_batch_size,
        num_workers, # for temporary debugging purposes
        max_length,
        **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = AutoTokenizer.from_pretrained(language_model_name_or_path,
                               cache_dir=language_model_cache_dir)
        self.category2idx={cat:i for i,cat in enumerate(['social','romance',
                                 'family','organizational','parasocial'])}

        # position embedding index for each feature type
        self.feature2pos_idx={}
        cnt = 0
        for user in ['a_data','b_data']:
            for typ in ['public-mention','direct-mention','retweet']:
                self.feature2pos_idx[(user,typ)] = [cnt]
                cnt += 1
            self.feature2pos_idx[(user,'bio')] = [cnt]
            cnt+=1
            self.feature2pos_idx[(user,'activity')] = [x for x in range(cnt,cnt+3)]
            cnt+=3
            self.feature2pos_idx[(user, 'count')] = [x for x in range(cnt,cnt+6)]
            cnt+=6
        self.feature2pos_idx['network'] = [cnt,cnt+1]

        # for idx,col in enumerate(['friends_count','followers_count', 'statuses_count',
        #     'num_pm', 'num_dm', 'num_rt', 'frac_pm', 'frac_dm', 'frac_rt',
        #     'net_jacc','net_aa']):

        # load feature bins
        # binned feature embedding index for numerical features
        self.feature2bins={}
        self.feature2emb_idx = {}
        with open(feature_bin_file_dir) as f:
            for i,line in enumerate(f):
                line=line.strip().split(',')
                col = line[0]
                bins=[float(x) for x in line[2:]]
                bins[0]=-np.inf
                bins.append(np.inf)
                self.feature2bins[col]=bins
                self.feature2emb_idx[col]=i

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("RelationshipClassificationDataModule")
        parser.add_argument("--train_data_file", type=str, default=None)
        parser.add_argument("--test_data_file", type=str, default=None)
        parser.add_argument("--val_data_file", type=str, default=None)
        parser.add_argument("--predict_data_file", type=str, default=None)
        # components of model
        parser.add_argument("--language_model_name_or_path", type=str, default='cardiffnlp/twitter-roberta-base-sentiment')
        parser.add_argument("--language_model_cache_dir", type=str, default='/shared/0/projects/relationships/.cache/huggingface/transformers')
        parser.add_argument("--feature_bin_file_dir", type=str, default='/shared/0/projects/relationships/working-dir/relationship-prediction-new/data/features/feature_bins.csv')
        # feature settings
        parser.add_argument("--use_public_mention", action='store_true')
        parser.add_argument("--use_direct_mention", action='store_true')
        parser.add_argument("--use_retweet", action='store_true')
        parser.add_argument("--use_bio", action='store_true')
        parser.add_argument("--use_name", action='store_true')
        parser.add_argument("--use_network", action='store_true')
        # settings related to dataset size
        parser.add_argument("--balance_training_set", action='store_true')
        parser.add_argument("--train_batch_size", type=int, default=4)
        parser.add_argument("--eval_batch_size", type=int, default=8)
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--max_length", type=int, default=512)
        return parent_parser


    def setup(self, stage=None):
        self.datasets = {}
        if stage in ['fit', None]:
            self.train_data = []
            with gzip.open(self.hparams.train_data_file, 'rb') as f:
                for line in f:
                    obj = json.loads(line.decode('utf-8'))
                    self.train_data.append(obj)
            if self.hparams.balance_training_set:
                self.upsample_training_set()
            self.datasets['train'] = DefaultDataset(self.train_data)

            self.val_data = []
            with gzip.open(self.hparams.val_data_file, 'rb') as f:
                for line in f:
                    obj = json.loads(line.decode('utf-8'))
                    self.val_data.append(obj)
            self.datasets['val'] = DefaultDataset(self.val_data)

        if stage in ['test',None]:
            self.test_data = []
            with gzip.open(self.hparams.test_data_file, 'rb') as f:
                for line in f:
                    obj = json.loads(line.decode('utf-8'))
                    self.test_data.append(obj)
            self.datasets['test'] = DefaultDataset(self.test_data)

        if stage in ['predict']:
            self.predict_data = []
            with gzip.open(self.hparams.predict_data_file, 'rb') as f:
                for line in f:
                    obj = json.loads(line.decode('utf-8'))
                    self.predict_data.append(obj)
            self.datasets['predict'] = DefaultDataset(self.predict_data)

    def upsample_training_set(self):
        # upsamples minority classes to match the majority class
        cat2samples = {}
        for obj in self.train_data:
            label = obj['category']
            if label not in cat2samples:
                cat2samples[label]=[]
            cat2samples[label].append(obj)
        # get counts
        max_ln = max([len(V) for cat,V in cat2samples.items()])
        self.train_data = []
        for label,V in cat2samples.items():
            if len(V)==max_ln:
                self.train_data.extend(V)
            else:
                self.train_data.extend(resample(V,replace=True,n_samples=max_ln))
        return

    def train_dataloader(self):
        return DataLoader(self.datasets['train'], shuffle=True,
              num_workers=self.hparams.num_workers,
              collate_fn=self.collate_fn,
              batch_size=self.hparams.train_batch_size,
              pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.datasets['val'], shuffle=False,
              num_workers=self.hparams.num_workers,
              collate_fn=self.collate_fn,
              batch_size=self.hparams.eval_batch_size,
              pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.datasets['test'], shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn,
            batch_size=self.hparams.eval_batch_size,
            pin_memory = True)

    def predict_dataloader(self):
        return DataLoader(self.datasets['predict'], shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn,
            batch_size=self.hparams.predict_batch_size,
            pin_memory = True)

    def collate_fn(self, batch):
        output = {}

        # encode various texts
        if self.hparams.use_public_mention | \
            self.hparams.use_direct_mention | \
            self.hparams.use_retweet | \
            self.hparams.use_bio:
            sentences = []
            sentence_pos_idx = []

            tweet_types = []
            if self.hparams.use_public_mention:
                tweet_types.append('public-mention')
            if self.hparams.use_direct_mention:
                tweet_types.append('direct-mention')
            if self.hparams.use_retweet:
                tweet_types.append('retweet')

            for dyad_idx, dyad in enumerate(batch):
                for user in ['a_data','b_data']:

                    for typ in tweet_types:
                        tweets = dyad[user][typ]
                        # if len(tweets)>5:
                        #     tweets = resample(tweets,replace=False,n_samples=5)
                        sentences.append(' '.join(tweets))
                        sentence_pos_idx.append((dyad_idx,self.feature2pos_idx[(user,typ)][0]))

                    if self.hparams.use_bio:
                        bio = dyad[user]['bio']
                        bio = bio if bio else ''
                        name = ' '.join(dyad[user]['name'])
                        bio_text = name+' '+bio
                        sentences.append(bio_text)
                        sentence_pos_idx.append((dyad_idx,
                             self.feature2pos_idx[(user,'bio')][0]))

            # encode text
            text_output = self.tokenizer.batch_encode_plus(
                sentences,
                max_length=self.hparams.max_length,
                padding='max_length',
                # padding='longest',
                truncation='longest_first',
                return_length=False)
            for k,v in text_output.items():
                output[k]=v
            output['text_pos_idx'] = sentence_pos_idx

        # add other features
        if self.hparams.use_activity | \
            self.hparams.use_count | \
            self.hparams.use_network:

            columns = [] # gets column names
            feat_emb_idx = [] # for identifying which embedding matrix to use
            feat_pos_idx = [] # gets position embedding index
            # get column names and embedding indices
            for c,user in zip(['a:','b:'],['a_data','b_data']):
                if self.hparams.use_activity:
                    subcolumns=['friends_count','followers_count','statuses_count']
                    columns.extend([c+col for col in subcolumns])
                    feat_emb_idx.extend([self.feature2emb_idx[col] for col in subcolumns])
                    feat_pos_idx.extend(self.feature2pos_idx[(user,'activity')])

                if self.hparams.use_count:
                    subcolumns = ['num_pm','num_dm','num_rt','frac_pm','frac_dm','frac_rt']
                    columns.extend([c+col for col in subcolumns])
                    feat_emb_idx.extend([self.feature2emb_idx[col] for col in subcolumns])
                    feat_pos_idx.extend(self.feature2pos_idx[(user, 'count')])

            if self.hparams.use_network:
                subcolumns=['net_jacc','net_aa']
                columns.extend(subcolumns)
                feat_emb_idx.extend([self.feature2emb_idx[col] for col in subcolumns])
                feat_pos_idx.extend(self.feature2pos_idx['network'])

            # collect data for each feature type
            raw_features = [] # matrix that stores the raw features
            for dyad_idx,dyad in enumerate(batch):
                tmp_arr = []
                for user in ['a_data', 'b_data']:
                    if self.hparams.use_activity:
                        tmp_arr.extend(dyad[user]['activity'])
                    if self.hparams.use_count:
                        cnt_arr = np.array([len(dyad[user][typ]) for typ in ['public-mention',
                                                                             'direct-mention',
                                                                             'retweet']])
                        cnt_arr_norm = cnt_arr/max(1.0,cnt_arr.sum())
                        tmp_arr.extend(cnt_arr.tolist()+cnt_arr_norm.tolist())
                if self.hparams.use_network:
                    tmp_arr.extend(dyad['network'])

                raw_features.append(tmp_arr)

            # convert to binned results
            df=pd.DataFrame(raw_features,columns=columns)
            # print(df)
            for i,col in enumerate(columns):
                bins=self.feature2bins[col.split(':')[-1]]
                # print(i,col,len(bins))
                df[col] = pd.cut(df[col],
                    bins=bins,
                    labels=list(range(len(bins)-1)),
                    include_lowest=True)
            # binned_features = df.to_numpy()
            binned_features = df.to_numpy(dtype=int)
            # print(binned_features)
            # print(binned_features.dtype)

            # add to output
            output['features_binned'] = binned_features
            output['features_pos_idx'] = feat_pos_idx
            output['features_emb_idx'] = feat_emb_idx

        output['labels']=[self.category2idx[dyad['category']] for dyad in batch]

        for k,v in output.items():
            # print(k,v)
            # print(k)
            # if k=='features_binned':
            #     print(v)
            output[k]=torch.tensor(v)
        return output

if __name__=='__main__':
    # snippet to test if dataloader worksß

    dm = RelationshipClassificationDataModule(
        train_data_file='/shared/0/projects/relationships/working-dir/relationship-prediction-new/data/processed-data/val_3plus.json.gz',
        val_data_file='/shared/0/projects/relationships/working-dir/relationship-prediction-new/data/processed-data/val_3plus.json.gz',
        balance_training_set=True,
        use_public_mention=True,
        use_direct_mention=True,
        use_retweet=True,
        use_bio=True,
        use_activity=True,
        use_count=True,
        use_network=True
    )
    dm.setup(stage='fit')
    batch=next(iter(dm.train_dataloader()))
    print(batch)
    for k,v in batch.items():
        print(k,v.size())