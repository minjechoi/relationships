import torch
from torch.utils.data import Dataset,DataLoader
import json
from random import sample
from tqdm import tqdm
from time import time
from transformers import AutoTokenizer

class CustomDataset(Dataset):
    def __init__(self, data_list, task_type, use_pm=False, use_dm=False, use_rt=False, use_bio=False, use_name=False,
                 use_network=False, use_count=False, max_samples=20):

        """
        A Dataset class for dealing with all the texts in this model
        :param data_list: a list where each sample is a dictionary object containing the information of a dyad required for classifying relationships

        Arguments for which features to include
        :param task_type: Whether the task is "train", "eval" or "infer"
        :param use_pm: If true, include public mention messages in the model
        :param use_dm: If true, include directed messages in the model
        :param use_rt: If true, include retweets in the model
        :param use_bio: If true, include bio of the two users as additional information
        :param use_name: If true, include information from the names of the two users as additional information
        :param use_network: If true, include the network features (jaccard similarity & adamic-adar) of the two users
        :param use_count: If true, include the distribution of the tweet types for each user (ratio of direct mentions / public mentions / replies)

        Arguments for which number of tweets to include
        :param max_samples: If the number of available tweets exceed this number, sample to this number
        """
        assert task_type in ['train','test','infer'], "The argument 'task_type' should either be 'train', 'eval', or 'infer'"
        self.task_type = task_type
        self.use_pm = use_pm
        self.use_dm = use_dm
        self.use_rt = use_rt
        self.use_bio = use_bio
        self.use_name = use_name
        if self.use_name:
            self.char2idx={}
            with open('data/char2idx.tsv') as f:
                for ln, line in enumerate(f):
                    self.char2idx[line.split()[0]] = ln
        self.use_network = use_network
        self.use_count = use_count
        self.max_samples = max_samples
        self.dyad_ids = []

        self.process_data_file(data_list)

        self.tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)


    def process_data_file(self, data_list):
        start = time()
        self.data = []
        # load text and bio from the tokenized file
        pbar = tqdm(data_list)
        for obj in pbar:
            self.dyad_ids.append((obj['aid'],obj['bid'])) # append id pairs to
            out_obj={'text':[]}

            if self.task_type!='infer':
                out_obj['label'] = ['social','romance','family','organizational','parasocial'].index(obj['category'])
            if self.use_bio:
                out_obj['bio']=[]
            if self.use_name:
                out_obj['name']=[]
            if self.use_count:
                out_obj['count_norm']=[]

            """
            For each data sample, append the position index
            1) tweet types for user A (0,1,2) and user B (3,4,5)
            2) description bio for user A (6) and user B (7)
            3) screen name and full name for user A (8,9) and user B (10,11)
            """

            for i,user in enumerate(['a_data', 'b_data']):
                # add text data
                text_samples = []
                for j,(flag,tweet_type) in enumerate(zip([self.use_pm,self.use_dm,self.use_rt],['public-mention','direct-mention','retweets'])):
                    if flag:
                        for tweet in obj[user][tweet_type]:
                            text_samples.append((i*3+j,tweet))
                if len(text_samples)>self.max_samples:
                    text_samples = sample(text_samples,self.max_samples)
                out_obj['text'].extend(text_samples)

                # add bio
                if self.use_bio:
                    out_obj['bio'].append((6+i,obj[user]['bio']))

                # add name
                if self.use_name:
                    for j,name in enumerate(obj[user]['name']):
                        out_obj['name'].append((8+2*i+j,name))

                # add counts
                if self.use_count:
                    out_obj['count_norm'].append(obj[user]['count_norm'])

            # add networks
            if self.use_network:
                out_obj['network']=obj['network']

            self.data.append(out_obj)
            pbar.set_description("Loading data...")

        print('%d seconds to load %d lines!' % (int(time() - start), len(self.data)))
        # get training dataset distribution
        print("%d samples"%len(self.data))
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self,sample_list):
        out_obj={}
        texts = [] # stacked version of all texts
        tweet_positions = [] # stacked version of all position encodings for text
        n_samples = [] # contains the number of tweets per each dyad
        if self.task_type!='infer':
            labels = [obj['label'] for obj in sample_list]
            out_obj['labels']=labels
        for obj in sample_list:
            # process text
            tweet_positions.extend([x[0] for x in obj['text']])
            texts.extend([x[1] for x in obj['text']])
            n_samples.append(len(obj['text']))
        enc = self.tokenizer(text=texts, padding='longest', truncation=True, max_length=64)
        tweet_input_ids, tweet_masks = torch.tensor(enc['input_ids']), torch.tensor(enc['attention_mask'])
        out_obj['tweet']=(tweet_input_ids,tweet_masks,n_samples,tweet_positions)

        if self.use_bio:
            bios = []
            bio_positions = []
            n_samples = []
            for obj in sample_list:
                bio_positions.append([x[0] for x in obj['bio']])
                bios.extend([x[1] for x in obj['bio']])
                n_samples.append(2)
            enc = self.tokenizer(text=bios, padding='longest', truncation=True, max_length=64)
            bio_input_ids,bio_masks = torch.tensor(enc['input_ids']), torch.tensor(enc['attention_mask'])
            out_obj['bio']=(bio_input_ids,bio_masks,n_samples,bio_positions)

        if self.use_name:
            names = []
            name_positions = []
            for obj in sample_list:
                name_positions.append([x[0] for x in obj['name']])
                names.extend([x[1] for x in obj['name']])
            max_ln = min(30, max([len(v) for v in names]))
            names_out = []
            unk_token = 300
            pad_token = 301
            for name in names:
                name2 = []
                for c in list(name):
                    if c in self.char2idx:
                        c = self.char2idx[c]
                        if c > unk_token:
                            c = unk_token
                    else:
                        c = unk_token
                    name2.append(c)
                if len(name2) > max_ln:
                    name2 = name2[:max_ln]
                else:
                    name2 = name2 + [pad_token] * (max_ln - len(name2))
                names_out.append(name2)
            enc = torch.tensor(names_out)
            out_obj['name']=(enc,name_positions)

        if (self.use_network)|(self.use_count):
            net_out = []
            for i, obj in enumerate(sample_list):
                net_tmp = []
                if self.use_network:
                    net_tmp.extend(obj['network'])
                if self.use_count:
                    for count_norm in obj['count_norm']:
                        net_tmp.extend(count_norm)
                net_out.append(net_tmp)
            out_obj['etc']=torch.tensor(net_out)

        return out_obj

if __name__=='__main__':
    # test custom loader
    out=[]
    with open('data/sample_outputs.json') as f:
        for line in f:
            obj=json.loads(line)
            out.append(obj)

    dataset = CustomDataset(data_list=out,task_type='infer',use_pm=True,use_dm=True,use_rt=True,use_bio=True,use_name=True,use_network=True,use_count=True,max_samples=20)
    print(dataset.data[0])

    dataloader = DataLoader(dataset, batch_size=16,shuffle=False, num_workers=4, collate_fn=dataset.collate_fn)
    obj = next(iter(dataloader))
    print(obj)