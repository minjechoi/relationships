import torch
from torch import nn
from transformers import RobertaModel, RobertaConfig

class ClassifierLayer(nn.Module):
    """
    An end-level module of linear layers that concatenates both embedding features and other features and obtains the classification results
    """
    def __init__(self, dim_h=768, n_class=5, use_network=False, use_count=False):
        super(ClassifierLayer, self).__init__()

        dim_in = dim_h
        if use_network:
            dim_in+=2
        if use_count:
            dim_in+=6 # 3x2
        self.layer1 = nn.Linear(dim_in,512)
        self.layer2 = nn.Linear(512,256)
        self.out_proj = nn.Linear(256,n_class)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1,inplace=False)
        self.loss_fn = nn.CrossEntropyLoss()
        return

    def forward(self, inputs, labels=None):
        x = self.layer1(inputs)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.dropout(x)
        outputs = self.out_proj(x)

        if labels!=None:
            loss = self.loss_fn(outputs,labels)
            return loss,outputs
        else:
            return outputs

class ConcatenatedClassifier(nn.Module):
    """
    An end-level classifier that concatenates both embedding features and other features
    """
    def __init__(self, classifier_config_dir, device, task_type, n_clf_layers=6, use_dm=True, use_pm=True, use_rt=True, use_bio=False, use_name=False, use_network=False, use_count=False):
        super(ConcatenatedClassifier, self).__init__()
        # load text model
        self.device = device
        self.task_type = task_type
        self.use_text = use_dm | use_pm | use_rt
        self.use_bio = use_bio
        self.use_name = use_name
        self.use_etc = use_network | use_count
        self.text_model = RobertaModel.from_pretrained("vinai/bertweet-base",
                       output_attentions=False,output_hidden_states=False)
        if self.use_name:
            self.charEmbedding = nn.Embedding(num_embeddings=302,embedding_dim=300,
                  padding_idx=301) # 302: 300-top frequent + pad + unk
            self.conv3 = nn.Conv1d(in_channels=300,out_channels=256,kernel_size=3,padding=1)
            self.conv4 = nn.Conv1d(in_channels=300,out_channels=256,kernel_size=4,padding=1)
            self.conv5 = nn.Conv1d(in_channels=300,out_channels=256,kernel_size=5,padding=1)

        # load classifier for combining these features
        config = RobertaConfig()
        config = config.from_json_file(classifier_config_dir)
        config.num_hidden_layers = n_clf_layers
        config.num_attention_heads = n_clf_layers
        config.max_position_embeddings = 7
        if self.use_bio:
            config.max_position_embeddings+=2
        if self.use_name:
            config.max_position_embeddings+=4
        self.concat_model = RobertaModel(config)
        self.classifier = ClassifierLayer(use_count=use_count,use_network=use_network)
        return

    def forward(self, batch):
        """
        A function for training each step
        :param batch: batch data
        :return: outputs,loss
        """
        # 1) load data and change to cuda
        if self.task_type!='infer':
            labels = batch['labels']
            labels = torch.tensor(labels).to(self.device)
        if self.use_text:
            texts, masks, n_samples, pos_out = batch['tweet']
            texts, masks = texts.to(self.device), masks.to(self.device)
            hidden = self.text_model(input_ids=texts, attention_mask=masks)[0]
        if self.use_bio:
            bio_texts, bio_masks, bio_n_samples, bio_pos_out = batch['bio']
            bio_texts, bio_masks = bio_texts.to(self.device), bio_masks.to(self.device)
            hidden_b = self.text_model(input_ids=bio_texts, attention_mask=bio_masks)[0]

        if self.use_name:
            names, name_pos_out = batch['name']
            names = names.to(self.device)
            emb_names = self.charEmbedding(names) # batch x seq x dim (300)
            emb_names = emb_names.transpose(2,1)
            hidden_n3 = self.conv3(emb_names).max(-1)[0] # batch x 256
            hidden_n4 = self.conv4(emb_names).max(-1)[0]
            hidden_n5 = self.conv5(emb_names).max(-1)[0]
            hidden_name = torch.cat([hidden_n3,hidden_n4,hidden_n5],1) # 4*batch x 256
            arr_n = hidden_name # batch*4 x 256 (4 since there are 4 types of names)

        if self.use_etc:
            # if using either networks or normalized counts
            hidden_etc = batch['etc'].to(self.device)

        # 3) reorganize hidden states so that each dyad gets a stacked matrix containing all the tweets and bio info
        max_ln = max(n_samples)
        text_lengths = masks.sum(1).tolist()  # length of each text sentence, will be used to get the avg of each
        if self.use_bio:
            bio_lengths = bio_masks.sum(1).tolist()
            max_ln += 2
        if self.use_name:
            max_ln += 4

        idx = 0  # idx of which sentence to look at
        idx_b = 0  # idx of which bio to look at
        concat_hidden = []  # concatenated version of all hidden layers
        concat_masks = []
        position_ids = []  # a batch * seq size matrix, storing the position ids

        start_idx=0
        for ln, n in enumerate(n_samples):
            arr = [hidden[i, :text_lengths[i], :].mean(0) for i in range(idx, idx + n)]  # get the summary of a sentence using the mean
            try:
                arr = torch.stack(arr, 0)  # stack the sentences to create a n_texts * dim size matrix
            except:
                print(ln,n,text_lengths)
                print(texts[0])
                import sys
                sys.exit(0)
            idx += n  # add idx to look into the next dyad and its sentences
            pos_tmp = []
            pos_tmp.extend(pos_out[start_idx:start_idx+n])  # the position settings of the sentences
            start_idx+=n
            if self.use_bio:
                arr_b = torch.stack([hidden_b[i, :bio_lengths[i]].mean(0) for i in range(idx_b, idx_b + 2)], 0)
                idx_b += 2
                arr = torch.cat([arr, arr_b], 0)
                # pos_tmp.extend(bio_pos_out[ln])
                pos_tmp.extend([6,7])

            if self.use_name:
                arr = torch.cat([arr,arr_n[ln*4:(ln+1)*4]], 0)
                pos_tmp.extend([8,9,10,11])

            concat_masks.append([1] * len(arr) + [0] * (max_ln - len(arr)))
            if len(arr) < max_ln:
                pos_tmp.extend(
                    [0] * (max_ln - len(arr)))  # attach zeros to pos_tmp so that it becomes the max sequence length

                # get a block of max_seq * dim by creating a temporary block filled with zeros
                tmp_block = torch.zeros(max_ln - len(arr), arr.shape[1])
                tmp_block = tmp_block.to(self.device)
                arr = torch.cat([arr, tmp_block], 0)
            concat_hidden.append(arr)  # concatenate these
            position_ids.append(pos_tmp)
        concat_hidden = torch.stack(concat_hidden, 0)
        concat_masks = torch.tensor(concat_masks).to(self.device)
        position_ids = torch.tensor(position_ids).to(self.device)
        token_type_ids = torch.zeros_like(position_ids).to(self.device)

        # second model: get vector representation of all text features
        hidden_2 = self.concat_model(inputs_embeds=concat_hidden, attention_mask=concat_masks, position_ids=position_ids, token_type_ids=token_type_ids)[0]
        hidden_2 = hidden_2[:,0] # batch x 768

        # third model: concatenate with existing features, then classify
        if self.use_etc:
            hidden_2 = torch.cat([hidden_2,hidden_etc],1) # batch x (768+n)
        if self.task_type=='infer':
            outputs = self.classifier(hidden_2,labels=None)
        else:
            outputs = self.classifier(hidden_2,labels=labels)
        return outputs
