import torch
from torch import nn
from torch.optim import AdamW
from transformers import (
    AutoModel,
    BertConfig,
    get_linear_schedule_with_warmup,
    get_constant_schedule,
)
from transformers.models.bert.modeling_bert import *
import numpy as np
from pytorch_lightning import LightningModule, Trainer
from sklearn.metrics import f1_score, accuracy_score

class RelationshipClassifier(LightningModule):
    def __init__(
        self,
        hidden_dim,
        position_embedding_dim,
        position_embedding_mode,
        learning_rate,
        adam_epsilon,
        warmup_steps,
        weight_decay,
        **kwargs
        ):
        super().__init__()
        self.save_hyperparameters()
        return

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("RelationshipClassifier")
        # model settings
        parser.add_argument("--hidden_dim",type=int,default=768)
        parser.add_argument("--position_embedding_dim",type=int,default=128)
        parser.add_argument("--position_embedding_mode",type=str,
                    default='append',choices=['append','add'])
        # training settings
        parser.add_argument("--learning_rate",type=float,default=1e-5)
        parser.add_argument("--adam_epsilon",type=float,default=1e-8)
        parser.add_argument("--warmup_steps",type=float,default=0.0)
        parser.add_argument("--weight_decay",type=float,default=0.01)
        return parent_parser

    def setup(self, stage: str = None) -> None:
        # setup model
        self.use_text = self.trainer.datamodule.hparams.use_public_mention | \
                        self.trainer.datamodule.hparams.use_direct_mention | \
                        self.trainer.datamodule.hparams.use_retweet | \
                        self.trainer.datamodule.hparams.use_bio
        self.use_feature = self.trainer.datamodule.hparams.use_activity | \
                           self.trainer.datamodule.hparams.use_count | \
                           self.trainer.datamodule.hparams.use_network

        if self.hparams.position_embedding_mode=='add':
            feature_embedding_dim = self.hparams.hidden_dim
        elif self.hparams.position_embedding_mode=='append':
            feature_embedding_dim = self.hparams.hidden_dim + self.hparams.position_embedding_dim

        # load text encoder
        self.text_encoder = AutoModel.from_pretrained(
            self.trainer.datamodule.hparams.language_model_name_or_path,
            cache_dir=self.trainer.datamodule.hparams.language_model_cache_dir)

        # load feature bins and create position and feature embeddings
        self.feature2bins={}
        self.idx2feature_name=[]
        with open(self.hparams.feature_bin_file_dir) as f:
            for i,line in enumerate(f):
                line=line.strip().split(',')
                self.feature2bins[line[0]]=int(line[1])+1
                self.idx2feature_name.append(line[0])

        embedding_dict={}
        embedding_dict['position_embedding'] = nn.Embedding(60,
                                                    self.hparams.position_embedding_dim)
        for k,n_bins in self.feature2bins.items():
            embedding_dict[k]= nn.Embedding(n_bins,self.hparams.hidden_dim)
        self.model_embeddings = nn.ModuleDict(embedding_dict)

        # load encoder mixer
        config = BertConfig()
        config.update({'hidden_size':feature_embedding_dim, 'num_attention_heads':8})
        self.encoder_mixer = nn.ModuleList([BertLayer(config) for _ in range(3)])
        self.classifier = nn.Linear(feature_embedding_dim,5)
        self.criterion = nn.CrossEntropyLoss()

        if stage!='fit':
            return
        # for training stage: get number of steps
        train_loader = self.trainer.datamodule.train_dataloader()

        # Calculate total steps
        tb_size = self.trainer.datamodule.hparams.train_batch_size
        # ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
        self.total_steps = (len(train_loader.dataset) // tb_size) * self.trainer.max_epochs
        if type(self.hparams.warmup_steps)==float:
            if self.hparams.warmup_steps<=1:
                self.hparams.warmup_steps = int(self.total_steps*self.hparams.warmup_steps)
            else:
                self.hparams.warmup_steps = int(self.hparams.warmup_steps)

    def forward(self, batch, use_text=True, use_feature=False):
        batch_size = len(batch['labels'])
        # stage 1: obtain text representations
        combined_outputs = []
        if use_text:
            # print(batch['input_ids'])
            # print(batch['input_ids'].max())
            pooled_outputs = self.text_encoder(input_ids=batch['input_ids'],
                                               attention_mask=batch['attention_mask'])[1]
            pooled_outputs = pooled_outputs.reshape(batch_size,-1,self.hparams.hidden_dim)
            # print(batch['input_ids'].shape)
            text_position_idxs = batch['text_pos_idx'][:,1].reshape(batch_size,-1)
            text_position_embeddings = self.model_embeddings['position_embedding'](text_position_idxs)
            if self.hparams.position_embedding_mode=='append':
                text_outputs = torch.cat([pooled_outputs,text_position_embeddings],dim=2)
            elif self.hparams.position_embedding_mode=='add':
                text_outputs = pooled_outputs + text_position_embeddings
            combined_outputs.append(text_outputs)

        if use_feature:
            feature_binned_embeddings = []
            # print(batch['features_binned'])
            # print(batch['features_emb_idx'])
            for i,feature_idx in enumerate(batch['features_emb_idx'].cpu().tolist()):
                bin_idxs = batch['features_binned'][:,i]
                # print('bin_idxs',bin_idxs)
                feature_name = self.idx2feature_name[feature_idx]
                # print(feature_name)
                # print(self.feature2bins[feature_name])
                # print('max_val',bin_idxs.max())
                embedding_values = self.model_embeddings[feature_name](bin_idxs)
                feature_binned_embeddings.append(embedding_values)
            feature_binned_embeddings = torch.stack(feature_binned_embeddings,dim=1)
            # print(batch['features_pos_idx'].shape)
            feature_position_embeddings = self.model_embeddings['position_embedding'](batch['features_pos_idx'])
            feature_position_embeddings = feature_position_embeddings.unsqueeze(0)
            size = feature_position_embeddings.shape
            feature_position_embeddings = feature_position_embeddings.expand(batch_size,size[1],size[2])
            if self.hparams.position_embedding_mode=='append':
                # print(feature_binned_embeddings.shape)
                # print(feature_position_embeddings.shape)
                feature_outputs = torch.cat([feature_binned_embeddings,feature_position_embeddings],dim=2)
            elif self.hparams.position_embedding_mode=='add':
                feature_outputs = feature_binned_embeddings + feature_position_embeddings
            combined_outputs.append(feature_outputs)
        if len(combined_outputs)==2:
            combined_outputs = torch.cat(combined_outputs,dim=1)
        elif len(combined_outputs)==1:
            combined_outputs = combined_outputs[0]

        # put into encoder mixer
        for layer in self.encoder_mixer:
            combined_outputs = layer(combined_outputs)[0]
        pooled_outputs = combined_outputs.max(1)[0]
        logits = self.classifier(pooled_outputs)
        return logits

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch, use_text=self.use_text, use_feature=self.use_feature)
        labels = batch['labels']
        loss = self.criterion(logits,labels)
        self.log("tr_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        result = {}
        logits = self.forward(batch, use_text=self.use_text, use_feature=self.use_feature)
        labels = batch['labels']
        loss = self.criterion(logits,labels)
        result['loss'] = loss.item()
        result['preds'] = logits.argmax(1).detach().cpu().tolist()
        result['answers'] = labels.detach().cpu().tolist()
        return result

    def validation_epoch_end(self, results):
        y_true, y_pred = [], []
        val_loss = []

        for res in results:
            y_pred.extend(res['preds'])
            y_true.extend(res['answers'])
            val_loss.append(res['loss'])
        f1 = f1_score(y_true,y_pred,average='macro')
        acc = accuracy_score(y_true,y_pred)
        val_loss = np.mean(val_loss)
        log_dict = {'val_f1':round(f1,3),
                    'val_acc':round(acc,3),
                    'val_loss':round(val_loss,3)}
        self.log_dict(log_dict,prog_bar=True)
        return

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, results):
        y_true, y_pred = [], []
        val_loss = []

        for res in results:
            y_pred.extend(res['preds'])
            y_true.extend(res['answers'])
            val_loss.append(res['loss'])
        f1 = f1_score(y_true,y_pred,average='macro')
        acc = accuracy_score(y_true,y_pred)
        val_loss = np.mean(val_loss)
        log_dict = {'test_f1':round(f1,3),
                    'test_acc':round(acc,3),
                    'test_loss':round(val_loss,3)}
        self.log_dict(log_dict,prog_bar=True)
        return log_dict

    def predict_step(self, batch, batch_idx):
        result = {}
        logits = self.forward(batch, use_text=self.use_text, use_feature=self.use_feature)
        logits = logits.softmax(dim=1).detach().cpu().max(dim=1)
        result['scores'] = logits.values.tolist()
        result['labels'] = logits.indices.tolist()
        return result

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']

        no_decay_params = []
        decay_params = []

        # all_params = []
        all_params = [self.text_encoder, self.model_embeddings, self.encoder_mixer, self.classifier]
        # for component in [self.text_encoder, self.model_embeddings, self.encoder_mixer, self.classifier]:
        #     all_params.extend(list(component.values()))
        # all_params.extend(list(self.classifier_dict.values()))
        # all_params.extend(list(self.classifier_dict.values()))
        for module in all_params:
            no_decay_params.extend([p for n, p in module.named_parameters() if not any(nd in n for nd in no_decay)])
            decay_params.extend([p for n, p in module.named_parameters() if any(nd in n for nd in no_decay)])

        optimizer_grouped_parameters = [
            {"params": no_decay_params, "weight_decay": self.hparams.weight_decay},
            {"params": decay_params, "weight_decay": 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.learning_rate,
                          eps=self.hparams.adam_epsilon)

        if self.hparams.warmup_steps>0:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.warmup_steps,
                num_training_steps=self.total_steps
            )
        else:
            scheduler = get_constant_schedule(
                optimizer
            )
        # scheduler = get_constant_schedule_with_warmup(
        #     optimizer,
        #     num_warmup_steps=self.hparams.warmup_steps,
        # )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

if __name__=='__main__':
    from pl_dataloader import *
    dm = RelationshipClassificationDataModule(
        train_data_file='/shared/0/projects/relationships/working-dir/relationship-prediction-new/data/processed-data/val_3plus.json.gz',
        val_data_file='/shared/0/projects/relationships/working-dir/relationship-prediction-new/data/processed-data/val_3plus.json.gz',
        test_data_file='/shared/0/projects/relationships/working-dir/relationship-prediction-new/data/processed-data/val_3plus.json.gz',
        balance_training_set=True,
        use_public_mention=True,
        use_direct_mention=True,
        use_retweet=True,
        use_bio=True,
        # use_activity=True,
        # use_count=True,
        use_network=True,
    )
    # dm.setup(stage='fit')
    # batch=next(iter(dm.test_dataloader()))
    cls=RelationshipClassifier(
    position_embedding_dim=128
    )
    trainer = Trainer(limit_test_batches=10,
                      accelerator='gpu')
    trainer.test(cls,datamodule=dm)
    # logits = cls.forward(batch)
    # print(logits)
    # print(cls)