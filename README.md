# relationships
Official repository for the ICWSM '21 paper "More than meets the tie: Examining the Role of Interpersonal Relationships in Social Networks" [(link)](https://ojs.aaai.org/index.php/ICWSM/article/view/18045)
If using the package for research, please use the following citation:
```
@article{choi2021relationships, 
    title={More than Meets the Tie: Examining the Role of Interpersonal Relationships in Social Networks}, 
    volume={15}, 
    url={https://ojs.aaai.org/index.php/ICWSM/article/view/18045}, 
    number={1}, 
    journal={Proceedings of the International AAAI Conference on Web and Social Media}, 
    author={Choi, Minje and Budak, Ceren and Romero, Daniel M. and Jurgens, David}, 
    year={2021}, 
    month={May}, 
    pages={105-116} }
```
## Requirements

This code was tested under the following package dependencies. The package versions do not have to be exact.

- pytorch==1.10.0
- transformers==4.9.2
- tokenizers==0.10.1
- tweepy==3.10.0
- tqdm==4.49.0
- pandas==1.2.4
- scikit-learn==0.24.2

This model also requires the installation of BerTweet [(link)](https://huggingface.co/vinai/bertweet-base). 
Please keep this in mind if you plan to run the model in an offline environment where the huggingface models and tokenizers cannot be automatically installed.

## Example of training model

```
CUDA_VISIBLE_DEVICES=0 python run.py \
 --do_train --use_dm --use_pm --use_rt --use_name --use_bio --use_network --use_count \
 --train_data_file=data/train-data-camera-ready.json \
 --val_data_file=data/train-data-camera-ready.json \
 --output_dir=data \
 --checkpoint_interval=50000 --valid_interval=50000 --batch_size=8 \
 --classifier_config_dir=data/bert-config.json
```


## Inferencing relationship types from tweets collected via the Twitter API
We provide methods for inferring the relationships between any two Twitter users, provided that both of their tweet information is available by the Twitter API.
You will need to have an application of the Twitter API [(link)](https://developer.twitter.com/en/docs/twitter-api).

### 0. Download the pretrained model provided by the authors
```
Model trained on all features: [(link)](https://drive.google.com/file/d/1jrDiaSWJyrx6U08qlXg9kaWq9-vgYRzN/view?usp=share_link)
Model trained on all features except network features: [(link)](https://drive.google.com/file/d/1Z1536SEIT9cVk127FY6nvdaCPf84GSD5/view?usp=share_link)
```

### 1. Update the credentials in `data/credentials.txt`.

### 2. Create a list of Twitter dyads that you want to infer in `data/dyad-examples.txt`. 
Either the username (the name following @) or the user's ID number can be used, but they must be separated through a ','.

```
Examples:

username1,username2
user_id1,user_id2
user_id3,username3
```

### 3. Download their interactions by running `python collect-from-twitter.py`. 
The interactions will be stored in `data/sample_outputs.json`.

### 4. Infer the relationships with the model using the following code. 
```
CUDA_VISIBLE_DEVICES=0 python run.py \
    --do_predict --predict_batch_size=16 --accelerator gpu \
    --use_public_mention --use_direct_mention --use_retweet \
    --use_bio --use_activity --use_count \
    --predict_data_file FILE_PATH_OF_INPUT_FILE \
    --save_file=FILE_PATH_OF_OUTPUT_FILE_CONTAINING_INFERRED_SCORES \
    --ckpt_path=FILE_PATH_OF_PRETRAINED_MODEL
```
The results will be stored in `data/sample_outputs.json-predictions.tsv`.

## Request for relationship dataset
Unfortunately, we do not provide the datasets which we used for training the models as it contains conversations of real Twitter users. 
However, we are able to provide a dataset consisting of the user IDs and the relationship categories of the user dyads in Twitter.
If interested, please send a request email to [minje@umich.edu].
These are available at [(link)](https://drive.google.com/file/d/1VKn49kVB-96RTpfvEWE95iUyfWlaxpFw/view?usp=share_link)

### Contact
For questions, please contact the main author via [minje@umich.edu]. Thanks!
