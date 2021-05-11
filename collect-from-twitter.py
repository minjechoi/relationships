"""
Uses the Twitter API to collect the data required for inferring the relationship between two users
Requires access to the Twitter API for inference
"""
import tweepy
import numpy as np
import json
from tqdm import tqdm

def get_api_credentials():
    credentials={}
    with open('data/credentials.txt') as f:
        for line in f:
            category,key = line.strip().split(':')
            credentials[category.strip()]=key.strip()

    # authorization of consumer key and consumer secret
    auth = tweepy.OAuthHandler(credentials['consumer_key'], credentials['consumer_secret'])
    # set access to user's access key and access secret
    auth.set_access_token(credentials['access_token'], credentials['access_token_secret'])
    # calling the api
    api = tweepy.API(auth,wait_on_rate_limit=True,wait_on_rate_limit_notify=True)
    print("Loaded Tweepy instance of the Twitter API!")
    return api

def get_dyads(dyad_file='data/dyad-examples.txt'):
    dyad_list=[]
    with open(dyad_file) as f:
        for line in f:
            dyad_list.append(line.strip().split(','))
    return dyad_list

def process_twitter_data():
    """
    Processes the data collected from Tweepy into a form that can be run by the model
    """
    # load the API
    api = get_api_credentials()

    def extract_text(tweet):
        if 'full_text' in tweet:
            return tweet['full_text']
        elif 'text' in tweet:
            return tweet['text']
        else:
            return None
    out_list = []

    list_of_dyads = get_dyads()

    print("Getting the data for dyads")
    for user_name_or_id_1,user_name_or_id_2 in tqdm(list_of_dyads):
        # get user objects and store user info into dictionary
        user1 = api.get_user(user_name_or_id_1)._json
        user2 = api.get_user(user_name_or_id_2)._json

        aid,bid = user1['id_str'],user2['id_str']

        out_obj={'aid':aid,'bid':bid,
                 'a_data':{'name':[user1['screen_name'],user1['name']],'bio':user1['description'],
                           'direct-mention':[],'public-mention':[],'retweets':[]},
                 'b_data':{'name':[user2['screen_name'],user2['name']],'bio':user2['description'],
                           'direct-mention':[],'public-mention':[],'retweets':[]},
                 }

        networked_users = {aid:set(),bid:set()}
        vol_users = {aid:0,bid:0}
        for uid1,uid2,direction in [(aid,bid,'a_data'),(bid,aid,'b_data')]:
            # get all tweets from a user's timeline
            all_tweets = []
            oldest_id=None
            while True:
                tweets = api.user_timeline(user_id=aid,count=200,include_rts=True,
                                           max_id=oldest_id,tweet_mode='extended')
                if len(tweets) == 0:
                    break
                oldest_id = tweets[-1].id-1
                all_tweets.extend([tweet._json for tweet in tweets])

            relevant_tweets = []
            for tweet in all_tweets:
                if 'retweeted_status' in tweet:
                    uid_rt = tweet['retweeted_status']['user']['id_str']
                    networked_users[uid1].add(uid_rt)
                    if uid_rt == bid:
                        relevant_tweets.append(('retweets', extract_text(tweet['retweeted_status'])))
                elif 'quoted_status' in tweet:
                    uid_qt = tweet['quoted_status']['user']['id_str']
                    networked_users[uid1].add(uid_qt)
                    if uid_qt == bid:
                        relevant_tweets.append(('retweets', extract_text(tweet['quoted_status'])))
                else:
                    if tweet['in_reply_to_user_id_str']:
                        uid_rp = tweet['in_reply_to_user_id_str']
                        networked_users[uid1].add(uid_rp)
                        if uid_rp == bid:
                            relevant_tweets.append(('direct-mention', extract_text(tweet)))
                    else:
                        uids_mn = [x['id_str'] for x in tweet['entities']['user_mentions']]
                        networked_users[uid1].update(uids_mn)
                        if bid in uids_mn:
                            relevant_tweets.append(('public-mention', extract_text(tweet)))
            # update with tweets
            for typ,tweet in relevant_tweets:
                out_obj[direction][typ].append(tweet)
            # get count-norm
            arr = np.array([len(out_obj[direction]['direct-mention']),len(out_obj[direction]['public-mention']),
                            len(out_obj[direction]['retweets'])])
            vol_users[uid1]=arr.sum()
            arr = arr/max(1,arr.sum())
            out_obj[direction]['count_norm']=arr.tolist()

            # fill none values with dummy
            for typ,V in out_obj[direction].items():
                if len(V)==0:
                    out_obj[direction][typ]=['<None>']

        # get Jaccard index based on neighbors
        jacc = len(networked_users[aid] & networked_users[bid]) / len(networked_users[aid] | networked_users[bid])

        # get reciprocity score based on activity
        rec = 1-np.abs(vol_users[aid]-vol_users[bid])/(vol_users[aid]+vol_users[bid])
        out_obj['network']=[jacc,rec]

        out_list.append(out_obj)

    # save out-list
    print("Saving to data/sample_outputs.json...")
    with open('data/sample_outputs.json','w') as outf:
        for obj in out_list:
            outf.write(json.dumps(obj)+'\n')

    return

if __name__=='__main__':
    process_twitter_data()