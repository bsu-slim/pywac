'''
Created on Aug 21, 2018

@author: ckennington
'''
from wac import WAC
import sqlite3
import pandas as pd
from scipy.spatial import distance
from pandasql import sqldf
import sklearn
from sklearn import ensemble
import random
import numpy as np

'''
This evaluates the WAC model using the opencv derived TAKE data. 
The final accuracy should be >=0.6 if you use `prod` for composition
'''

wac = WAC('take_test')

'''
requires take.db to be in the working folder
'''
print('preparing data...')
# connect to the database
con = sqlite3.connect('take.db')
# get raw features
tiles = pd.read_sql_query("SELECT * FROM cv_piece_raw", con)
# do a one-hot encoding of string features
tiles['v_top_skewed'] = tiles.v_skew == 'top_skewed'
tiles.v_top_skewed = tiles.v_top_skewed.astype(int)
tiles['v_symmetric'] = tiles.v_skew == 'symmetric'
tiles.v_symmetric = tiles.v_symmetric.astype(int)
tiles['v_bottom_skewed'] = tiles.v_skew == 'bottom-skewed'
tiles.v_bottom_skewed = tiles.v_bottom_skewed.astype(int)
tiles['h_top_skewed'] = tiles.h_skew == 'right_skewed'
tiles.h_top_skewed = tiles.v_bottom_skewed.astype(int)
tiles['h_symmetric'] = tiles.h_skew == 'symmetric'
tiles.h_symmetric = tiles.v_bottom_skewed.astype(int)
tiles['h_bottom_skewed'] = tiles.h_skew == 'left-skewed'
tiles.h_bottom_skewed = tiles.v_bottom_skewed.astype(int)
# now drop non-continious columns
tiles.drop(['h_skew','v_skew','position'], 1, inplace=True)
# add feature: eucliden distance from center
center = (0,0)
tiles['c_diff'] = tiles.apply(lambda x: distance.euclidean(center, (x['pos_x'], x['pos_y'])), axis=1)
# obtain the referents
targs = pd.read_sql_query("SELECT * FROM referent", con)
targs.columns = ['episode_id', 'target']
# this should result in a dataframe of the target objects' features
query = '''
SELECT tiles.* FROM
targs 
INNER JOIN
tiles
ON targs.episode_id = tiles.episode_id
AND targs.target = tiles.id;
'''
targets = sqldf(query, locals())

# this should result in a datafrom of the non-target (i.e., distractor) features
query = '''
SELECT tiles.* FROM
tiles
LEFT OUTER JOIN
targs
ON targs.episode_id = tiles.episode_id
AND targs.target = tiles.id
WHERE targs.target is null;
'''
non_targets = sqldf(query, locals())

# obtain the referring expressions as utts
utts = pd.read_sql_query("SELECT * FROM hand", con)

# the result of this shuold be the words and corresponding object features for positive examples
query = '''
SELECT utts.word, utts.inc, targets.* FROM
targets 
INNER JOIN
utts
ON targets.episode_id = utts.episode_id
'''
positive = sqldf(query, locals())

# the result of this shuold be the words and corresponding object features for negative examples
query = '''
SELECT utts.word, utts.inc, non_targets.* FROM
non_targets 
INNER JOIN
utts
ON non_targets.episode_id = utts.episode_id
'''

num_eval = 100
negative = sqldf(query, locals())
negative.drop_duplicates(subset=['inc', 'episode_id', 'id'], inplace=True)
eids = set(positive.episode_id)
test_eids = set(random.sample(eids, num_eval))
train_eids = eids - test_eids
positive_train = positive[positive.episode_id.isin(train_eids)]
negative_train = negative[negative.episode_id.isin(train_eids)]
words = list(set(utts.word))
todrop = ['word', 'inc', 'episode_id', 'id']

# now we finally use our data for training
for word in words:
    pos_word_frame = positive_train[positive_train.word == word]
    pos_word_frame = np.array(pos_word_frame.drop(todrop, 1))
    neg_word_frame = negative_train[negative_train.word == word]
    neg_word_frame = np.array(neg_word_frame.drop(todrop, 1))
    # make sure the neg_word_frame is the same length as pos_word_frame
    neg_word_frame = random.sample(list(neg_word_frame), len(pos_word_frame))
    if len(neg_word_frame) == 0: continue
    # add positive and negative examples
    wac.add_multiple_observations(word, pos_word_frame, [1] * len(pos_word_frame))
    wac.add_multiple_observations(word, neg_word_frame, [0] * len(neg_word_frame))
    
print('training, persiting, and re-loading model...')
wac.train()
wac.persist_model()
wac.load_model()

print('performing evaluation...')
# test
# prepare eval data
utts = pd.read_sql_query("SELECT * FROM hand", con)
utts_eval = utts[utts.episode_id.isin(test_eids)]
utts_eval = utts_eval[utts_eval.word.isin(wac.vocab())] # remove words not in WAC
tiles_eval = tiles[tiles.episode_id.isin(test_eids)]
# get obj features
query = '''
SELECT utts_eval.word, utts_eval.inc, tiles_eval.* FROM
utts_eval
INNER JOIN
tiles_eval
ON utts_eval.episode_id = tiles_eval.episode_id
'''
eval_data = sqldf(query, locals())


corr = []
for eid in list(set(eval_data.episode_id)):
    wac.new_utt()
    episode = eval_data[eval_data.episode_id == eid]
    for inc in list(set(episode.inc)):
        increment = episode[episode.inc == inc]
        word = increment.word.iloc[0] # all the words in the increment are the same, so just get the first one
        intents = increment.id
        feats = np.array(increment.drop(todrop, 1))
        wac.add_increment(word, (intents, feats)) #the first four columns are word, inc, episode_id, and id
    corr.append(wac.get_predicted_intent()[0]==list(set(targs[targs.episode_id == eid].target))[0])
print('accuracy on test set of 100 random items:', sum(corr)/len(corr))

        

