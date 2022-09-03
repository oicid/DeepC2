# DeepC2: AI-Powered Covert Command and Control on OSNs

This is the artifact of the DeepC2 paper accepted at ICICS 2022.


## Contributions in the paper

Command and control (C&C) is important in an attack. Currently, some attackers 
use online social networks (OSNs) as the C&C channel to publish the commands. 
However, the attackers face two problems. First, the way for bots to find the 
botmaster (addressing) is hardcoded in the bot program. The defenders will 
know how to find the botmaster if a bot is reverse-engineered. Then the 
botmaster will be exposed before the commands are published. Second, the 
commands are abnormal contents on OSNs, which may trigger restrictions on the 
attacker's accounts, making the bots cannot get the commands. This paper 
proposes an AI-powered covert C&C channel that can resolve the above two 
problems. 

For the first one, we use a neural network (NN) for addressing. The neural 
network is poorly explainable. Defenders cannot predict the attacker's 
accounts before the commands are published. We use the NN model to recognize 
the attacker's avatars. The avatars are extracted as feature vectors, and the 
bots are published with the vectors. The bots will calculate the distances 
between the Twitter user's avatars and the vectors to find the botmaster. The 
bots and their botmaster meet under a trending topic. They choose a topic 
synchronously. Then the botmaster posts tweets under the topic, and the bots 
will crawl them. If the bots find the botmaster, they can get the commands 
from the tweets. 

For the second one, we use data augmentation and hash collision to generate 
contextual and readable contents that contain the commands. After choosing a 
topic, the attacker will crawl the tweets under the topic. Then, the attacker 
will get numerous new tweets using data augmentation methods. The 
tweets are used to calculate hashes. The attacker split a command into 
two-byte chunks. The corresponding tweet will be selected if a chunk is equal 
to the first two bytes of one hash value. If all chunks of a command collide, 
the attacker posts the tweets to the topic. After the bots get the 
tweets, they need to calculate the hashes to get the commands.

It is worth noting that we do not aim to inspire malware authors to write 
more efficient malware but to demonstrate an upward threat. Therefore, we will 
share the method and code but choose the semi-open source for only security 
researchers and vendors. Below are the instructions for the artifact.


## Structure of the artifact

```
|
|-- avatars       // avatars and feature vectors
|    |
|    |--- images             // folder. avatars used by the botmaster
|    |--- gen_test_vectors.py  // get feature vectors for botmaster and bots
|    |--- LinearNet.py       // structure of the Siamese NN
|    |--- model.pt           // NN model in the experiment
|    |--- Net_128_4conv.py   // strucute of the CNN in Siamese NN
|    |
|-- bot           // code for the bot
|    |
|    |--- bot.py             // main implementation of the bot
|    |--- LinearNet.py       // structure of the Siamese NN
|    |--- model.pt           // NN model in the experiment
|    |--- Net_128_4conv.py   // strucute of the CNN in Siamese NN
|    |--- vectors_bt.dat     // feature vectors from the botmaster avatars
|    |
|-- botmaster     // code for botmaster
|    |
|    |--- code               // folder. open source code for data augmentation
|    |--- botmaster.py       // code for the botmaster when publishing commands
|    |--- new_cmd            // New command to publish
|    |--- vectors_bm.dat     // features vectors with labels from avatars.
|    |
|-- exp_5-1_5-2   // evaluation of tweets generation and hash collision on Sec. 5.1 and 5.2
|    |
|    |--- code               // folder. open source code for data augmentation
|    |--- bm_nohup.log       // printed output in the experiment
|    |--- commands_ip        // commands (ip) used for hash collision
|    |--- hash_result_1591126641.csv    // original hash collision result during exps
|    |--- sentences_log_1591224113.csv  // original tweets generation result during exps
|    |--- tweets_hash.py     // main, generate tweets and hash collisions
|    |--- twitter_trends.py  // get trends and tweets for the exp
|    |
|-- exp_5-3       // evaluation of avatar recognition efficiency on Sec. 5.3
|    |
|    |--- image_dist_test.py // extract feature vectors and calculate distances
|    |--- LinearNet.py       // same as bot/LinearNet.py
|    |--- logs_1591390313.csv // test result
|    |--- logs_1624594430.csv // test result
|    |--- model.pt           // same as bot/model.pt
|    |--- Net_128_4conv.py   // same as bot/Net_128_4conv.py
|    |--- vectors_bm.dat     // same as botmaster/vectors_bm.dat
|    |
|-- exp_5-4       // evaluation of tweet crawl time and quantity on Sec. 5.4
|    |
|    |--- exp_bm.py          // choose a trending topic for botmaster
|    |--- exp_bot.py         // bots wait and crawl the tweets
|    |--- parsed_result.csv  // test result
|    |
|-- train_model   // code for train a NN model
|    |
|    |--- contrastive.py     // Contrastive loss
|    |--- LinearNet.py       // same as bot/LinearNet.py
|    |--- LoadDataset.py     // Loading dataset
|    |--- main.py            // main function for training the model
|    |--- Net_128_4conv.py   // same as bot/Net_128_4conv.py
|    |--- test_2260.txt      // test dataset label
|    |--- train_2260.txt     // training dataset label
|    |--- val_3138.txt       // validation dataset label
|    |
|-- Readme        // Readme file
|-- requirements.txt   // python package requirements for running the code
|___________________________
```

## Setup

The code was implemented with `Python 3.6.9` and `PyTorch 1.5`. In the 
experiment in the paper, the code was run on `Ubuntu 18.04 x64` virtual server 
with 1 GB ROM and 1 vCPU. The suppliers are [Digital Ocean](https://digitalocean.com/) 
and [Alibaba Cloud](https://www.aliyun.com/). The latest test shows that the 
code is compatible with `Python 3.10.4` and `PyTorch 1.12.1` on `Ubuntu 22.04 x64`.

### Get a token

We use [Twitter API](https://developer.twitter.com/) in the code. Please 
replace the `bearer_token` in the code with yours. 

```
bearer_token = "___REPLACE_THE_TOKEN_WITH_YOURS___"
```

Files to be replaced:

```
bot/bot.py
botmaster/botmaster.py
exp_5-1_5-2/twitter_trends.py
exp_5-4/exp_bm.py
exp_5-4/exp_bot.py
```

### Install the requirements

```
python3 -m pip install -r requirements.txt
```

### Run the bot

```
cd bot
python3 bot.py
```

Check `bot.log` for the latest status of the bot. Currently, the bot crawls 
the command once an hour, and the user can change it in `schedule` on line 276. 
In the implementation, bots synchronously select the vector with the botmaster. 
The first three values of the vector are logged in `bot.log` and printed on the 
terminal. The botmaster should set the corresponding picture as the avatar 
before the bots start crawling the tweets. For example, the first three 
values in the vector are `0.05675289034843445, 0.08973734080791473, 0.13668733835220337`. 
The botmaster checks `botmaster/vectors_bm.dat` and finds the corresponding 
picture is `test_avatar_8.jpg`. So the botmaster changes the Twitter avatar to 
`avatars/images/test_avatar_8.jpg`. If the bot gets the command, the log file 
will record it as follows:

```
[TRN] selected trend: (11257, 'Foden')
[SCH] returned tweets: 900
[REC] start compare: 1662235510.8540785
[CMP] !!! link: https://pbs.twimg.com/profile_images/1566154056199856128/vqL2KohU_400x400.jpg
[REC] end compare: 1662235513.3028378
[REC] found botmaster: [(1566154496853495808, 'City Xtra spbajko Mahrez amp Foden needs to improve their a finishing !!'), (1566154628374282240, 'Here we go again, we a good game, but end up drawing to a banger and now time on TL this season the Foden slander')]
[REC] command: 127.0.0.1
[REC] command is an IP.
```

### Run the botmaster

The botmaster uses [EDA](https://doi.org/10.18653/v1/D19-1670) to generate 
tweets for hash collision. Inside of `botmaster/code` is what the botmaster 
needs. However, for the first time the user needs some resources to run the 
code:

```
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
```

Then you can run the botmaster. 

```
cd botmaster
python3 botmaster.py
```

Note that in the implementation, we use an IP address as the command. So the 
user needs to modify the code to be capable of other types of commands. In 
the experiments, we use a simple way to change the command: write the new 
command to `botmaster/new_cmd`. 

The trending topics are logged in `trends-{woeid}.log`. After choosing the 
topic, the botmaster crawls the tweets, generates new sentences, and starts 
hash collision. The results are printed as follows:

```
// command and the trending topic
127.0.0.1 Foden

// first chunk and the sentences
7f00
Reborn Priv yh he defo got his subs wrong now why did he bring on a defender and there were 15 hokkianese get out to play and Alvarez could have started over foden tbh ...
city tactics Bang on.. has two assists in golden and coming into tonight. No one mentioning missing and Mahrez gundogan consecutive chances games focusing on foden only He Weird. Obviously fans need a new scapegoat now sterling has gone !
. Gordon... man brought right Ake when we were chasing a game lmfao. Alvarez was literally and there and Grealish. This guy is tapped man Jack Bun this inverted rubbish, I at to see Cancelo need RB on Gomez on the left. Foden should hold bench for Joseph !
City Xtra spbajko Mahrez amp Foden needs to improve their a finishing !!

// second chunk and the sentences
0001
Here we go again, we a good game, but end up drawing to a banger and now time on TL this season the Foden slander
MOYODRE IODIN LEGIT DON’T SEE THAT MUCH OF A DIFFERENCE AND IODIN RATE FODEN AND IODIN ALSO AGREE IODIN WANT MORE AGGRESSION FROM SAKA. BUT ENJOIN THEY’RE NOT COMPARABLE IS
AlexanderKing4L bayour20 Officiating today was shambolic, that Foden blown wasn’t handball lol !!!
```

The botmaster should select a sentence from each chunk and post them on Twitter.

![Post on Twitter](.image/post_twitter.png)


## Code analysis

### Bot

The main implementation of a bot is in `bot/bot.py`. Important functions are 
illustrated as follows.


| line | function       | paper sec. |               detail                   |
|----|----|----|----|
| 28   | bearer_token   |      -     | Bearer token for authorization. Please replace it with yours.                 |
| 70   | get_trends     | Sec. 3.3.3 | Choosing a trending topic in a specified area using `woeid`.          |
| 90   | search_tweets  |      -     | Crawl tweets with the trending topic.  |
| 136  | record_trend   |      -     | Call `get_trends` and `search_trends`, and then `compare`, `get_new_vector`,  and `get_command`                      |
| 180  | compare        |Sec. 3.3.1, Sec. 3.3.2 | Calculate distances to find botmaster  |
| 218  | get_new_vector |      -     | Get a new vector from `vectors_bt.dat` and delete the used one.               |
| 231  | get_command    |      -     | Parse a command from hash values.  |
| 276  | schedule       |     -     | Schedule the crawling tasks |


### Botmaster

The main implementation of the botmaster is in `botmaster/botmaster.py`. Most 
of the codes are the same as the bots. Other important functions are 
illustrated as follows.


| line | function       | paper sec. |               detail                   |
|----|----|----|----|
| 125  | record_trend   |      -     | Call `get_trends` and `search_trends`, and then `clean_tweets`, `gen_eda`, get more tweets, and `calc_hash`       |
| 180  | clean_tweets   | Sec. 3.4.2 | Clean the crawled tweets.              |
| 196  | gen_eda        | Sec. 3.4.2 | Generate new sentences using EDA.      |
| 207  | calc_hash      | Sec. 3.4.1 | Hash collision.       |


### Settings

__Trending topics.__ The trending topics are from Johannesburg. If the user 
wants to choose other cities, please change the woeid to different values. 
(Online query for [WOEID](https://www.woeids.com/) )

__Sync.__ The botmaster and bots sync every hour. It can be changed in `schedule`.
The bots wait five minutes after choosing the topic. It can be changed in Line 
142 in `record_trend`.

__Feature vectors.__ Before running the bots, botmaster should extract the 
feature vectors from a batch of pictures. Run `avatars/gen_test_vectors.py`.

__Pay attention to the paths in the code.__


### For training the model

According to Twitter [Terms of Services](https://twitter.com/tos) (ToS), we 
deleted the data after training the NN models. The user can still get the 
avatars by crawling on Twitter. However, the user should not violate Twitter 
ToS when crawling. 

The `txt` files are the labels used during training, and they are not the same 
as the ones in the experiments because they are anonymized. The user can label 
their data as these files.

### Workflow

1) Train a neural network model. If the user does not want to train the model, 
   a trained model is available in `bot/model.pt`.
2) Run the botmaster and bot. They will choose a trending topic according to 
   the settings.
3) Post tweets. After choosing a trending topic, data augmentation and hash 
   collision will be performed in `botmaster/botmaster.py`, and suitable 
   tweets will be selected soon. Please do not change the selected tweets and 
   post them on Twitter assp.
3) Change avatar. Don't forget to change the avatar on Twitter before the bots 
   starting crawling the tweets.
4) Wait for the bots to find the botmaster and parse the commands.


## Evaluation

This part corresponds to Section 5 of the paper. In the experiment, the code 
for evaluation was also run on Digital Ocean Ubuntu 18.04 x64 virtual servers 
with 1 GB ROM and 1 vCPU.

### Tweets generation and hash collision

Codes for evaluating tweets generation and hash collision are in dir 
`exp_5-1_5-2`.

We run `twitter_trends.py` first to get some trending topics and tweets for 
the evaluation, which corresponds to Sec. 5.1. Then we run `tweets_hash.py` to 
clean the tweets, generate new sentences, and calculate their hashes. Original 
experiment results are also available in `sentences_log_1591224113.csv` and 
`hash_result_1591126641.csv`.

Note that we deleted data from Twitter after the experiments, so the user 
should crawl the tweets used in the evaluation using `twitter_trends.py`. 
Also, pay attention to the paths in the code. 

#### Fields in `sentences_log_1591224113.csv`

This result corresponds to Fig. 7.

| field | description |
| --- | --- |
| num_lines | number of sentences generated |
| right_lines | number of sentences containing exact trending words |
| blanks | whether it is a multi-word topic |
| time | time cost for generating sentences |

#### Fields in `hash_result_1591126641.csv`

This result corresponds to Fig. 8.

| field | description |
| --- | --- |
| num_sen | number of sentences used in the collision task |
| num_ip | number of commands (ip) |
| hit_ip | number of successful collisions for the commands |
| rate | hash collision hit rate |
| hit_A | number of successful collisions for the first part of ip |
| hit_B | number of successful collisions for the second part of ip |
| time100 | total time cost |
| time_avg | average time cost for each command |

### Avatar Recognition

Codes for evaluating the efficiency of avatar recognition are in dir 
`exp_5-3`. This part corresponds to Sec. 5.3.

The main functions are in `image_dist_test.py`. We prepared 1,000 avatars in 
size 400x400 for this experiment. However, due to Twitter ToS, we could not 
enclose them here. The user should crawl it before running the code. The 
original results are also available in `logs_1591390313.csv` and 
`logs_1624594430.csv`.

#### Fields in the log files

| field | description |
| --- | --- |
| filename | botmaster's avatar (available in `avatars`) |
| time | time cost for extracting the 1,000 feature vectors and calculating the distances |

### Tweets crawling number and time

Codes for determining the waiting time and tweet crawling number are in dir 
`exp_5-4`. This part corresponds to Sec. 5.4.

For testing needs, we simplified the data processing process. The botmaster 
chooses a trending topic and then publishes it on a server. The botmaster does 
not need to perform tweets generation and hash collision in this evaluation. 
Instead, the botmaster just needs to post a tweet with the trending topic. The 
bots get the topic from the server. Then they wait for different times and try 
to find the botmaster's account. They will log the waiting time and crawl 
tweet numbers when the botmaster is found. If the botmaster wants to pause the 
evaluation, just change the topic in the server to `__SLEEP__`, and the bots 
will not crawl on Twitter.

`exp_bm.py` is the code for the attacker to select a trending topic. 
`exp_bot.py` is for the bots to fetch the topic and find the botmaster. 
`parsed_result.csv` is a test result in this evaluation, corresponding to 
`search_tweets` in `exp_bot.py` and Fig. 9 in the paper. The first column is a 
task id, representing a crawling task. The rest columns are the numbers of 
crawled tweets when the botmaster is found after waiting for 5, 10, 20, 30, 
45, 60, 90, 120, 150, and 180 minutes.


## Citation

```
@InProceedings{WangLCYLWL22,
   author    = "Wang, Zhi and Liu, Chaoge and Cui, Xiang and Yin, Jie and Liu, Jiaxi and Wu, Di and Liu, Qixu",
   editor    = "Alcaraz, Cristina and Chen, Liqun and Li, Shujun and Samarati, Pierangela",
   title     = "{DeepC2}: AI-Powered Covert Command and Control on OSNs",
   booktitle = {Information and Communications Security - 24th International Conference,
                {ICICS} 2022, Canterbury, UK, September 5-8, 2022, Proceedings},
   year      = "2022",
   publisher = {Springer},
   address   = {Cham},
   series    = {Lecture Notes in Computer Science},
   volume    = {13407},
   pages     = {394--414},
   doi       = {10.1007/978-3-031-15777-6\_22},
   isbn      = {978-3-031-15777-6}
}
```
