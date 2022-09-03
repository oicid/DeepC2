import hashlib
import json
import logging
import re
import time
import requests
from datetime import datetime, timedelta
from threading import Timer
from urllib.parse import unquote
from code.eda import *

logging.basicConfig(level=logging.INFO, filename='botmaster.log', filemode='a',
                    format='%(asctime)s - %(levelname)s: %(message)s')
logging.Formatter.converter = time.gmtime
logging.info("[SYS] started.")

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
    r = requests.get(trends_url % woeid, headers=header, timeout=30)
    if r.status_code != 200:
        logging.exception(f"[TRN] not 200(t3): {r.text}")
        return
    res = r.text
    res = json.loads(res)
    # choose the last one above 10,000
    trends = [(_["tweet_volume"], _["query"]) for _ in res[0]["trends"] if _["tweet_volume"] is not None]
    trends.sort()
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


def record_trend(woeid, vec="0"):
    time_now = datetime.strftime(datetime.utcnow(), '%Y-%m-%d-%H-%M')
    keyword = get_trends(woeid)
    try:
        with open('new_cmd') as f:
            vec = f.read()
        vec = vec.strip()
    except FileNotFoundError as e:
        logging.info(f"{e}: no new command, using existing one.")
    logging.info(f"Command: {vec}")
    with open(f"trends-{woeid}.log", "a", encoding="utf-8") as f:
        f.write(f"{time_now}\t{keyword}\n")
    tweets = search_tweets(keyword, 1000)
    if tweets is None:
        logging.error(f"[REC] got None tweets. retry next hour.")
        loop((woeid, vec))
        return

    if isinstance(vec, list):
        # get avatars and compare (bot)
        logging.info(f"[REC] I am not a bot!")
    else:
        # log to generate many related tweets (botmaster)
        cleaned_tweets = clean_tweets(tweets)
        data0 = gen_eda(cleaned_tweets, 0.1, 50)
        data = list(set(data0))
        data += [_.upper() for _ in data0]
        data += [f"{_} ." for _ in data0]
        data += [f"{_} .." for _ in data0]
        data += [f"{_} ..." for _ in data0]
        data += [f"{_} !" for _ in data0]
        data += [f"{_} !!" for _ in data0]
        data += [f"{_} !!!" for _ in data0]

        calc = calc_hash(vec, data)
        if calc.get('result'):
            # sentences should contain trending keywords
            keyword = unquote(keyword)
            print(vec, keyword)
            target1 = calc['hash'][0]
            print(target1)
            for _ in calc[target1]:
                if keyword.lower() in _.lower():
                    print(_)
            target2 = calc['hash'][1]
            print(target2)
            for _ in calc[target2]:
                if keyword.lower() in _.lower():
                    print(_)
        else:
            print(f"{time_now}: hash collision not success")

    loop((woeid, vec))


def clean_tweets(tweets):
    dtl = list()
    for tweet in tweets:
        sentence = tweet.get("text")
        if len(sentence.split(' ')) < 10:
            continue
        sentence = re.sub(r'https://t.co/\w+', '', sentence)
        if sentence[-1] in '.!?~':
            sentence = sentence[:-1]
        if sentence.startswith('RT @'):
            dtl.append(sentence[sentence.index(":") + 2:].strip())
        else:
            dtl.append(sentence.strip())
    return dtl


def gen_eda(sentences, alpha, num_aug=9):
    '''generate sentences using EDA'''
    auged = list()
    for i, sentence in enumerate(sentences):
        aug_sentences = eda(sentence, alpha_sr=alpha, alpha_ri=alpha, alpha_rs=alpha, p_rd=alpha, num_aug=num_aug)
        for aug_sentence in aug_sentences:
            aug_sentence = aug_sentence.replace(' .', '.').replace(' ?', '?').replace(' !', '!')
            auged.append(aug_sentence)
    return auged


def calc_hash(ip, sentences):
    logging.info(f"ip: {ip}, length of data: {len(sentences)}")

    ip_list = ip.split('.')
    ip_str_sp = list()

    # convert decimal IP to two hex strings
    for _ in ip_list:
        _t = hex(int(_))[2:]
        if len(_t) == 1:
            _t = "0" + _t
        ip_str_sp.append(_t)

    target1 = f"{ip_str_sp[0]}{ip_str_sp[1]}"
    target2 = f"{ip_str_sp[2]}{ip_str_sp[3]}"

    logging.info(f"{ip}: {target1}, {target2}")

    hit = dict()
    hit["hash"] = [target1, target2]
    hit[target1] = list()
    hit[target2] = list()
    hit['result'] = False
    isA, isB = False, False
    for sp in sentences:
        sp = sp.strip()
        sha2 = hashlib.sha256(sp.encode()).hexdigest()
        if sha2.startswith(target1):
            isA = True
            hit[target1].append(sp)
            logging.info(f'1: {sp} {sha2}')
        elif sha2.startswith(target2):
            isB = True
            hit[target2].append(sp)
            logging.info(f'2: {sp} {sha2}')
    if isA and isB:
        hit['result'] = True
        logging.info("hit")
    return hit


def loop(_id, set_time=0):
    logging.info(f"[LOP] new schedule arranged. (set time: {set_time})")
    if set_time == 0:
        set_time = schedule()
    timer = Timer(set_time, record_trend, args=_id)
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
    vec = "127.0.0.1"
    record_trend(woeid, vec)
