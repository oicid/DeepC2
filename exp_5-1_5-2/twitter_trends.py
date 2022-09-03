import requests
import re
import json
from datetime import datetime, timedelta
from threading import Timer
import logging
import time
from uuid import uuid4
# from tqdm import tqdm

logging.basicConfig(level=logging.INFO, filename='bt_auto.log', filemode='a',
                    format='%(asctime)s - %(levelname)s: %(message)s')
logging.Formatter.converter = time.gmtime
logging.info("[SYS] started.")

global_id = uuid4()
logging.info(f'[SYS] id: {global_id}')

global_header = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:75.0) Gecko/20100101 Firefox/75.0",
    "Accept-Language": "en-US,en;q=0.5"
}
bearer_token = "___REPLACE_THE_TOKEN_WITH_YOURS___"
trends_url = "https://api.twitter.com/1.1/trends/place.json?id=%d"
token_url = "https://api.twitter.com/1.1/guest/activate.json"
search_url_base = "https://api.twitter.com/1.1/search/tweets.json"
search_url = "https://api.twitter.com/1.1/search/tweets.json?q=%s&lang=en&count=100&&tweet_mode=extended"


def get_oauth(headers, guest_token=False):
    headers["authorization"] = f"Bearer {bearer_token}"
    if guest_token:
        r = requests.post(token_url, headers=headers, timeout=30)
        if r.status_code != 200:
            logging.exception(f"[OAUTH] not 200(3): {r.text}")
            return -1
        res = json.loads(r.text)
        logging.info(f"[OAUTH] guest token: {res}")
        headers['x-guest-token'] = res['guest_token']
    return headers


def deal_oauth_error(url, header_130, max_try=5):
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
    time_now = datetime.strftime(datetime.utcnow(), '%Y-%m-%d-%H')
    r = requests.get(trends_url % woeid, headers=header, timeout=30)
    if r.status_code != 200:
        logging.exception(f"[TRN] not 200(t3): {r.text}")
        return
    res = r.text
    res = json.loads(res)
    trends = [(_["tweet_volume"], _["query"]) for _ in res[0]["trends"] if _["tweet_volume"] is not None]
    trends.sort()
    with open(f"trends_{time_now}_{woeid}.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(trends))
    logging.info(f"[TRN] selected trend: {trends[0]}")
    return [trends[0][1], trends[2][1], trends[5][1], trends[8][1]]


def search_tweets(keyword, max_count=500):
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
    while len(tweets_id) < max_count:
        r = requests.get(f'{search_url_base}{res["search_metadata"]["next_results"]}&tweet_mode=extended', headers=header, timeout=30)
        if r.status_code != 200:
            logging.exception(f"[SCH] not 200(s2): {r.text}")
            r = deal_oauth_error(search_url_base + res["search_metadata"]["next_results"], header, max_try=3)
            if r == -1:
                break
        res = json.loads(r.text)
        tweets += res["statuses"]
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
    try:
        keyword = get_trends(woeid)
    except:
        loop((woeid, vec))
        return
    # log_trend(keyword)
    with open(f"trends-{woeid}.log", "a", encoding="utf-8") as f:
        f.write(f"{time_now}\t{keyword}\n")
    if isinstance(vec, list):
        time.sleep(60 * 5)
    for key_ in keyword:
        tweets = search_tweets(key_, 1000)
        if tweets is None:
            logging.error(f"[REC] got None tweets. retry next hour.")
            continue

        with open(f"new/{key_}_{woeid}_trends_{time_now}_1000.log", "w", encoding="utf-8") as f:
            for i, _ in enumerate(tweets):
                text = _['text'].replace('\n', ' ')
                f.write(f"{i}\t{text}\n")
        with open(f"long/{key_}_{woeid}_trends_{time_now}_1000.log", "w", encoding="utf-8") as f:
            f.write(json.dumps(tweets))

    loop((woeid, vec))


def loop(_id, set_time=0):
    logging.info(f"[LOP] new schedule arranged. (set time: {set_time})")
    if set_time == 0:
        set_time = schedule()
    timer = Timer(set_time, record_trend, args=_id)
    timer.start()


def schedule():
    now = datetime.now()
    next_hour = now + timedelta(minutes=30)
    delta = (next_hour - now).total_seconds()
    logging.info(f"[SCHE] now: {now}, next: {next_hour}, delta: {delta}")
    return delta


if __name__ == "__main__":
    record_trend(1582504)
    record_trend(2487956)   # San Francisco
    record_trend(1105779)   # Sydney - Sydney
    record_trend(44418)     # London
