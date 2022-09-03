import requests
import json
from datetime import datetime, timedelta
from threading import Timer
import logging
import time
from uuid import uuid4

logging.basicConfig(level=logging.INFO, filename='bt_auto.log', filemode='a',
                    format='%(asctime)s - %(levelname)s: %(message)s')
logging.Formatter.converter = time.gmtime
logging.info("[SYS] started.")

global_header = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:79.0) Gecko/20100101 Firefox/79.0",
    "Accept-Language": "en-US,en;q=0.5"
}
bearer_token = "___REPLACE_THE_TOKEN_WITH_YOURS___"
token_url = "https://api.twitter.com/1.1/guest/activate.json"
search_url_base = "https://api.twitter.com/1.1/search/tweets.json"
search_url = "https://api.twitter.com/1.1/search/tweets.json?q=%s&lang=en&count=100&&tweet_mode=extended"

waiting_time = [5, 10, 20, 30, 45, 60, 90, 120, 150, 180]
waiting_time = [_ * 60 for _ in waiting_time]

# botmaster's account
crawled_id = "1469788801"
max_search_count = 5000


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


def get_trends():
    '''To simplify, we choose a trending topic and publish it on a server,
       and the bots get it from the server. If we don't post tweets after 
       choosing the topic, we set the topic as `__SLEEP__`.'''
    local_id = uuid4()
    r = requests.get('http://v4bs.in0.in/assets/topic')
    if r.status_code != 200:
        logging.exception(f"[TRN] not 200(t3): {r.text}")
        loop()
        return
    logging.info(f"[TRN] selected trend: {r.text}")
    if "__SLEEP__" not in r.text:
        for t in waiting_time:
            logging.info(f"[TRN] Searching scheduled: {t / 60}min")
            Timer(t, search_tweets, args=(local_id, r.text, t)).start()
    loop()


def search_tweets(task_id, keyword, mint, max_count=max_search_count):
    header = get_oauth(global_header, True)
    res = None
    current_count = 0
    time_now = datetime.strftime(datetime.utcnow(), '%Y-%m-%d-%H-%M')
    while current_count < max_count:
        if res:
            url = f'{search_url_base}{res["search_metadata"]["next_results"]}&tweet_mode=extended'
        else:
            url = search_url % keyword
        r = requests.get(url, headers=header, timeout=30)
        if r.status_code != 200:
            logging.exception(f"[SCH] not 200(s2): {r.text}")
            r = deal_oauth_error(url, header, max_try=3)
            if r == -1:
                current_count -= 10000
                break
        res = json.loads(r.text)
        tweets = res["statuses"]
        for _ in tweets:
            current_count += 1
            if _.get("user").get("id_str") == crawled_id:
                current_count += 10000
                break
    logging.info(f"[SCH] result: {current_count}")
    with open("result_record.csv", "a", encoding="utf-8") as w:
        if current_count > 10000:
            w.write(f'{task_id},{time_now},{mint},true,{current_count - 10000},{_.get("id_str")},{_.get("created_at")},'
                    f'{_.get("user").get("id_str")},{_.get("user").get("screen_name")},{_.get("full_text")},'
                    f'{_.get("user").get("profile_image_url_https")}\n')
        elif current_count > 0:
            w.write(f'{task_id},{time_now},{mint},false,{current_count}\n')
    return current_count


def loop(set_time=0):
    logging.info(f"[LOP] new schedule arranged. (set time: {set_time})")
    if set_time == 0:
        set_time = schedule()
    timer = Timer(set_time, get_trends)
    timer.start()


def schedule():
    now = datetime.now()
    if now.minute > 28:
        next_time = now + timedelta(hours=1)
        next_time = next_time.replace(minute=0, second=0, microsecond=0)
    else:
        next_time = now.replace(minute=30, second=0, microsecond=0)
    delta = (next_time - now).total_seconds()
    logging.info(f"[SCHE] now: {now}, next: {next_time}, delta: {delta}")
    return delta


if __name__ == "__main__":
    loop()
