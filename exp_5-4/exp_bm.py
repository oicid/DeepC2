import requests
import json
from datetime import datetime
import logging
import time

logging.basicConfig(level=logging.INFO, filename='bm_auto.log', filemode='a',
                    format='%(asctime)s - %(levelname)s: %(message)s')
logging.Formatter.converter = time.gmtime
logging.info("[SYS] started.")

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
    header = global_header
    header["authorization"] = f"Bearer {bearer_token}"
    time_now = datetime.strftime(datetime.utcnow(), '%Y-%m-%d-%H-%M')
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
    print(trends[0][1])


if __name__ == '__main__':
    woeid = 1582504
    get_trends(woeid)
