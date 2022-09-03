import hashlib
import time
import json
import logging
import os
from code.eda import *

logging.basicConfig(level=logging.INFO, filename="24.log")


time1 = time.time()
with open('commands_ip') as f:
    ips = json.load(f)

mark = int(time1)
with open(f'hash_result_{mark}.csv', 'w', encoding='utf-8')as f:
    f.write("\ufeffnum_sen,num_ip,hit_ip,rate,hit_A,hit_B,time100,time_avg\n")
with open(f'sentences_log_{mark}.csv', 'w', encoding='utf-8') as f:
    f.write("\ufeffnum_lines,right_lines,blanks,time\n")

original_sentences_dir = "new"
cleaned_sentences_dir = "new_cleaned"
trends_dir = "new_augmented"


def clean_tweets():
    '''clean the tweets as in Sec. 3.4.2'''
    original_data = os.listdir(original_sentences_dir)
    for original_sentences in original_data:
        with open(f'{original_sentences_dir}/{original_sentences}', encoding='utf-8') as f:
            data = f.read()
        sentences = data.split('\n')
        sen_list = [_ for _ in sentences if len(_.split(' ')) > 10]
        for idx, _ in enumerate(sen_list):
            simple = _.split('\t')
            try:
                sentence = simple[1]
            except:
                continue
            sentence = re.sub(r'https://t.co/\w+', '', sentence)
            if sentence.startswith('RT @'):
                if ':' in sentence:
                    sen_list[idx] = f'{simple[0]}\t{sentence[sentence.index(":") + 2:].strip()}'
            else:
                sen_list[idx] = f'{simple[0]}\t{sentence.strip()}'

        with open(f'{cleaned_sentences_dir}/{original_sentences}', 'w', encoding='utf-8') as f:
            f.write('\n'.join(sen_list))


def generate_sentences(_alpha=0.1, _num_aug=50):
    '''generate new sentences using eda'''
    orig_data = os.listdir(cleaned_sentences_dir)
    for idx, f in enumerate(orig_data):
        train_orig = os.path.join(cleaned_sentences_dir, f)
        trend = f.split('_')[0]
        if '+' in trend:
            g22 = True
        else:
            g22 = False
        # decode url encoding
        trend = trend.replace('+', ' ').replace('%23', '#').replace('%22', '')
        gen_eda(train_orig, f"{trends_dir}/augmented_{idx}.txt", _alpha, num_aug=_num_aug, trend=trend.lower(), g22=g22)
        print(int(time.time() - time1))


def main():
    ''''efficiency test of hash collision, as in Sec. 5.2'''
    new_aug = os.listdir(trends_dir)
    for i in new_aug:
        with open(f'{trends_dir}/{i}', encoding='utf-8') as f:
            data0 = f.read()

        data0 = data0.split('\n')
        data0 = data0[:-1]
        data = list(set(data0))
        data += [_.upper() for _ in data0]
        calc_hash(ips, data)
        data += [f"{_} ." for _ in data0]
        data += [f"{_} .." for _ in data0]
        calc_hash(ips, data)
        data += [f"{_} ..." for _ in data0]
        data += [f"{_} !" for _ in data0]
        calc_hash(ips, data)
        data += [f"{_} !!" for _ in data0]
        data += [f"{_} !!!" for _ in data0]
        calc_hash(ips, data)


def calc_hash(ips, data):
    all_ip = 0
    hit_ip = 0
    time0 = time.time()
    time_avg = list()
    for ip in ips.keys():
        all_ip += 1
        target1, target2 = ips[ip][0], ips[ip][1]
        logging.info(f"{ip}, {target1}, {target2}")
        isA, isB = False, False
        time2 = time.time()
        for _ in data:
            sp = _.split('\t')
            if len(sp) != 2:
                logging.exception(sp)
                continue
            sha2 = hashlib.sha256(sp[1].strip().encode()).hexdigest()
            if sha2.startswith(target1):
                isA = True
                logging.info(f'1: {sp[0]} {sp[1]} {sha2}')
            elif sha2.startswith(target2):
                isB = True
                logging.info(f'2: {sp[0]} {sp[1]} {sha2}')
        time3 = time.time() - time2
        time_avg.append(time3)
        if isA and isB:
            hit_ip += 1
            logging.info("hit")
        logging.info(f"{time3}, {time.time() - time1}")
        logging.info("done\n")
    with open(f'hash_result_{mark}.csv', 'a', encoding='utf-8')as f:
        f.write(f"{len(data)},{all_ip},{hit_ip},{hit_ip / all_ip},0,0,{time.time()-time0},{sum(time_avg) / len(time_avg)}\n")
    print(f"data: {len(data)}, all: {all_ip}, hit: {hit_ip}, rate: {hit_ip / all_ip}")


def gen_eda(train_orig, output_file, alpha, num_aug=9, trend="T", g22=False):
    writer = open(output_file, 'w', encoding='utf-8')
    lines = open(train_orig, 'r', encoding='utf-8').readlines()
    num_lines = 0
    right_lines = 0
    time_0 = time.time()

    for i, line in enumerate(lines):
        parts = line[:-1].split('\t')
        try:
            label = parts[0]
            sentence = parts[1]
            aug_sentences = eda(sentence, alpha_sr=alpha, alpha_ri=alpha, alpha_rs=alpha, p_rd=alpha, num_aug=num_aug)
            for aug_sentence in aug_sentences:
                num_lines += 1
                if trend in aug_sentence.lower():
                    right_lines += 1
                aug_sentence = aug_sentence.replace(' .', '.').replace(' ?', '?').replace(' !', '!')
                writer.write(label + "\t" + aug_sentence + '\n')
        except Exception as e:
            print(f"{e} | {train_orig} | {line}")

    writer.close()
    time_1 = time.time() - time_0
    with open(f'sentences_log_{mark}.csv', 'a', encoding='utf-8') as f:
        f.write(f"{num_lines},{right_lines},{g22},{time_1}\n")
    print("generated augmented sentences with eda for " + train_orig + " to " + output_file + " with num_aug=" + str(
        num_aug))


if __name__ == '__main__':
    clean_tweets()
    generate_sentences()
    main()
