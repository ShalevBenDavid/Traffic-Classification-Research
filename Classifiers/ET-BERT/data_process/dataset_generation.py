#!/usr/bin/python3
# -*- coding:utf-8 -*-

import os
import sys
import copy
import json
import tqdm
import shutil
import pickle
import random
import binascii
import operator
import numpy as np
import pandas as pd
import scapy.all as scapy
from functools import reduce
from flowcontainer.extractor import extract

random.seed(40)

word_dir = "C:\\Users\\code\\Traffic-Classification-Research\\ET-BERT\\corpora"
word_name = "encrypted_burst.txt"


def get_first_packet_source_ip(pcap_file):
    try:
        packets = scapy.rdpcap(pcap_file)
        for packet in packets:
            if scapy.IP in packet:
                return packet[scapy.IP].src
    except Exception as e:
        print(f"Error reading {pcap_file}: {e}")
    return None


def convert_pcapng_2_pcap(pcapng_path, pcapng_file, output_path):
    pcap_file = output_path + pcapng_file.replace('pcapng', 'pcap')
    cmd = "\"/media/Data/Datasets/QUIC PCAP 2/split-Pcaps/SplitCap.exe\" -F pcap '%s' '%s' -p 1018"
    command = cmd % (pcapng_path + pcapng_file, pcap_file)
    os.system(command)
    return 0


def split_cap(pcap_path, pcap_file, pcap_name, pcap_label='', dataset_level='flow'):
    if not os.path.exists(pcap_path + "/splitcap"):
        os.mkdir(pcap_path + "/splitcap")
    if pcap_label != '':
        if not os.path.exists(pcap_path + "/splitcap/" + pcap_label):
            os.mkdir(pcap_path + "/splitcap/" + pcap_label)
        if not os.path.exists(pcap_path + "/splitcap/" + pcap_label + "/" + pcap_name):
            os.mkdir(pcap_path + "/splitcap/" + pcap_label + "/" + pcap_name)

        output_path = pcap_path + "/splitcap/" + pcap_label + "/" + pcap_name
    else:
        if not os.path.exists(pcap_path + "/splitcap/" + pcap_name):
            os.mkdir(pcap_path + "/splitcap/" + pcap_name)
        output_path = pcap_path + "/splitcap/" + pcap_name

    if dataset_level == 'flow':
        cmd = "\"/media/Data/Datasets/QUIC PCAP 2/split-Pcaps/SplitCap.exe\" -r '%s' -s session -o '%s' -p 1018" % (
            pcap_file, output_path)
    elif dataset_level == 'packet':
        cmd = "\"/media/Data/Datasets/QUIC PCAP 2/split-Pcaps/SplitCap.exe\" -r '%s' -s packets 1 -o '%s' -p 1018" % (
            pcap_file, output_path)

    os.system(cmd)
    return output_path

def cut(obj, sec):
    result = [obj[i:i + sec] for i in range(0, len(obj), sec)]
    try:
        remanent_count = len(result[0]) % 4
    except Exception as e:
        remanent_count = 0
        print("cut datagram error!")
    if remanent_count == 0:
        pass
    else:
        result = [obj[i:i + sec + remanent_count] for i in range(0, len(obj), sec + remanent_count)]
    return result


def bigram_generation(packet_datagram, packet_len=64, flag=True):
    result = ''
    generated_datagram = cut(packet_datagram, 1)
    token_count = 0
    for sub_string_index in range(len(generated_datagram)):
        if sub_string_index != (len(generated_datagram) - 1):
            token_count += 1
            if token_count > packet_len:
                break
            else:
                merge_word_bigram = generated_datagram[sub_string_index] + generated_datagram[sub_string_index + 1]
        else:
            break
        result += merge_word_bigram
        result += ' '

    return result


def get_burst_feature(label_pcap):
    packets = scapy.rdpcap(label_pcap)

    packet_direction = []
    packet_length = []
    packet_time = []

    for packet in packets:
        # Length
        packet_length.append(len(packet))

        # Time
        packet_time.append(packet.time)

        # Direction (assuming positive length means outgoing, negative for incoming)
        packet_direction.append(1 if len(packet) > 0 else -1)

    # Format these lists as needed for your analysis
    return packet_length, packet_time, packet_direction


def get_feature_packet(label_pcap, payload_len):
    feature_data = []

    packets = scapy.rdpcap(label_pcap)
    packet_data_string = ''

    for packet in packets:
        packet_data = packet.copy()
        data = (binascii.hexlify(bytes(packet_data)))

        packet_string = data.decode()

        new_packet_string = packet_string[76:]

        packet_data_string += bigram_generation(new_packet_string, packet_len=payload_len, flag=True)

    feature_data.append(packet_data_string)
    return feature_data


def get_feature_flow(label_pcap, host_ip):
    packets = scapy.rdpcap(label_pcap)
    packet_length = []
    packet_time = []
    packet_direction = []  # 1 for outgoing, -1 for incoming

    for packet in packets:
        packet_length.append(len(packet))
        packet_time.append(packet.time)

        # Check if IP layer is present in the packet
        if scapy.IP in packet:
            if packet[scapy.IP].src == host_ip:
                packet_direction.append(1)  # Outgoing
            elif packet[scapy.IP].dst == host_ip:
                packet_direction.append(-1)  # Incoming
            else:
                packet_direction.append(0)  # Neither incoming nor outgoing
        else:
            packet_direction.append(0)  # IP layer not present

    return packet_length, packet_time, packet_direction

def generation(pcap_path, samples, dataset_save_path="C:\\Users\\code\\Traffic-Classification-Research\\Datasets\\NetFlow-QUIC1", dataset_level="flow"):
    dataset_pickle_path = os.path.join(dataset_save_path, "traffic_dict.pkl")  # Define the pickle path
    features = ['length', 'time', 'direction']

    if os.path.exists(dataset_pickle_path):
        print(f"The pcap file at {pcap_path} is already processed.")
        with open(dataset_pickle_path, "rb") as f:
            dataset = pickle.load(f)
            X, Y = obtain_data(pcap_path, samples, features, dataset_save_path, data=dataset)


        return X, Y

    dataset = {}
    label_name_list = []
    session_pcap_path = {}
    label_file_counts = {}  # Dictionary to keep track of file counts per label
    print(pcap_path)
    for parent, dirs, files in os.walk(pcap_path):
        if not label_name_list:
            label_name_list.extend(dirs)
            print("Labels found:", label_name_list)

        for dir in label_name_list:
            session_pcap_path[dir] = os.path.join(pcap_path, dir)
            print(f"Constructed path for label '{dir}': {session_pcap_path[dir]}")
        break

    label_id = {label_name_list[index]: index for index in range(len(label_name_list))}

    r_file_record = []
    print("\nBegin to generate features.")

    for key in tqdm.tqdm(session_pcap_path.keys()):
        dataset[label_id[key]] = {
            "samples": 0,
            "length": {},
            "time": {},
            "direction": {}
        }
   
        target_all_files = [x[0] + "/" + y for x in [(p, f) for p, d, f in os.walk(session_pcap_path[key])] for y in
                            x[1]]
        print(f"Files found for label {key}: {len(target_all_files)}")
        label_file_counts[key] = len(target_all_files)
        num_samples = min(samples, label_file_counts[key])
        r_files = random.sample(target_all_files, num_samples) if num_samples > 0 else []

        for r_f in r_files:
            host_ip = get_first_packet_source_ip(r_f)
            if host_ip is None:
                continue  # Skip this file if the IP couldn't be read

            if dataset_level == "flow":
                packet_length, packet_time, packet_direction = get_feature_flow(r_f,host_ip)
            elif dataset_level == "packet":
                packet_length, packet_time, packet_direction = get_feature_packet(r_f)

            if packet_length == -1:  # Check for error condition
                continue
            r_file_record.append(r_f)
            sample_index = str(dataset[label_id[key]]["samples"] + 1)
            dataset[label_id[key]]["samples"] += 1
            dataset[label_id[key]]["length"][sample_index] = packet_length
            dataset[label_id[key]]["time"][sample_index] = packet_time
            dataset[label_id[key]]["direction"][sample_index] = packet_direction

    print(label_name_list)
    print("here1")
    all_data_number = 0
    for label in label_name_list:
        print("%s\t%s\t%d" % (label_id[label], label, dataset[label_id[label]]["samples"]))
        all_data_number += dataset[label_id[label]]["samples"]
    print("all\t%d" % all_data_number)

    with open(dataset_pickle_path, "wb") as f:
        pickle.dump(dataset, f)

    # Prepare X, Y using the generated dataset
    features = ['length', 'time', 'direction']  # This line already exists in your code
    # Modify the function call to include the 'features' argument
    print("here\n")
    X, Y = obtain_data(pcap_path, samples, features, dataset_save_path, data=dataset)


    # Print the file counts after processing all labels
    for label, count in label_file_counts.items():
        print(f"Label: {label}, Available Samples: {count}")

    return X, Y


def read_data_from_json(json_data, features, samples):
    X, Y = [], []
    ablation_flag = 0
    for feature_index in range(len(features)):
        x = []
        label_count = 0
        for label in json_data.keys():
            sample_num = json_data[label]["samples"]
            if X == []:
                if not ablation_flag:
                    y = [label] * sample_num
                    Y.append(y)
                else:
                    if sample_num > 1500:
                        y = [label] * 1500
                    else:
                        y = [label] * sample_num
                    Y.append(y)
            if samples[label_count] < sample_num:
                x_label = []
                for sample_index in random.sample(list(json_data[label][features[feature_index]].keys()), 1500):
                    x_label.append(json_data[label][features[feature_index]][sample_index])
                x.append(x_label)
            else:
                x_label = []
                for sample_index in json_data[label][features[feature_index]].keys():
                    x_label.append(json_data[label][features[feature_index]][sample_index])
                x.append(x_label)
            label_count += 1
        X.append(x)
    return X, Y


def obtain_data(pcap_path, samples, features, dataset_save_path, data=None):
    if data:
        # Process data as before but using the 'data' parameter
        X, Y = read_data(data, features, samples)
    else:
        print("Reading dataset from pickle file.")
        with open(dataset_save_path + "/traffic_dict.pkl", "rb") as f:
            dataset = pickle.load(f)
        X, Y = read_data(dataset, features, samples)

    for index in range(len(X)):
        if len(X[index]) != len(Y):
            print("data and labels are not properly associated.")
            print("x:%s\ty:%s" % (len(X[index]), len(Y)))
            return -1
    return X, Y


def combine_dataset_json():
    dataset_name = "I:\\traffic_pcap\\splitcap\\dataset-"
    # dataset vocab
    dataset = {}
    # progress
    progress_num = 8
    for i in range(progress_num):
        dataset_file = dataset_name + str(i) + ".json"
        with open(dataset_file, "r") as f:
            json_data = json.load(f)
        for key in json_data.keys():
            if i > 1:
                new_key = int(key) + 9 * 1 + 6 * (i - 1)
            else:
                new_key = int(key) + 9 * i
            print(new_key)
            if new_key not in dataset.keys():
                dataset[new_key] = json_data[key]
    with open("I:\\traffic_pcap\\splitcap\\dataset.json", "w") as f:
        json.dump(dataset, fp=f, ensure_ascii=False, indent=4)
    return 0


def pretrain_dataset_generation(pcap_path):
    output_split_path = "/media/Data/Datasets/QUIC PCAP2/"
    pcap_output_path = "/media/Data/Datasets/QUIC PCAP2/split-Pcaps"

    if not os.listdir(pcap_output_path):
        print("Begin to convert pcapng to pcap.")
        for _parent, _dirs, files in os.walk(pcap_path):
            for file in files:
                if 'pcapng' in file:
                    # print(_parent + file)
                    convert_pcapng_2_pcap(_parent, file, pcap_output_path)
                else:
                    shutil.copy(_parent + "/" + file, pcap_output_path + file)

    if not os.path.exists(output_split_path + "splitcap"):
        print("Begin to split pcap as session flows.")

        for _p, _d, files in os.walk(pcap_output_path):
            for file in files:
                split_cap(output_split_path, _p + file, file)
    print("Begin to generate burst dataset.")
    # burst sample
    for _p, _d, files in os.walk(output_split_path + "splitcap"):
        for file in files:
            get_burst_feature(_p + "/" + file, payload_len=64)
    return 0


def size_format(size):
    # 'KB'
    file_size = '%.3f' % float(size / 1000)
    return file_size

def read_data(data, features, samples):
    X, Y = [], []
    for feature in features:
        feature_data = []
        for label in data.keys():
            label_data = data[label][feature]
            feature_data.extend(label_data.values())
        X.append(feature_data)

    # Generate labels (Y)
    for label in data.keys():
        label_samples = min(samples, len(data[label][features[0]]))
        Y.extend([label] * label_samples)

    return X, Y


if __name__ == '__main__':
    pcap_path, samples = "C:\\Users\\code\\Traffic-Classification-Research\\Datasets\\NetFlow-QUIC1\\splitcap", 3000
    X, Y = generation(pcap_path, samples)
    # pretrain data
    # pretrain_dataset_generation(pcap_path)
    print("X:%s\tx:%s\tY:%s" % (len(X), len(X[0]), len(Y)))
    print("First few elements of X:")
    print("First few elements of X:")
    # combine dataset.json
    # combine_dataset_json()