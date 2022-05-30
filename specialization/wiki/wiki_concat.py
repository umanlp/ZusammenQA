import random
import numpy as np
import glob
# km: upsampling needed!
# select 7500 for each of the language
def save_file(file_name, corpus_list):
    c = 0
    with open(file_name, 'a') as s:
        for i, element in enumerate(corpus_list):
            element = element.replace('&lt;br&gt;', "").strip()
            c+=1
            s.write("{}\n".format(element))
            if i%10000==0:
                print(i)
    print(c)
    
random_seed = 30
save_file_train_path = "/work-ceph/wifo3/wiki/all16/concat_train.txt"
save_file_test_path = "/work-ceph/wifo3/wiki/all16/concat_test.txt"
random.seed(random_seed)
np.random.seed(random_seed)
files =  glob.glob("/work-ceph/wifo3/wiki/final/*.txt")
final_train = []
final_test = []
for input_file in files:
    with open(input_file, 'r') as f:
        data = f.read().split('\n')
    data = [d for d in data if len(d)>30]
    print(input_file)
    print("Original data size: {}".format(len(data)))
    random.shuffle(data)
    test = data[0:750]
    if 'km' not in input_file:
        train = data[750:8250]
    else:
        train = data[750:8250]+data[750:8250]+data[750:8250]
        random.shuffle(train)
        train = train[0:7500]
        train_km = train
    print("Testing data size: {}".format(len(test)))
    print("Training data size: {}".format(len(train)))
    final_train+=train
    final_test+=test
    #print("Testing data size: {}".format(len(test)))
random.shuffle(final_train)
random.shuffle(final_test)
print("Final train data size: {}".format(len(final_train)))
print("Final test data size: {}".format(len(final_test)))
save_file(save_file_train_path, final_train)
save_file(save_file_test_path, final_test)