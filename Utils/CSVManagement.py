import csv
import os
from tqdm.notebook import tqdm

def combine_results(read_path, 
                    write_path,
                    n_labels):
    all_results = []

    for file in tqdm(sorted(os.listdir(read_path))):
        one_file = []

        with open(read_path+file, newline='') as f:
            read = csv.reader(f)
            read_list = list(read)

            for i in range(len(read_list)):
                one_file[i*(n_labels+2):i*(n_labels+2)+n_labels+1] = read_list[i][0:n_labels+1]

            one_file.insert(0,file.split('.',1)[0])
            all_results.append(one_file)

    with open(write_path+'all_results.csv','w') as f:
        write = csv.writer(f)
        write.writerows(all_results)