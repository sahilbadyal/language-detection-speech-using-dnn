"""split data into training and validation sets"""
import csv
import os.path as path


def exists(file_name,index):
        if path.isfile("./data/train/pngaugm/"+file_name+'.'+str(i)+'.png'):
                return True
        return False


with open('./data/valEqual.csv', 'r') as csvfile:
    data = list(csv.reader(csvfile, delimiter=','))
    with open('./data/valEqualAugm.csv', 'w') as augmCsv:
            for line in data:
                    for i in range(20):
                            if (exists(line[0],i)):
                                    augmCsv.write(line[0]+'.'+str(i)+','+line[1]+'\n')
