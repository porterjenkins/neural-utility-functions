import re
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--file", type = str)

args = parser.parse_args()

in_file = args.file

total = []
with open(in_file) as file:
    list = []
    for line in file:
        for word in line.split():
            if re.match('\d*?\.\d+', word):
                list.append(word)

            if '**' in word:
                if len(list) > 0:
                    total.append(list.copy())
                    list.clear()

    total.append(list)
    # print(total)p

    for row in range(len(total[0])):
        for col in range(len(total)):
            if row >= len(total[0]) - 2:
                print(total[col][row], end='\t')
            else:
                print(total[col][row], end='\t')
        print()




