import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--original", type=str)
parser.add_argument("--new", type=str)
parser.add_argument("--experiment", type=str)

args = parser.parse_args()

original_file = args.original
new_file = args.new
experiment = args.experiment

print(f'{experiment+"_original"} vs. {experiment}')
with open(original_file) as file:
    og_times = []
    for line in file:
        for s in line.split():
            if s.isdigit():
                og_times.append(int(s))

with open(new_file) as file:
    new_times = []
    for line in file:
        for s in line.split():
            if s.isdigit():
                new_times.append(int(s))


print("Original\tNew")
for i in range(len(og_times)):
    print(f'{og_times[i]}\t\t{new_times[i]}')

old_avg = sum(og_times)/len(og_times)
new_avg = sum(new_times)/len(new_times)


print(f'\tAverages')
print(f'Original\tNew')
print(f'{old_avg}\t{new_avg}')
if old_avg - new_avg > 0:
    print(f"Original is {old_avg - new_avg} seconds slower on average")
else:
    print(f"New is {round(-1 * (old_avg - new_avg),3)} seconds slower on average")

print("\n")
