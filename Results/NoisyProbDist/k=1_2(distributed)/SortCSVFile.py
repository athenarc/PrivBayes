import csv
import operator

raw_data = open("Results.csv", 'rt')
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
x = list(reader)
raw_data.close()
x.sort(key=operator.itemgetter(4))

with open("Results_sorted.csv", 'w') as f:
    for row in x:
        for i in range(0,len(row)):
            if (i != len(row) - 1):
                f.write(row[i] + ",")
            else:
                f.write(row[i])
        f.write('\n')
