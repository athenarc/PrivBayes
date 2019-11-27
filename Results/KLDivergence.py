import pickle
import csv
import math

try:
        filename = 'KL_Results.csv'
        raw_data = open(filename,'rt')
        reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
        x = list(reader)
        raw_data.close()
        Completed_Outputs = [int(row[0]) for row in x]
except:
        x = []
        print("No results available!")
        Completed_Outputs = []

filename = 'NoisyProbDist/Results.csv'
raw_data = open(filename,'rt')
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
x = list(reader)
raw_data.close()

NoisyCases = []

for row in x: # List of (ite_cnt,dataset,split_choice,str_learning,k)
    if (int(row[0]) not in Completed_Outputs):
        NoisyCases.append((int(row[0]),int(row[1]),int(row[3]),int(row[4]),int(row[5]),float(row[7])))

filename = 'WithoutNoiseProbDist/Results.csv'
raw_data = open(filename,'rt')
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
x = list(reader)
raw_data.close()
NotNoisyCases = [(int(row[0]),int(row[1]),int(row[4]),int(row[5])) for row in x] # List of (ite_cnt,dataset,str_learning,k)

for idx,case in enumerate(NoisyCases):

    if (idx in Completed_Outputs):
        continue

    try:
        fp = open ("NoisyProbDist/" + str(case[0]), 'rb')
    except FileNotFoundError:
        print("File " + "NoisyProbDist/" + str(case[0]) + " not found!")
        continue

    P = pickle.load(fp) # Noisy Distribution
    fp.close()

    Q = 0

    for notnoisycase in NotNoisyCases:
        if (notnoisycase[1] == case[1]) and (notnoisycase[2] == case[2]) and (notnoisycase[3] == case[3]):
            f = open ("WithoutNoiseProbDist/" + str(notnoisycase[0]), 'rb')
            Q = pickle.load(f) # Not Noisy Distribution
            f.close()
            break

    KLDiv = 0

    for key in P:
        if (Q[key] != 0) and (P[key] != 0):
            KLDiv = KLDiv + P[key] * math.log(P[key]/Q[key],2)

    f = open("KL_Results.csv","a+")
    f.write(str(case[0]) + "," + str(case[1]) + "," + str(case[2]) + "," + str(case[3]) + "," + str(abs(KLDiv)) + '\n')

