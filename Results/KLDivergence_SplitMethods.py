import pickle
import csv
import math
try:
        filename = 'KL_Results_Splt.csv'
        raw_data = open(filename,'rt')
        reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
        x = list(reader)
        raw_data.close()
        Completed_Outputs = [int(row[0]) for row in x]
except:
        x = []
        print("No results available!")
        Completed_Outputs = []

filename = 'NoisyProbDist/k=1_2(distributed)/Results.csv'
raw_data = open(filename,'rt')
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
x = list(reader)
raw_data.close()
NoisyCases = []

for row in x: # List of (ite_cnt,dataset,split_choice,str_learning,k)
    NoisyCases.append((int(row[0]),int(row[1]),int(row[3]),int(row[4]),int(row[5]),float(row[7]),0))

filename = 'NoisyProbDist/Centralized/Results.csv'
raw_data = open(filename,'rt')
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
x = list(reader)
raw_data.close()

for row in x: # List of (ite_cnt,dataset,split_choice,str_learning,k)
    NoisyCases.append((int(row[0]),int(row[1]),int(row[3]),int(row[4]),int(row[5]),float(row[7]),1))

filename = 'WithoutNoiseProbDist/Split Methods/Results.csv'
raw_data = open(filename,'rt')
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
x = list(reader)
raw_data.close()

NotNoisyCases = [(int(row[0]),int(row[1]),int(row[3]),int(row[4]),int(row[5])) for row in x] # List of (ite_cnt,dataset,split_choice,str_learning,k)

for idx,case in enumerate(NoisyCases):

    if (idx in Completed_Outputs):
        continue

    if (case[-1] == 0):
        try:
            fp = open ("NoisyProbDist/k=1_2(distributed)/" + str(case[0]), 'rb')
        except FileNotFoundError:
            print("File " + "k=1_2(distributed)/" + str(case[0]) + " not found!")
            continue
    else:
        try:
            fp = open ("NoisyProbDist/Centralized/" + str(case[0]), 'rb')
        except FileNotFoundError:
            print("File " + "Centralized/" + str(case[0]) + " not found!")
            continue

    P = pickle.load(fp) # Noisy Distribution
    fp.close()

    Q = 0

    for notnoisycase in NotNoisyCases:
        if (notnoisycase[1] == case[1]) and (notnoisycase[3] == case[3]) and (notnoisycase[4] == case[4]):
            f = open ("WithoutNoiseProbDist/Split Methods/" + str(notnoisycase[0]), 'rb')
            Q = pickle.load(f) # Not Noisy Distribution
            f.close()
            break

    KLDiv = 0

    for key in P:
        try:
            if (Q[key] != 0) and (P[key] != 0):
                KLDiv = KLDiv + P[key] * math.log(P[key]/Q[key],2)
        except:
            print(Q)

    f = open("KL_Results_Splt.csv","a+")
    f.write(str(case[0]) + "," + str(case[1]) + "," + str(case[2]) + "," + str(case[3]) + "," + str(case[4]) + "," + str(case[5]) + "," + str(abs(KLDiv)) + '\n')

