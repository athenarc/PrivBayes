import numpy as np
import csv

Datasets = ['adult.data','adult.test']

for dataset in Datasets:

        raw_data = open(dataset,'rt')
        reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
        x = list(reader)
        data = np.array(x)

        for attr in range(0,15):
            if ((attr == 0) or (attr == 2) or (attr == 4) or (attr == 10) or (attr == 11) or (attr == 12)): # Continuous
                continue
            elif (attr == 1):
                for i in range(0,len(data[:,attr])):
                    if (data[:,attr][i]) == "Private":
                        data[:,attr][i] = 0
                    elif (data[:,attr][i]) == "Self-emp-not-inc":
                        data[:,attr][i] = 1
                    elif (data[:,attr][i]) == "Self-emp-inc":
                        data[:,attr][i] = 2
                    elif (data[:,attr][i]) == "Federal-gov":
                        data[:,attr][i] = 3
                    elif (data[:,attr][i]) == "Local-gov":
                        data[:,attr][i] = 4
                    elif (data[:,attr][i]) == "State-gov":
                        data[:,attr][i] = 5
                    elif (data[:,attr][i]) == "Without-pay":
                        data[:,attr][i] = 6
                    elif (data[:,attr][i]) == "Never-worked":
                        data[:,attr][i] = 7
            elif (attr == 3):
                for i in range(0,len(data[:,attr])):
                    if (data[:,attr][i]) == "Bachelors":
                        data[:,attr][i] = 0
                    elif (data[:,attr][i]) == "Some-college":
                        data[:,attr][i] = 1
                    elif (data[:,attr][i]) == "11th":
                        data[:,attr][i] = 2
                    elif (data[:,attr][i]) == "HS-grad":
                        data[:,attr][i] = 3
                    elif (data[:,attr][i]) == "Prof-school":
                        data[:,attr][i] = 4
                    elif (data[:,attr][i]) == "Assoc-acdm":
                        data[:,attr][i] = 5
                    elif (data[:,attr][i]) == "Assoc-voc":
                        data[:,attr][i] = 6
                    elif (data[:,attr][i]) == "9th":
                        data[:,attr][i] = 7
                    elif (data[:,attr][i]) == "7th-8th":
                        data[:,attr][i] = 8
                    elif (data[:,attr][i]) == "12th":
                        data[:,attr][i] = 9
                    elif (data[:,attr][i]) == "Masters":
                        data[:,attr][i] = 10
                    elif (data[:,attr][i]) == "1st-4th":
                        data[:,attr][i] = 11
                    elif (data[:,attr][i]) == "10th":
                        data[:,attr][i] = 12
                    elif (data[:,attr][i]) == "Doctorate":
                        data[:,attr][i] = 13
                    elif (data[:,attr][i]) == "5th-6th":
                        data[:,attr][i] = 14
                    elif (data[:,attr][i]) == "Preschool":
                        data[:,attr][i] = 15
            elif (attr == 5):
                for i in range(0,len(data[:,attr])):
                    if (data[:,attr][i]) == "Married-civ-spouse":
                        data[:,attr][i] = 0
                    elif (data[:,attr][i]) == "Divorced":
                        data[:,attr][i] = 1
                    elif (data[:,attr][i]) == "Never-married":
                        data[:,attr][i] = 2
                    elif (data[:,attr][i]) == "Separated":
                        data[:,attr][i] = 3
                    elif (data[:,attr][i]) == "Widowed":
                        data[:,attr][i] = 4
                    elif (data[:,attr][i]) == "Married-spouse-absent":
                        data[:,attr][i] = 5
                    elif (data[:,attr][i]) == "Married-AF-spouse":
                        data[:,attr][i] = 6
            elif (attr == 6):
                for i in range(0,len(data[:,attr])):
                    if (data[:,attr][i]) == "Tech-support":
                        data[:,attr][i] = 0
                    elif (data[:,attr][i]) == "Craft-repair":
                        data[:,attr][i] = 1
                    elif (data[:,attr][i]) == "Other-service":
                        data[:,attr][i] = 2
                    elif (data[:,attr][i]) == "Sales":
                        data[:,attr][i] = 3
                    elif (data[:,attr][i]) == "Exec-managerial":
                        data[:,attr][i] = 4
                    elif (data[:,attr][i]) == "Prof-specialty":
                        data[:,attr][i] = 5
                    elif (data[:,attr][i]) == "Handlers-cleaners":
                        data[:,attr][i] = 6
                    elif (data[:,attr][i]) == "Machine-op-inspct":
                        data[:,attr][i] = 7
                    elif (data[:,attr][i]) == "Adm-clerical":
                        data[:,attr][i] = 8
                    elif (data[:,attr][i]) == "Farming-fishing":
                        data[:,attr][i] = 9
                    elif (data[:,attr][i]) == "Transport-moving":
                        data[:,attr][i] = 10
                    elif (data[:,attr][i]) == "Priv-house-serv":
                        data[:,attr][i] = 11
                    elif (data[:,attr][i]) == "Protective-serv":
                        data[:,attr][i] = 12
                    elif (data[:,attr][i]) == "Armed-Forces":
                        data[:,attr][i] = 13
            elif (attr == 7):
                for i in range(0,len(data[:,attr])):
                    if (data[:,attr][i]) == "Wife":
                        data[:,attr][i] = 0
                    elif (data[:,attr][i]) == "Own-child":
                        data[:,attr][i] = 1
                    elif (data[:,attr][i]) == "Husband":
                        data[:,attr][i] = 2
                    elif (data[:,attr][i]) == "Not-in-family":
                        data[:,attr][i] = 3
                    elif (data[:,attr][i]) == "Other-relative":
                        data[:,attr][i] = 4
                    elif (data[:,attr][i]) == "Unmarried":
                        data[:,attr][i] = 5
            elif (attr == 8):
                for i in range(0,len(data[:,attr])):
                    if (data[:,attr][i]) == "White":
                        data[:,attr][i] = 0
                    elif (data[:,attr][i]) == "Asian-Pac-Islander":
                        data[:,attr][i] = 1
                    elif (data[:,attr][i]) == "Amer-Indian-Eskimo":
                        data[:,attr][i] = 2
                    elif (data[:,attr][i]) == "Other":
                        data[:,attr][i] = 3
                    elif (data[:,attr][i]) == "Black":
                        data[:,attr][i] = 4
            elif (attr == 9):
                for i in range(0,len(data[:,attr])):
                    if (data[:,attr][i]) == "Female":
                        data[:,attr][i] = 0
                    elif (data[:,attr][i]) == "Male":
                        data[:,attr][i] = 1
            elif (attr == 13):
                for i in range(0,len(data[:,attr])):
                    if (data[:,attr][i]) == "United-States":
                        data[:,attr][i] = 0
                    elif (data[:,attr][i]) == "Cambodia":
                        data[:,attr][i] = 1
                    elif (data[:,attr][i]) == "England":
                        data[:,attr][i] = 2
                    elif (data[:,attr][i]) == "Puerto-Rico":
                        data[:,attr][i] = 3
                    elif (data[:,attr][i]) == "Canada":
                        data[:,attr][i] = 4
                    elif (data[:,attr][i]) == "Germany":
                        data[:,attr][i] = 5
                    elif (data[:,attr][i]) == "Outlying-US(Guam-USVI-etc)":
                        data[:,attr][i] = 6
                    elif (data[:,attr][i]) == "India":
                        data[:,attr][i] = 7
                    elif (data[:,attr][i]) == "Japan":
                        data[:,attr][i] = 8
                    elif (data[:,attr][i]) == "Greece":
                        data[:,attr][i] = 9
                    elif (data[:,attr][i]) == "South":
                        data[:,attr][i] = 10
                    elif (data[:,attr][i]) == "China":
                        data[:,attr][i] = 11
                    elif (data[:,attr][i]) == "Cuba":
                        data[:,attr][i] = 12
                    elif (data[:,attr][i]) == "Iran":
                        data[:,attr][i] = 13
                    elif (data[:,attr][i]) == "Honduras":
                        data[:,attr][i] = 14
                    elif (data[:,attr][i]) == "Philippines":
                        data[:,attr][i] = 15
                    elif (data[:,attr][i]) == "Italy":
                        data[:,attr][i] = 16
                    elif (data[:,attr][i]) == "Poland":
                        data[:,attr][i] = 17
                    elif (data[:,attr][i]) == "Jamaica":
                        data[:,attr][i] = 18
                    elif (data[:,attr][i]) == "Vietnam":
                        data[:,attr][i] = 19
                    elif (data[:,attr][i]) == "Mexico":
                        data[:,attr][i] = 20
                    elif (data[:,attr][i]) == "Portugal":
                        data[:,attr][i] = 21
                    elif (data[:,attr][i]) == "Ireland":
                        data[:,attr][i] = 22
                    elif (data[:,attr][i]) == "France":
                        data[:,attr][i] = 23
                    elif (data[:,attr][i]) == "Dominican-Republic":
                        data[:,attr][i] = 24
                    elif (data[:,attr][i]) == "Laos":
                        data[:,attr][i] = 25
                    elif (data[:,attr][i]) == "Ecuador":
                        data[:,attr][i] = 26
                    elif (data[:,attr][i]) == "Taiwan":
                        data[:,attr][i] = 27
                    elif (data[:,attr][i]) == "Haiti":
                        data[:,attr][i] = 28
                    elif (data[:,attr][i]) == "Columbia":
                        data[:,attr][i] = 29
                    elif (data[:,attr][i]) == "Hungary":
                        data[:,attr][i] = 30
                    elif (data[:,attr][i]) == "Guatemala":
                        data[:,attr][i] = 31
                    elif (data[:,attr][i]) == "Nicaragua":
                        data[:,attr][i] = 32
                    elif (data[:,attr][i]) == "Scotland":
                        data[:,attr][i] = 33
                    elif (data[:,attr][i]) == "Thailand":
                        data[:,attr][i] = 34
                    elif (data[:,attr][i]) == "Yugoslavia":
                        data[:,attr][i] = 35
                    elif (data[:,attr][i]) == "El-Salvador":
                        data[:,attr][i] = 36
                    elif (data[:,attr][i]) == "Trinadad&Tobago":
                        data[:,attr][i] = 37
                    elif (data[:,attr][i]) == "Peru":
                        data[:,attr][i] = 38
                    elif (data[:,attr][i]) == "Hong":
                        data[:,attr][i] = 39
                    elif (data[:,attr][i]) == "Holand-Netherlands":
                        data[:,attr][i] = 40
            elif (attr == 14):
                for i in range(0,len(data[:,attr])):
                    if (data[:,attr][i]) == ">50K":
                        data[:,attr][i] = 0
                    elif (data[:,attr][i]) == "<=50K":
                        data[:,attr][i] = 1

        #print(data)

        with open(dataset + "_new", 'w') as f:
            for row in data:
                line = ''
                for i in range(0,15):
                    if (i != 14):
                        line = line + str(float(row[i])) + ','
                    else:
                        line = line + str(float(row[i]))

                f.write(line + '\n')

