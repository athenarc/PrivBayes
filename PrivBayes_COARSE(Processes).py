import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
from scipy import special as sp
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
import numpy as np
import csv
import itertools
import random as rd
import networkx as nx
import math
import xgboost as xgb
from multiprocessing import Process, Lock, RLock
import sys
import time
import multiprocessing
import gc
import pickle

def GetSubsets(attr_set,k,choice):
    if (choice == 0): # Get all possible subsets A' (with size k + 1) of the attribute set A
        List = (list(itertools.combinations(attr_set,k+1)))
    else: # Get all possible subsets A' (with size less than or equal to k) of the attribute set A
        List = []
        for i in range (1,k+1):
            List = List + (list(itertools.combinations(attr_set,i)))
    return List

def normalize(List): # Normalize to [0,1]
    total_sum = sum(List)
    if (total_sum != 0):
        List = [(x/total_sum) for x in List]
    return List

def GetCounts(Attributes,Dataset,dA_sub,Attribute_dict,AddNoise,StrL,epsilon,k): # This is a function that each user runs on his own dataset from a dataset (returns counts cA of dataset)

    NumOfAttrs = len(Attributes)

    if (AddNoise):
        if(StrL == 2):
            e_2 = epsilon
            mu = 0
            b = 2*NumOfAttrs/(e_2)
        elif (StrL == 1):
            mu = 0
            b = (2*sp.comb(NumOfAttrs,k + 1,exact='True'))/epsilon
    else:
        mu = 1
        b = 1

    Counts = [0 for i in range(0,len(dA_sub))]
    Indexes = [Attribute_dict[attr] for attr in Attributes]
    noise = []

    Ai_Dataset_Columns = Dataset[:, Indexes]
    del Indexes
    gc.collect()

    for row in Ai_Dataset_Columns:
        i = 0
        for attr_values in dA_sub:
            if (list(row) == list(attr_values)):
                Counts[i] = Counts[i] + 1
                break

            i = i + 1

    if (AddNoise):
        noise = np.random.laplace(loc=mu,scale=b,size=len(dA_sub)).tolist()
        noise = [round(x) for x in noise] # noise must be integers
        noise = [0 if (x<0) else x for x in noise] # noise must be >=0

        Counts = AddListsOfNumbers(Counts,noise) # Add noise to count to ensure ε-differential privacy
        Counts = [round(x) for x in Counts] # counts must be integers
        Counts = [0 if (x<0) else x for x in Counts] # counts must be >=0

    return Counts,sum(noise)

def AddListsOfNumbers(List1,List2):

    if (len(List1) == len(List2)):
        Result = [0 for i in range(0,len(List1))]

        for i in range(0,len(List1)): # Sum counts of each dataset to find cAi to calculate the probability (Line 5)
            Result[i] = List1[i] + List2[i]

        return Result
    else:
        print("Error!! The lists must have the same length !!")

def AttrDomProd(Attributes,Attr_domains): # Calculate dA_sub' for current A'

    Needed_Attr_domains = [Attr_domains[attr] for attr in Attributes]
    dA_sub = list(itertools.product(*Needed_Attr_domains)) # We construct the arbitrary ordered dA_sub

    return dA_sub

def unique(list1):

    # insert the list to the set
    list_set = set(list1)
    # convert the set to the list
    unique_list = (list(list_set))

    return unique_list

def CalculateProbabilities(A,Attr_domains,All_Probabilities,k,Child_Parents):

    #print("Child_Parents are",Child_Parents)

    New_Probabilities = {} # Of k-sized or less attr comb
    Up_to_k_sized_attr_combinations = GetSubsets(A,k,1)

    '''    
    else: # We want only those to calculate the cond_probs joint and marginal probabilities
        Up_to_k_sized_attr_combinations = GetSubsets(A,1,1) # We want all the marginal probabilities P(A) of the attributes
        for child,parents in Child_Parents.items():
            if (len(parents) >= 1): # We dont want the empty set that is the parent set of the starting attribute or to re add marginal probabilities
                Up_to_k_sized_attr_combinations.append(tuple(parents))    # We want all the joint probabilities of the parents ONLY of the attributes
                print("[child] + parents are",[child] + parents)
                if (len ([child] + parents) < k) and (len ([child] + parents) > 1):
                    Up_to_k_sized_attr_combinations.append(tuple([child] + parents))

    '''
    #print ("Up_to_k_sized_attr_combinations are ",Up_to_k_sized_attr_combinations)

    for attr_comb in Up_to_k_sized_attr_combinations:

            Needed_Attr_domains = [Attr_domains[attr] for attr in attr_comb]
            current_attrs_tuples = list(itertools.product(*Needed_Attr_domains))

            for cnt in range(0,len(current_attrs_tuples)):
                pairs = [(attr_comb[i],current_attrs_tuples[cnt][i]) for i in range(0,len(attr_comb))]
                New_Probabilities[frozenset(pairs)] = 0

    del Up_to_k_sized_attr_combinations
    gc.collect()
    #print ("New Probabilites are: ",New_Probabilities)

    # Calculate the probabilities for all combinations with size less or equal to k

    for Small_dist in New_Probabilities:
        for key1 in All_Probabilities:
            if (Small_dist.issubset(key1)):
                Larg_dist = key1
                #print("\nLarger Distribution is ", Larg_dist)
                #print("Current distribution is ", Small_dist)
                # If we want P(Class = 0) and the distribution we will use to find it is (Sex,Class = 0 , Pain) then atrr_to_add = [Sex,Pain],attr_to_find = [Class]

                attr_to_add = [x[0] for x in Larg_dist]
                attr_to_find = [x[0] for x in Small_dist]

                #print(attr_to_add)
                #print(attr_to_find)

                attr_to_add = (list(set(attr_to_add) - set(attr_to_find)))
                #print(attr_to_add)

                subset = list(Small_dist) # It must contain the attributes and values of the probability we wish to calculate

                Value_combs = FindFrozenSets(attr_to_add,Attr_domains,subset)

                for comb in Value_combs:
                    New_Probabilities[Small_dist] = All_Probabilities[comb] + New_Probabilities[Small_dist]

                break

    return New_Probabilities

def FindFrozenSets(Attrs_for_sets,Attr_domains,List_to_include):

    Value_combs = [] # List of frozensets e.g frozen set = frozenset({('Sex', 1), ('Pain', 0), ('Class', 0)})
    Needed_Attr_domains = [Attr_domains[attr] for attr in Attrs_for_sets]
    current_attrs_tuples = list(itertools.product(*Needed_Attr_domains))

    for cnt in range(0,len(current_attrs_tuples)):
        pairs = [(Attrs_for_sets[i],current_attrs_tuples[cnt][i]) for i in range(0,len(Attrs_for_sets))]

        if (len(List_to_include) != 0):
            Value_combs.append(frozenset(pairs + List_to_include))
        else:
            Value_combs.append(frozenset(pairs))

    return Value_combs

def CalculateMutualInformation(X,Pa,Probabilities,Attr_domains): # I(X,Pa)

    # Create the needed frozen sets to use as keys

    X_sets = FindFrozenSets(X,Attr_domains,[])
    #print(Pa)
    Pa_sets = FindFrozenSets(list(Pa),Attr_domains,[])
    X_Pa_sets = FindFrozenSets(X + list(Pa),Attr_domains,[])

    '''
    print("X_sets")
    for i in X_sets:
        print(i,Probabilities[i])

    print("\nPa_sets")
    for i in Pa_sets:
        print(i,Probabilities[i])

    print("\nX_Pa_sets")
    for i in X_Pa_sets:
        print(i,Probabilities[i])
    '''

    MutualInf = 0

    for x in X_sets:
        px = Probabilities[x]
        if (px == 0):
            continue
        else:
            for pa in Pa_sets:
                p_pa = Probabilities[pa]
                p_x_pa = Probabilities[frozenset(list(x) + list(pa))]
                try:
                    MutualInf = p_x_pa * math.log(p_x_pa/(px*p_pa),2) + MutualInf
                except:
                    continue

    #print ("I(",  X ,",", Pa,") = " + str(MutualInf))
    #print()
    return MutualInf

def isBinary(attr_list,Attr_domains):
    for attr in attr_list:
        if (len(Attr_domains[attr]) != 2):
            return False

    return False

def CalcSensitivity(x,n,Attr_domains):

    if (isBinary([x[0]],Attr_domains)) or ((isBinary(x[1],Attr_domains)) and (len(x[1]) == 1)): # We must have ONE parent and either the child or the parent (or both) must binary
            Delta_I = ((1/n) * math.log(n,2)) + (((n-1)/n) * math.log(n/(n-1),2))
    else:
            Delta_I = ((2/n) * math.log((n+1)/2,2)) + (((n-1)/n) * math.log((n+1)/(n-1),2))

    return Delta_I

def NoiseChoice(Str_to_print):

    choice = int(input("Add noise? (0 for No, 1 for Yes): "))

    while (choice < 0) or (choice > 1) or (not isinstance(choice,int)):
        choice = int(input("Add noise? (0 for No, 1 for Yes): "))

    if (choice == 1):

            epsilon = float(input(Str_to_print))

            while (epsilon <= 0) or (not isinstance(epsilon,float)):
                epsilon = float(input(Str_to_print))

            AddNoise = True
    else:
            epsilon = 0
            AddNoise = False

    return AddNoise,epsilon

def ExponentialMechanism(List,Attr_domains,n,epsilon_1,N):  # The list contains tuples (X,Pa,I(X,Pa)), Returns diction with list elements as keys and probability as element

    ExpProbs = []

    for x in List:

        Delta_I = CalcSensitivity(x,n,Attr_domains)
        I = x[2]
        ExpProbs.append(math.exp((I*epsilon_1)/(2*(N-1)*Delta_I)))

    ExpProbs = normalize(ExpProbs)
    #print("ExpProbs are",ExpProbs)

    rand_choice = rd.random()
    low_bound = 0
    high_bound = 0
    choice = 0

    for i in range(0,len(ExpProbs)):

        low_bound = high_bound
        high_bound = low_bound + ExpProbs[i]

        if (rand_choice >= low_bound) and (rand_choice < high_bound):
            choice = i

    #print ("Chosen Pair (X,Pa,I) =",List[choice])
    #print()

    return List[choice]

def GetVotes(Dataset,A,k,Attr_domains,V,Datasets_Probabilities,Dataset_idx,AddNoise,epsilon_1): # This is a function that each user runs on his own dataset (Returns Child,Parent pair with the highest mutual information)

    # Calculate the needed probabilities (Ths must be done by every user separately and privately, only once)
    # Normally the dataset probabilities should only exist in this function. However, we define it in main() to access it more easily and to avoid redundant calculations
    # (calculating the same probabilities again and again) but the analyst does NOT know them

    if (Datasets_Probabilities[Dataset_idx] == 0): # We need to calculate the probabilities

        Attribute_dict = {A[i] : i for i in range (0,len(A))}
        A_subs = GetSubsets(A,k,0) # Creating A' sets
        #print("A_subs are",A_subs)

        # Calculate the probabilities

        A_sub_probabilities = {} # {Ai',[ probabilities of tuples in dAi'_sub ]}
        dA_sub_all = {} # {Ai',dA_sub}

        for A_sub in A_subs:  # Sharing the Noisy Sufficient Statistics line 2
            #print ("\n#################### {} : ####################".format(A_sub))

            dA_sub = AttrDomProd (A_sub,Attr_domains)
            dA_sub_all[A_sub] = dA_sub
            #print("dA_sub' = ",dA_sub)
            #print()

            All_A_sub_counts = {} # {Ai',[ counts of tuples in in dAi'_sub ]}
            All_A_sub_counts[A_sub] = [0 for i in range(0,len(dA_sub))]

            ## WE DONT USE LAPLACE BUT EXPONENTIAL HERE

            #print ("================== Dataset {} : ====================".format(i + 1))
            A_sub_count,Total_Noise = GetCounts(A_sub,Dataset,dA_sub,Attribute_dict,False,2,0,k) # Sharing the Noisy Sufficient Statistics line 4  - cAi' ^ (j)

            #print (A_sub_count)

            All_A_sub_counts[A_sub] = AddListsOfNumbers(All_A_sub_counts[A_sub],A_sub_count) # Sum counts of each dataset to find cAi to calculate the probability (Line 5)
            #print(sum(All_A_sub_counts[A_sub]))

            # Calculate probability PΑi for each Ai (Line 5)
            A_sub_probabilities[A_sub] = [All_A_sub_counts[A_sub][i]/len(Dataset) for i in range(0,len(dA_sub))]

        # Line 6

        del Attribute_dict
        del A_subs
        del All_A_sub_counts
        gc.collect()
        Datasets_Probabilities[Dataset_idx] = {} # e.g {frozenset(((A,0),(B,0))) : 0} = P(A = 0, B = 0) = 0

        for key in dA_sub_all.keys():
            for cnt in range(0,len(dA_sub_all[key])):
                pairs = [(key[i],dA_sub_all[key][cnt][i]) for i in range(0,len(key))]
                Datasets_Probabilities[Dataset_idx] [frozenset(pairs)] = A_sub_probabilities[key][cnt]

        del dA_sub_all
        gc.collect()
        Datasets_Probabilities[Dataset_idx].update(CalculateProbabilities(A,Attr_domains,Datasets_Probabilities[Dataset_idx],k,[]))

        #print ("\nJoint Probabilities for all distributions are:\n")
        #for values,probability in Datasets_Probabilities[Dataset_idx].items():
            #print("P(", sorted(list(values)), ") = ",round(probability,3))

    All_Probabilities = Datasets_Probabilities[Dataset_idx]

    tuple_with_max_I = (0,0,0)
    Omega = []
    NumOfAttributes = len(A)

    Parents = GetSubsets(V,k,1) # Pa(X) such as |Pa(X)| <= k

    if (NumOfAttributes != len(V)):

        for x in A :

            if (x in V): # A \ V
                continue
            else:
                #print("Parents are :",Parents)
                #print(" =============== ")

                for Pa in Parents: # Line 12 - 13
                    I = CalculateMutualInformation([x],Pa,All_Probabilities,Attr_domains)
                    tupl = (x,Pa,I)
                    Omega.append(tupl) # Line 13

        # Choose which I to return using the exponential mechanism (Line 12 of ALg 3.2)

        if (AddNoise):
            tuple_with_max_I = ExponentialMechanism(Omega,Attr_domains,len(Dataset),epsilon_1,NumOfAttributes)
        else:
            tuple_with_max_I = Omega[0]
            for item in Omega:
                if (item[2] > tuple_with_max_I[2]):
                    tuple_with_max_I = item

            #if (final_max_tuple[2] < tuple_with_max_I[2]): # REDUNDANT
                #final_max_tuple = tuple_with_max_I[:]

    return tuple_with_max_I

def StructureLearning1(Datasets,k,A,Attr_domains,Dataset_size,AddNoise,epsilon):

    V = []
    Attribute_dict = {A[i] : i for i in range (0,len(A))}
    A_subs = GetSubsets(A,k,0) # Creating A' sets
    #print("A_subs are",A_subs)
    M = len(Datasets) # Num of users

    # Calculate the probabilities

    A_sub_probabilities = {} # {Ai',[ probabilities of tuples in dAi'_sub  ]}
    dA_sub_all = {} # {Ai',dA_sub}

    New_Dataset_Size = Dataset_size

    for A_sub in A_subs:  # Sharing the Noisy Sufficient Statistics line 2

        New_Dataset_Size = Dataset_size

        #print ("\n#################### {} : ####################".format(A_sub))

        dA_sub = AttrDomProd (A_sub,Attr_domains)
        dA_sub_all[A_sub] = dA_sub
        #print("dA_sub' = ",dA_sub)
        #print()

        All_A_sub_counts = {} # {Ai',[ counts of tuples in in dAi'_sub ]}
        All_A_sub_counts[A_sub] = [0 for i in range(0,len(dA_sub))]

        for i in range(0,M): # For every dataset
            #print ("================== Dataset {} : ====================".format(i + 1))
            A_sub_count, Total_Noise = GetCounts(A_sub,Datasets[i],dA_sub,Attribute_dict,AddNoise,1,epsilon,k) # Sharing the Noisy Sufficient Statistics line 4  - cAi' ^ (j)
            #print (A_sub_count)

            New_Dataset_Size = New_Dataset_Size + Total_Noise
            #print (A_sub_count)

            All_A_sub_counts[A_sub] = AddListsOfNumbers(All_A_sub_counts[A_sub],A_sub_count) # Sum counts of each dataset to find cAi to calculate the probability (Line 5)
            #print(sum(All_A_sub_counts[A_sub]))

        A_sub_probabilities[A_sub] = []

        # Calculate probability PΑi for each Ai (Line 5)
        for i in range(0,len(dA_sub)):
            if (not AddNoise):
                A_sub_probabilities[A_sub].append(All_A_sub_counts[A_sub][i]/Dataset_size)
            else:
                try:
                    A_sub_probabilities[A_sub].append(All_A_sub_counts[A_sub][i]/New_Dataset_Size)
                except:
                    print("Dataset size is :",Dataset_size)
                    print("New Dataset size is :",New_Dataset_Size)

    del All_A_sub_counts
    gc.collect()

    #print()
    #for x in A_sub_probabilities.keys():
        #print ("Attributes: ", x, "\nCombinations: ", dA_sub_all[x])
        #print ("Probabilities: ", A_sub_probabilities[x] , "\n=======================================")

    # Line 6

    choice = rd.randint(0,len(A) - 1)
    chosen_attr = A[choice]
    first_chosen_attr = chosen_attr
    #print ("\nStarting node: ", first_chosen_attr)

    All_Probabilities = {} # e.g {frozenset(((A,0),(B,0))) : 0} = P(A = 0, B = 0) = 0

    for key in dA_sub_all.keys():
        for cnt in range(0,len(dA_sub_all[key])):
            pairs = [(key[i],dA_sub_all[key][cnt][i]) for i in range(0,len(key))]
            All_Probabilities[frozenset(pairs)] = A_sub_probabilities[key][cnt]

    del dA_sub_all
    del A_sub_probabilities
    gc.collect()
    #print ("Joint Probabilities for k+1 distributions are:\n")
    #for values,probability in All_Probabilities.items():
        #print("P(", list(values), ") = ",round(probability,2))

    All_Probabilities.update(CalculateProbabilities(A,Attr_domains,All_Probabilities,k,[]))

    #print ("\nJoint Probabilities for all distributions are:\n")
    #for values,probability in All_Probabilities.items():
        #print("P(", sorted(list(values)), ") = ",round(probability,3))

    # Line 7
    Gen_BN = nx.DiGraph()
    Gen_BN.add_node(chosen_attr)

    V.append(chosen_attr)
    A.remove(chosen_attr) # A \ V

    for i in range (0,len(A)):

        Omega = []

        #print()
        #print("A \ V is :",A)
        #print("V is :",V)
        #print()

        for x in A :

            Parents = GetSubsets(V,k,1) # Pa(X) such as |Pa(X)| <= k

            for Pa in Parents: # Line 12 - 13
                I = CalculateMutualInformation([x],Pa,All_Probabilities,Attr_domains)
                tupl = (x,Pa,I)
                Omega.append(tupl) # Line 13

        # Line 14
        tuple_with_max_I = Omega[0]
        for item in Omega:
            if (item[2] > tuple_with_max_I[2]):
                tuple_with_max_I = item

        # Line 15
        A.remove(tuple_with_max_I[0]) # A \ V
        V.append(tuple_with_max_I[0])

        if (not Gen_BN.has_node(chosen_attr)):
            Gen_BN.add_node(chosen_attr)

        for attr in tuple_with_max_I[1]:
            if (not Gen_BN.has_node(attr)):
                Gen_BN.add_node(attr)
            Gen_BN.add_edge(attr,tuple_with_max_I[0])

    #print("\nGenerated Bayesian Network is : ")
    #for i in Gen_BN.edges():
        #print (i[0] + " ----> " + i[1])
    #print()

    return Gen_BN,All_Probabilities,first_chosen_attr

def StructureLearning2(Datasets,k,A,Attr_domains,AddNoise,epsilon):

    Datasets_Probabilities = {i:0 for i in range(0,len(Datasets))} # {Dataset Index:Dictionary that contains its joint probabilities}

    # Line 1,2

    choice = rd.randint(0,len(A) - 1)
    first_chosen_attr = A[choice]
    #print ("\nStarting node: ", first_chosen_attr)

    V = []
    Gen_BN = nx.DiGraph()
    Gen_BN.add_node(first_chosen_attr)
    mu = 0
    b = 0
    e_1 = 0

    V.append(first_chosen_attr)
    #print("Starting attribute is",first_chosen_attr)

    # Determine noise

    if (AddNoise):
        e_1 = epsilon

    # Line 5 - 7

    for i in range(0,len(A)):
        Votes = {}
        for j in range(0,len(Datasets)):
            #print("\nDataset " + str(j))
            #print()
            Max_I_Pair = GetVotes(Datasets[j],A[:],k,Attr_domains,V,Datasets_Probabilities,j,AddNoise,e_1)
            #print("A is ", A)
            #print("V is ", V)
            Max_I_Pair = (Max_I_Pair[0],Max_I_Pair[1])

            if (Max_I_Pair in Votes.keys()):
                Votes[Max_I_Pair] = Votes[Max_I_Pair] + 1
            else:
                Votes[Max_I_Pair] = 1

        if (i == 0):
            '''
            print("The probabilities of the datasets are")
            for j in range(0,len(Datasets)):
                print("Dataset " + str(j))
                for values,probability in Datasets_Probabilities[j].items():
                    print("P(", sorted(list(values)), ") = ",round(probability,3))
            '''

            #print("\nThe votes for each value of i are: \n")

        # Line 8

        max_votes = 0
        max_pair = 0

        for key,votes in Votes.items():
            if (votes > max_votes):
                max_votes = votes
                max_pair = key
            if (votes == max_votes):
                if (rd.random() > 0.5): # Arbitrarily break ties
                    max_votes = votes
                    max_pair = key

        # Line 9

        if (i != len(A) - 1):
            #print("Votes " + str(i) + " are ",Votes)

            if (max_pair[0] in V):
                pass
            else:
                V.append(max_pair[0])

            #print("Max pair is :",max_pair,"from Votes " + str(i))
            #print()

            for attr in max_pair[1]:
                if (not Gen_BN.has_node(attr)):
                    Gen_BN.add_node(attr)
                Gen_BN.add_edge(attr,max_pair[0])

    del Datasets_Probabilities
    gc.collect()

    #print("\nGenerated Bayesian Network is : ")
    #for i in Gen_BN.edges():
        #print (i[0] + " ----> " + i[1])
    #print()

    return Gen_BN,first_chosen_attr

def GreedyBayes(dataset,NumOfAttrs,A,k,Attr_domains,AddNoise1,epsilon1):

    Attribute_dict = {A[i] : i for i in range (0,len(A))}
    A_subs = GetSubsets(A,k,0) # Creating A' sets
    #print("A_subs are",A_subs)

    # Calculate the probabilities

    A_sub_probabilities = {} # {Ai',[ probabilities of tuples in dAi'_sub  ]}
    dA_sub_all = {} # {Ai',dA_sub}

    Dataset_size = len(dataset)

    for A_sub in A_subs:

        #print ("\n#################### {} : ####################".format(A_sub))

        dA_sub = AttrDomProd (A_sub,Attr_domains)
        dA_sub_all[A_sub] = dA_sub
        #print("dA_sub' = ",dA_sub)
        #print()

        All_A_sub_counts = {} # {Ai',[ counts of tuples in in dAi'_sub ]}
        All_A_sub_counts[A_sub] = [0 for i in range(0,len(dA_sub))]

        #print ("================== Dataset {} : ====================".format(i + 1))
        A_sub_count, Total_Noise = GetCounts(A_sub,dataset,dA_sub,Attribute_dict,False,1,0,k) # Sharing the Noisy Sufficient Statistics line 4  - cAi' ^ (j)
        #print (A_sub_count)

        All_A_sub_counts[A_sub] = AddListsOfNumbers(All_A_sub_counts[A_sub],A_sub_count) # Sum counts of each dataset to find cAi to calculate the probability (Line 5)
        #print(sum(All_A_sub_counts[A_sub]))

        A_sub_probabilities[A_sub] = []

        # Calculate probability PΑi for each Ai (Line 5)
        for i in range(0,len(dA_sub)):
            A_sub_probabilities[A_sub].append(All_A_sub_counts[A_sub][i]/Dataset_size)

    #print()
    #for x in A_sub_probabilities.keys():
        #print ("Attributes: ", x, "\nCombinations: ", dA_sub_all[x])
        #print ("Probabilities: ", A_sub_probabilities[x] , "\n=======================================")

    del Attribute_dict
    del A_subs
    del All_A_sub_counts
    gc.collect()
    All_Probabilities = {} # e.g {frozenset(((A,0),(B,0))) : 0} = P(A = 0, B = 0) = 0

    for key in dA_sub_all.keys():
        for cnt in range(0,len(dA_sub_all[key])):
            pairs = [(key[i],dA_sub_all[key][cnt][i]) for i in range(0,len(key))]
            All_Probabilities[frozenset(pairs)] = A_sub_probabilities[key][cnt]

    del A_sub_probabilities
    gc.collect()

    #print ("Joint Probabilities for k+1 distributions are:\n")
    #for values,probability in All_Probabilities.items():
        #print("P(", list(values), ") = ",round(probability,2))

    All_Probabilities.update(CalculateProbabilities(A,Attr_domains,All_Probabilities,k,[]))

    # Build the Bayesian Network

    V = []

    choice = rd.randint(0,NumOfAttrs - 1)
    chosen_attr = A[choice]
    first_chosen_attr = chosen_attr

    Gen_BN = nx.DiGraph()
    Gen_BN.add_node(chosen_attr)

    V.append(chosen_attr)
    Parents = GetSubsets(V,k,1) # Pa(X) such as |Pa(X)| <= k
    A.remove(chosen_attr) # A \ V

    for i in range (0,NumOfAttrs):

        Omega = []

        #print(A)
        #print(V)

        if (len(A) == 0):
            break

        for x in A :

            #print(Parents)

            for Pa in Parents: # Line 12 - 13

                I = CalculateMutualInformation([x],Pa,All_Probabilities,Attr_domains)
                tupl = (x,Pa,I)
                Omega.append(tupl) # Line 13

        if (AddNoise1):
            tuple_with_max_I = ExponentialMechanism(Omega,Attr_domains,len(dataset),epsilon1,NumOfAttrs)
        else:
            tuple_with_max_I = Omega[0]
            for item in Omega:
                if (item[2] > tuple_with_max_I[2]):
                    tuple_with_max_I = item

        # Line 15
        A.remove(tuple_with_max_I[0]) # A \ V
        V.append(tuple_with_max_I[0])
        Parents = GetSubsets(V,k,1) # Pa(X) such as |Pa(X)| <= k

        if (not Gen_BN.has_node(chosen_attr)):
            Gen_BN.add_node(chosen_attr)

        for attr in tuple_with_max_I[1]:
            if (not Gen_BN.has_node(attr)):
                Gen_BN.add_node(attr)
            Gen_BN.add_edge(attr,tuple_with_max_I[0])

    return Gen_BN,All_Probabilities,first_chosen_attr

def NoisyConditionals(k,NumOfAttrs,Joint_Probabilities,epsilon,Dataset_size): ## Add noise to Joint probabilities

    mu = 0
    b = (4 * (NumOfAttrs - k)) / (Dataset_size * epsilon)
    Attr_keys = []

    # Add Laplace noise to every probability
    for key in Joint_Probabilities.keys():
        Joint_Probabilities[key] = Joint_Probabilities[key] + np.random.laplace(loc=mu,scale=b,size=1).tolist()[0]
        attrs = [pair[0] for pair in list(key)]
        Attr_keys.append(frozenset(attrs))
        if (Joint_Probabilities[key] < 0): # If less than zero, make zero
            Joint_Probabilities[key] = 0

    Attr_keys = set(Attr_keys)
    #print(Attr_keys)

    # Normalize the probabilities
    for attr_key in Attr_keys:

        prob_sum = 0
        for key in Joint_Probabilities.keys(): # Calculate the sum of probabilities
            attrs = frozenset([pair[0] for pair in list(key)])

            if (attr_key == attrs):
                prob_sum = Joint_Probabilities[key] + prob_sum

        for key in Joint_Probabilities.keys(): # Normalize them
            attrs = frozenset([pair[0] for pair in list(key)])

            if (attr_key == attrs):
                if (prob_sum !=0):
                    Joint_Probabilities[key] = Joint_Probabilities[key]/prob_sum

    return Joint_Probabilities

def SharingNoisyModel(Datasets,k,A,Attr_domains,AddNoise1,epsilon1,AddNoise2,epsilon2,Iter_cnt):

    CompleteDataset = []
    NumOfAttrs = len(A)

    cnt = 1

    for dataset in Datasets:

        G, Joint_Probabilities , starting_attr = GreedyBayes(dataset,NumOfAttrs,A[:],k,Attr_domains,AddNoise1,epsilon1)


        Child_Parents = {x:[] for x in A}

        #plt.show()

        logging.info("First Phase (Structure Learning) is complete!!")

        # Second Phase of PrivBayes: Parameter Learning

        for edge in G.edges():
             Child_Parents[edge[1]].append(edge[0])

        # Add noise to Joint Probabilities
        if(AddNoise2):
            Joint_Probabilities = NoisyConditionals(k,NumOfAttrs,Joint_Probabilities,epsilon2,len(dataset))
            #print(Joint_Probabilities)

        Conditional_Probabilities = ParameterLearning(Joint_Probabilities,Child_Parents,Attr_domains)

        if (AddNoise2): # If we added noise, normalize conditional probabilites
                for key in Conditional_Probabilities.keys():

                    for pair in key: # Get the parent part of this conditional probability
                        if (pair[0] == 'Parents'):
                            parents = pair[1]
                            break

                    # Get the all the combinations of values that this set of parents can give
                    Parents_combs = FindFrozenSets(list(parents),Attr_domains,[])
                    #print("Parents_combs",Parents_combs)
                    for comb in Parents_combs:
                        prob_sum = 0

                        for attr_comb in Conditional_Probabilities[key].keys(): # Sum all cond probabilities where parents have same values and then divide those probs by the sum
                            if (comb.issubset(attr_comb)):
                                if (Conditional_Probabilities[key][attr_comb] != math.inf):
                                    prob_sum = Conditional_Probabilities[key][attr_comb] + prob_sum

                        for attr_comb in Conditional_Probabilities[key].keys(): # Sum all cond probabilities where parents have same values and then divide those probs by the sum
                            if (frozenset(comb).issubset(attr_comb)):
                                if (Conditional_Probabilities[key][attr_comb] != math.inf):
                                    #print(prob_sum)
                                    if (prob_sum != 0):
                                        Conditional_Probabilities[key][attr_comb] = Conditional_Probabilities[key][attr_comb] / prob_sum
                                    #print("Changed prob:",key,attr_comb,Conditional_Probabilities[key][attr_comb])

        #print("Parameter Learning is complete! (" + str(multiprocessing.current_process()) + ")")
        '''
        print ("\nConditional Probabilities for the generated Bayesian Network are:")
        for child_parents,value_probabilites in Conditional_Probabilities.items():
            print("\nP(", child_parents , ") : ")
            for key,prob in value_probabilites.items():
                if (prob != math.inf):
                    print ("\t","P",sorted(key)," = ",round(prob,3))
                else:
                    print ("\t","P",sorted(key)," =  NOT DEFINED")
        '''

        SizeOfSynthDataset = len(dataset)
        Synth_Dataset = PriorSampling(G,Conditional_Probabilities,SizeOfSynthDataset,Child_Parents,Attr_domains)
        CompleteDataset = CompleteDataset + Synth_Dataset

    FinalSynthDataset = []

    for i in range(0,len(CompleteDataset)):

        tuple = []

        for attr in A:
            tuple.append(CompleteDataset[i][attr])
        FinalSynthDataset.append(tuple)

    return np.array(FinalSynthDataset).astype('float')

def GetProbabilities(Child_Parents,Datasets,Dataset_size,NumOfAttrs,k,Attr_domains,A,AddNoise,epsilon): # If we choose StrLearning2, so retireved = False, use this to get (noisy) joint probabilities

    '''
    Joint_Prob_Combs = []

    for child,parents in Child_Parents.items():
        temp_list = []
        temp_list.append(child)
        temp_list = temp_list + parents
        Joint_Prob_Combs.append(frozenset(temp_list))

    #print("Joint_Prob_Combs are ",Joint_Prob_Combs)
    '''

    Attribute_dict = {A[i] : i for i in range (0,len(A))}
    A_subs = GetSubsets(A,k,0) # Creating A' sets
    M  = len(Datasets) # Num of users

    # Calculate the probabilities

    A_sub_probabilities = {} # {Ai',[ probabilities of tuples in dAi'_sub  ]}
    dA_sub_all = {} # {Ai',dA_sub}

    for A_sub in A_subs:

        New_Dataset_Size = Dataset_size

        dA_sub = AttrDomProd (A_sub,Attr_domains)
        dA_sub_all[A_sub] = dA_sub
        #print("dA_sub' = ",dA_sub)
        #print()

        All_A_sub_counts = {} # {Ai',[ counts of tuples in in dAi'_sub ]}
        All_A_sub_counts[A_sub] = [0 for i in range(0,len(dA_sub))]

        for i in range(0,M):

            #print ("================== Dataset {} : ====================".format(i + 1))
            A_sub_count,Total_Noise = GetCounts(A_sub,Datasets[i],dA_sub,Attribute_dict,AddNoise,2,epsilon,k) # Sharing the Noisy Sufficient Statistics line 4  - cAi' ^ (j)
            New_Dataset_Size = New_Dataset_Size + Total_Noise
            #print (A_sub_count)

            All_A_sub_counts[A_sub] = AddListsOfNumbers(All_A_sub_counts[A_sub],A_sub_count) # Sum counts of each dataset to find cAi to calculate the probability (Line 5)
            #print(sum(All_A_sub_counts[A_sub]))

        A_sub_probabilities[A_sub] = []

            # Calculate probability PΑi for each Ai (Line 5)
        for i in range(0,len(dA_sub)):
            if (not AddNoise):
                A_sub_probabilities[A_sub].append(All_A_sub_counts[A_sub][i]/Dataset_size)
            else:
                A_sub_probabilities[A_sub].append(All_A_sub_counts[A_sub][i]/New_Dataset_Size)

    #print()
    #for x in A_sub_probabilities.keys():
        #print ("Attributes: ", x, "\nCombinations: ", dA_sub_all[x])
        #print ("Probabilities: ", A_sub_probabilities[x] , "\n=======================================")

    del All_A_sub_counts
    del A_subs
    del Attribute_dict
    gc.collect()

    All_Probabilities = {} # e.g {frozenset(((A,0),(B,0))) : 0} = P(A = 0, B = 0) = 0

    for key in dA_sub_all.keys():
        for cnt in range(0,len(dA_sub_all[key])):
            pairs = [(key[i],dA_sub_all[key][cnt][i]) for i in range(0,len(key))]
            All_Probabilities[frozenset(pairs)] = A_sub_probabilities[key][cnt]

    del A_sub_probabilities
    gc.collect()

    #print ("Joint Probabilities for k+1 distributions are:\n")
    #for values,probability in All_Probabilities.items():
        #print("P(", list(values), ") = ",round(probability,2))

    All_Probabilities.update(CalculateProbabilities(A,Attr_domains,All_Probabilities,k,Child_Parents))

    #print ("\nJoint Probabilities for the desired child-parents pairs are:\n")
    #for values,probability in All_Probabilities.items():
        #print("P(", sorted(list(values)), ") = ",round(probability,3))

    return All_Probabilities

def ParameterLearning(Joint_Probabilities,Child_Parents,Attr_domains):

    cond_prob = {}  # e.g {(("Child",X), ("Parents", [Par1,Par2,...])) : { frozenset((X,Value1), (Par1,Value), ....) : cond_probability }

    for key in Child_Parents.keys():
        child_part = ("Child",key)
        parents_part = ["Parents",[par for par in Child_Parents[key]]]
        parents_part[1] = frozenset(parents_part[1])
        new_key = tuple([child_part,tuple(parents_part)])

        cond_prob[new_key] = {}

    for key in cond_prob.keys():
        all_attrs = [] # Collect all attributes that this cond prob contains (child and parents)
        cur_child = key[0][1]
        all_attrs.append(cur_child)
        for parent in key[1][1]:
            all_attrs.append(parent)

        subkeys = FindFrozenSets(all_attrs,Attr_domains,[])
        #print('Cur_child is',cur_child)
        #print("Subkeys are",subkeys)

        for subkey in subkeys: # Calculate the conditional probability
            parents_fr_set = list(subkey)

            child_idx = 0

            for i in range(0,len(parents_fr_set)): # Remove the (child,value) tuple from subkey so that only the parents remain
                if (parents_fr_set[i][0] == cur_child):
                    child_tuple = parents_fr_set.pop(i)
                    break

            parents_fr_set = frozenset(parents_fr_set)

            #print("Subkey is",subkey)

            if (len(parents_fr_set) == 0): # we are currently calculating the joint probability of the starting attribute, so P(A,Pa) = P(A,empty set) = P(A)
                cond_prob[key][subkey] = Joint_Probabilities[subkey]
            else:
                if (Joint_Probabilities[parents_fr_set] != 0):
                        cond_prob[key][subkey] = Joint_Probabilities[subkey]/Joint_Probabilities[parents_fr_set]
                else:
                        cond_prob[key][subkey] = math.inf # Conditional Probability not defined

    return cond_prob

def PriorSampling(G,Conditional_Probabilities,SizeOfSynthDataset,Child_Parents,Attr_domains):

    SamplingQueue = list(nx.topological_sort(G))
    #print("\nThe sampling (topologic) order of the attributes is: ", SamplingQueue)
    SynthDataset = []

    for j in range(0,SizeOfSynthDataset): # Generating SizeOfSynthDataset tuples
        gen_tuple = {SamplingQueue[i]:0 for i in range(0,len(SamplingQueue))}

        #print()
        for i in range(0,len(SamplingQueue)): # Generating ONE tuple
            attr = SamplingQueue[i]
            set_Attr_value = False
            sample = rd.random()
            #print(round(sample,4))

            for key in Conditional_Probabilities.keys():
                if ('Child', attr) in key:
                    cur_probs = Conditional_Probabilities[key]
                    break

            if (attr == SamplingQueue[0]): # It is the starting attribute
                low_prob_bound = 0
                up_prob_bound = 0
                for key, prob in cur_probs.items(): # For every subinterval
                    if (prob == 0) or (prob == math.inf):
                        continue
                    cur_x_value = list(key)[0][1]
                    low_prob_bound = up_prob_bound
                    up_prob_bound = up_prob_bound + prob
                    if (sample >= low_prob_bound) and (sample < up_prob_bound):
                        #print("Interval of sample ",round(sample,4),": [",round(low_prob_bound,4),",",round(up_prob_bound,4),")")
                        gen_tuple[attr] = cur_x_value
                        set_Attr_value = True
                        break

                    #if (gen_tuple['Thal'] == 0) and (attr == "Thal"):
                            #print("(Sample,low_bound,high_bound) =",sample,low_prob_bound,up_prob_bound)

                if (set_Attr_value == False): # In case all probabilities are NOT DEFINED
                    gen_tuple[attr] = Attr_domains[attr][rd.randint(0,len(Attr_domains[attr]) -1)]

            else:
                low_prob_bound = 0
                up_prob_bound = 0

                cur_x_parents = Child_Parents[attr]
                cur_par_val_set = [(par,gen_tuple[par]) for par in cur_x_parents]
                cur_par_val_set = frozenset(cur_par_val_set)

                for key, prob in cur_probs.items(): #{frozenset((X,Value), (Pa,Val),.... : prob}
                    if (prob == 0) or (prob == math.inf):
                        continue

                    if (cur_par_val_set.issubset(key)): # We need the respective probability
                        low_prob_bound = up_prob_bound
                        up_prob_bound = up_prob_bound + prob

                        for pair in key:
                            if pair[0] == attr:
                                cur_x_value = pair[1]

                        if (sample >= low_prob_bound) and (sample < up_prob_bound):
                            #print("Interval of sample ",round(sample,4),": [",round(low_prob_bound,4),",",round(up_prob_bound,4),")")
                            gen_tuple[attr] = cur_x_value
                            set_Attr_value = True
                            break

                if (set_Attr_value == False): # In case all probabilities are NOT DEFINED
                    gen_tuple[attr] = Attr_domains[attr][rd.randint(0,len(Attr_domains[attr]) -1)]

        SynthDataset.append(gen_tuple)
        #print("Generated tuple is: ",gen_tuple)

    return SynthDataset

def RejectionSampling(G,Conditional_Probabilities,SizeOfSynthDataset,Child_Parents,evidence):

    SamplingQueue = list(nx.topological_sort(G))
    print("\nThe sampling (topologic) order of the attributes is: ", SamplingQueue)
    SynthDataset = []

    for j in range(0,SizeOfSynthDataset): # Generating SizeOfSynthDataset tuples
        continue_var = 0
        gen_tuple = {SamplingQueue[i]:0 for i in range(0,len(SamplingQueue))}

        for i in range(0,len(SamplingQueue)): # Generating ONE tuple
            attr = SamplingQueue[i]
            sample = rd.random()
            #print(round(sample,4))

            for key in Conditional_Probabilities.keys():
                if ('Child', attr) in key:
                    cur_probs = Conditional_Probabilities[key]
                    break

            if (attr == SamplingQueue[0]): # It is the starting attribute
                low_prob_bound = 0
                up_prob_bound = 0
                for key, prob in cur_probs.items(): # For every subinterval
                    if (prob == 0) or (prob == math.inf):
                        continue
                    cur_x_value = list(key)[0][1]
                    low_prob_bound = up_prob_bound
                    up_prob_bound = up_prob_bound + prob
                    if (sample >= low_prob_bound) and (sample < up_prob_bound):
                        #print("Interval of sample ",round(sample,4),": [",round(low_prob_bound,4),",",round(up_prob_bound,4),")")
                        gen_tuple[attr] = cur_x_value

                        if attr in evidence.keys():
                            if (gen_tuple[attr] != evidence[attr]):
                                continue_var = 1
                                break

                        break

                if (continue_var == 1):
                    break

            elif (continue_var == 1):
                break

            else:
                low_prob_bound = 0
                up_prob_bound = 0

                cur_x_parents = Child_Parents[attr]
                cur_par_val_set = [(par,gen_tuple[par]) for par in cur_x_parents]
                cur_par_val_set = frozenset(cur_par_val_set)

                for key, prob in cur_probs.items(): #{frozenset((X,Value), (Pa,Val),.... : prob}
                    if (prob == 0) or (prob == math.inf):
                        continue

                    if (cur_par_val_set.issubset(key)): # We need the respective probability
                        low_prob_bound = up_prob_bound
                        up_prob_bound = up_prob_bound + prob

                        for pair in key:
                            if pair[0] == attr:
                                cur_x_value = pair[1]

                        if (sample >= low_prob_bound) and (sample < up_prob_bound):
                            #print("Interval of sample ",round(sample,4),": [",round(low_prob_bound,4),",",round(up_prob_bound,4),")")
                            gen_tuple[attr] = cur_x_value

                            if attr in evidence.keys():
                                if (gen_tuple[attr] != evidence[attr]):
                                    continue_var = 1
                                    break

                if (continue_var == 1):
                    break

            if (continue_var == 1):
                    break

        if (continue_var == 1):
            continue_var = 0
            continue

        SynthDataset.append(gen_tuple)
        #print("Generated tuple is: ",gen_tuple)

    return SynthDataset

def LikelihoodWeighting(G,Conditional_Probabilities,SizeOfSynthDataset,Child_Parents,evidence):

    SamplingQueue = list(nx.topological_sort(G))
    print("\nThe sampling (topologic) order of the attributes is: ", SamplingQueue)
    SynthDataset = []
    Weights = []
    continue_var = 0

    for j in range(0,SizeOfSynthDataset): # Generating SizeOfSynthDataset tuples
        gen_tuple = {SamplingQueue[i]:0 for i in range(0,len(SamplingQueue))}
        tuple_weights = {SamplingQueue[i]:1 for i in range(0,len(SamplingQueue))}

        for i in range(0,len(SamplingQueue)): # Generating ONE tuple
            attr = SamplingQueue[i]
            if (attr in evidence.keys()): # force the evidence values
                gen_tuple[attr] = evidence[attr] # Value

                new_key1 = (('Child',attr),('Parents',frozenset(Child_Parents[attr])))
                new_key2 = [(attr,evidence[attr])]

                for par in Child_Parents[attr]:
                    new_key2.append((par,gen_tuple[par]))

                tuple_weights[attr] = Conditional_Probabilities[new_key1][frozenset(new_key2)] # Weight
                if (tuple_weights[attr] == math.inf): # Reject tuple
                    continue_var = 1
                    break
            else:
                sample = rd.random()
                for key in Conditional_Probabilities.keys():
                    if ('Child', attr) in key:
                        cur_probs = Conditional_Probabilities[key]
                        break

                if (attr == SamplingQueue[0]): # It is the starting attribute
                    low_prob_bound = 0
                    up_prob_bound = 0
                    for key, prob in cur_probs.items(): # For every subinterval
                        if (prob == 0) or (prob == math.inf):
                            continue
                        cur_x_value = list(key)[0][1]
                        low_prob_bound = up_prob_bound
                        up_prob_bound = up_prob_bound + prob
                        if (sample >= low_prob_bound) and (sample < up_prob_bound):
                            #print("Interval of sample ",round(sample,4),": [",round(low_prob_bound,4),",",round(up_prob_bound,4),")")
                            gen_tuple[attr] = cur_x_value
                            break
                else:
                    low_prob_bound = 0
                    up_prob_bound = 0

                    cur_x_parents = Child_Parents[attr]
                    cur_par_val_set = [(par,gen_tuple[par]) for par in cur_x_parents]
                    cur_par_val_set = frozenset(cur_par_val_set)

                    for key, prob in cur_probs.items(): #{frozenset((X,Value), (Pa,Val),.... : prob}
                        if (prob == 0) or (prob == math.inf):
                            continue

                        if (cur_par_val_set.issubset(key)): # We need the respective probability
                            low_prob_bound = up_prob_bound
                            up_prob_bound = up_prob_bound + prob

                            for pair in key:
                                if pair[0] == attr:
                                    cur_x_value = pair[1]

                            if (sample >= low_prob_bound) and (sample < up_prob_bound):
                                #print("Interval of sample ",round(sample,4),": [",round(low_prob_bound,4),",",round(up_prob_bound,4),")")
                                gen_tuple[attr] = cur_x_value
                                break

        if(continue_var==1):
            continue_var = 0
            continue

        SynthDataset.append(gen_tuple)
        Weights.append(tuple_weights)
        #print("Generated tuple is: ",gen_tuple)

    return SynthDataset, Weights

def SplitDataset(data,M,Attr_Domains,Attr_dict,split_choice):

    Datasets = []
    Dataset_Size = len(data)

    if (split_choice == 0):
        M = 1
        Datasets.append(data)
    elif (split_choice == 1):
        Num_of_tuples = int(len(data)/M)

        for i in range (0,M):

            end_limit = ((i+1)*Num_of_tuples) # In case len(data)/M is not an exact division
            if (end_limit >= len(data)):
                end_limit = len(data)

            Datasets.append(data[i*Num_of_tuples:end_limit])
    elif (split_choice == 2): # Random-sized datasets

        #print(len(data))

        random_intervals = [rd.randint(1,Dataset_Size) for i in range(0,M)]

        # Normalize the random values to [0,300]

        random_intervals = normalize(random_intervals)

        # Then scale to [x=0,y=Dataset_Size]:
        y = Dataset_Size
        x = 0
        range2 = y - x

        for i in range(0,len(random_intervals)):
            random_intervals[i] = round((random_intervals[i] * range2) + x)
            if (random_intervals[i] == 0) or (random_intervals[i] == 1):
                random_intervals[i] = 2

        random_intervals_sum = sum(random_intervals)

        '''
        print(random_intervals)
        print(random_intervals_sum)
        '''

        difference = abs(Dataset_Size - random_intervals_sum)

        if (Dataset_Size > random_intervals_sum):
            random_intervals[0] = random_intervals[0] + difference
        else:
            value = 0
            for i in range(0,len(random_intervals)): # Find a value that will not become zero if we subtract the difference
                if (random_intervals[i] - difference > 0):
                    value = i
                    break

            random_intervals[value] = random_intervals[value] - difference

        '''
        print()
        print(random_intervals)
        random_intervals_sum = sum(random_intervals)
        print(random_intervals_sum)
        '''

        start = 0
        end = 0

        for x in random_intervals:
            start = end
            end = start + x
            dataset = data[start:end]
            if (len(dataset) != 0):
                Datasets.append(dataset)
            #print(len(dataset))
    elif (split_choice == 3): # As many datasets/users as classes
        for class_val in Attr_Domains['Class']:
            dataset = []
            for row in data:
                if (row[-1] == class_val):

                    '''

                    temp = list(row) # Convert dataset values to integers
                    for i in range(0,len(temp)):
                        temp[i] = int(temp[i])
                        
                    dataset.append(temp)
                        
                    '''

                    dataset.append(list(row))

            #print(dataset)
            #print()
            if (len(dataset) != 0):
                Datasets.append(np.asarray(dataset))
    elif (split_choice == 4): # As many datasets/users as different values of the chosen attribute
        NumOfAtts = len(Attr_Domains.keys())
        chosen_attr = list(Attr_Domains.keys())[rd.randint(0,NumOfAtts-1)]
        print("Selected attribute for split is",chosen_attr)
        attr_pos = Attr_dict[chosen_attr]
        for val in Attr_Domains[chosen_attr]:
            dataset = []
            for row in data:
                if (row[attr_pos] == val):

                    '''

                    temp = list(row) # Convert dataset values to integers
                    for i in range(0,len(temp)):
                        temp[i] = int(temp[i])
                        
                    dataset.append(temp)
                        
                    '''

                    dataset.append(list(row))

            #print(dataset)
            #print()
            if (len(dataset) != 0):
                Datasets.append(np.asarray(dataset))

    M = len(Datasets)
    #print(M)
    #print(Datasets)
    return Datasets,M

def print_dataset(names,data):

    with np.printoptions(suppress = True,edgeitems=200, linewidth=150):
        print (names)
        print (data)

def NeuralNetworkClassifier(NumOfAttrs,NumOfClasses,X_train,X_test,y_train,y_test):

    # Normalize input data (standardization)
    def normalize(xdata):
        global m, s  #The mean (m) and std (s) of the data (xdata)
        m = np.mean(xdata,axis=0)  # mean
        s = np.std(xdata,axis=0)  # standard deviation
        x_norm = (xdata - m) / s
        return(x_norm)

    # Define the linear model
    def combine_inputs(X):
        Y_predicted_linear = tf.matmul(X, W) + b
        return Y_predicted_linear

    # Define the sigmoid inference model over the data X and return the result
    def inference(X):
        Y_prob = tf.nn.softmax(combine_inputs(X)) #Defines the output of the SoftMax (Probabilities)
        Y_predicted = tf.argmax(Y_prob, axis = 1, output_type=tf.int32) #Get the output with the largest probability
        return Y_prob, Y_predicted

    # Compute the loss over the training data using the predictions and true labels Y
    def loss(X, Y):
        Yhat = combine_inputs(X)
        SoftMaxCE = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=Yhat, labels=Y)
        loss =  tf.reduce_mean(SoftMaxCE)
        return loss

    # Optimizer
    def train(total_loss):
        learning_rate = 0.01
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        trainable_variables = tf.trainable_variables()
        update_op = optimizer.minimize(total_loss, var_list=trainable_variables)
        return update_op

    def evaluate(Xtest, Ytest):
        Y_prob, Y_predicted = inference(Xtest)
        accuracy= tf.reduce_mean(tf.cast(tf.equal(Y_predicted, Ytest), tf.float32))
        return accuracy

    #Initializations
    num_features = NumOfAttrs
    output_dim = NumOfClasses
    batch_size = 100

    with tf.variable_scope("other_charge", reuse=tf.AUTO_REUSE) as scope:

        # Variables of the model
        W = tf.get_variable(name='W', shape=(num_features, output_dim), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name='b', shape=(output_dim, ), dtype=tf.float32, initializer=tf.constant_initializer(value=0, dtype=tf.float32))

        # Input - output placeholders
        X = tf.placeholder(shape=(batch_size, num_features), dtype=tf.float32)  # Placeholder for a batch of input vectors
        Y = tf.placeholder(shape=(batch_size, ), dtype=tf.int32)  # Placeholder for target values

        # We have a better result with small test set and bigger training set
        #X_np = X_train
        #test_data = X_test
        X_np = normalize(X_train)
        test_data = normalize(X_test)
        Y_np = y_train
        test_labels = y_test

        init_op = tf.global_variables_initializer()

        # Execution: Training and Evaluation of the model
        with tf.Session() as sess:
            sess.run(init_op)
            num_epochs = 50
            num_examples = X_np.shape[0] - batch_size + 1
            total_loss = loss(X,Y)
            train_op = train(total_loss)
            perm_indices = np.arange(num_examples)

            for epoch in range(num_epochs):
                epoch_loss = 0
                np.random.shuffle(perm_indices)

                for i in range(num_examples-batch_size+1):
                    X_batch = X_np[perm_indices[i:i+batch_size], :]
                    Y_batch = Y_np[perm_indices[i:i+batch_size]]
                    feed_dict = {X: X_batch, Y: Y_batch}
                    batch_loss, _ = sess.run([total_loss, train_op] , feed_dict)
                    epoch_loss += batch_loss

                epoch_loss /= num_examples

            # Start the Evaluation based on the trained model
            Xtest = tf.placeholder(shape=(None, num_features), dtype=tf.float32) # Placeholder for one input vector
            Ytest = tf.placeholder(shape=(None, ), dtype=tf.int32)

            Ytest_prob, Ytest_predicted = inference(Xtest) # Define the graphs for the inference (probability) and prediction (binary)
            feed_dict_test = {Xtest: test_data, Ytest: test_labels.astype(int)}
            accuracy_np = evaluate (Xtest,Ytest)

            return "SoftMaxNeuralNetworkClassifier", sess.run(accuracy_np, feed_dict_test)

def Classification(data,repetitions,NumOfAttrs,NumOfClasses,testing_data):

    Results = []

    data_labels = data[:,-1]
    data = data[:,0:-1]
    testing_data_labels = testing_data[:,-1]
    testing_data = testing_data[:,0:-1]

    # We have a better result with small test set and bigger training set

    classifiers = [
        xgb.XGBClassifier(),
        DecisionTreeClassifier(),
        KNeighborsClassifier(),
        LinearSVC(),
        SVC(),
        RandomForestClassifier(),
        MLPClassifier(),
        AdaBoostClassifier(),
        GradientBoostingClassifier(),
        GaussianNB(),
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis()]

    #for test_size in np.arange(0.1,0.6,0.1):
    best_classifier_name = 0
    best_classifier_acc = 0

    classifiers_total_accuracies = {clf:0 for clf in classifiers}
    total_neural_network_acc = 0

    X_train = data
    X_test = testing_data
    y_train = data_labels
    y_test = testing_data_labels

    for k in range(0,repetitions):
        for clf in classifiers:

            name = clf.__class__.__name__

            try:
                #print("Classifier",name,"began!")
                clf.fit(X_train, y_train)
                train_predictions = clf.predict(X_test)
                #print("Classifier",name,"finished!")
            except BaseException as e:
                #print("Classifier",name,"did not work!")
                #print(str(e))
                continue

            acc = accuracy_score(y_test, train_predictions)
            classifiers_total_accuracies[clf] = classifiers_total_accuracies[clf] + acc

            #print("Accuracy: {:.4%}".format(acc))

        ######## Neural Network #########
        #print("Classifier SoftMaxNeuralNetwork began!")
        #name, accuracy = NeuralNetworkClassifier(NumOfAttrs - 1,NumOfClasses,X_train,X_test,y_train,y_test)
        #total_neural_network_acc = total_neural_network_acc + accuracy
        #print("Classifier SoftMaxNeuralNetwork finished!")

        #print("="*30)
        #print(name)
        #print('****Results****')
        #print("Accuracy: {:.4%}".format(accuracy))

        #print("="*30)

    for clf in classifiers:
        acc = classifiers_total_accuracies[clf]
        final_accuracy = (acc/repetitions)*100
        Results.append(round(final_accuracy,2))

        #f1.write(str(final_accuracy) + ",")

        name = clf.__class__.__name__
        #if (best_classifier_acc < final_accuracy):
            #best_classifier_acc = final_accuracy
            #best_classifier_name = name

    #if (best_classifier_acc < (total_neural_network_acc/repetitions)*100):
            #best_classifier_acc = (total_neural_network_acc/repetitions)*100
            #best_classifier_name = "SoftMaxNeuralNetworkClassifier"

    #Results.append((total_neural_network_acc/repetitions)*100)

    return Results

def ContinuousToCategorical (data,cont_attr_idx): # Quantization

    for attr in cont_attr_idx: # Continuous attribute (its entire column)

        temp_data_col = np.array(data[:,attr]).astype('float')

        max_val = max(temp_data_col)
        min_val = min(temp_data_col)
        #print(min_val,max_val)

        distance = max_val/10
        ranges = np.arange(0,max_val+1,distance)
        #print(ranges)

        for i in range(0,len(temp_data_col)):
            for middle_val in ranges:
                if ((temp_data_col[i] >= middle_val - (distance/2)) and (temp_data_col[i] <= middle_val)) or ((temp_data_col[i] < middle_val + (distance/2)) and (temp_data_col[i] >= middle_val)):
                    #print(data[:,attr][i],middle_val)
                    data[:,attr][i] = float(middle_val)
    return data

def GetAttrDomains(attributes,data):

    Attr_Domains = {}

    for i in range(0,len(attributes)):
        attr_column = data[: , i]
        domain = unique(attr_column)
        Attr_Domains[attributes[i]] = domain

    return Attr_Domains

def PrivBayes(dataset_choice,M,split_choice,k,str_choice,AddNoise,epsilon,AddNoise1,epsilon1,AddNoise2,epsilon2,classfication_choice,repetitions,Iter_cnt,data,testing_data,Attr_domains,Child_Parents,NumOfAttrs,attributes,Classif_Lock,Datasets,Results):

    NumOfClasses = len(Attr_domains['Class'])
    #sys.stdout.flush()

    '''
    Classif_Lock.acquire()

    with open("Outputs/inputData_" + str(Iter_cnt) + ".csv", 'w') as f:
        for row in data:
            line = ''
            for i in range(0,len(row)):
                if (i != len(row) -1):
                    line = line + str(row[i]) + ','
                else:
                    line = line + str(row[i])

            f.write(line + '\n')

    Classif_Lock.release()
    '''

    # First Phase of PrivBayes: Structure Learning

    Joint_Probabilities = []
    Dataset_size = len(data)

    if (str_choice == 1):
        ##print ("Please wait. This may take a few minutes.....")
        G,Joint_Probabilities,starting_attr = StructureLearning1(Datasets,k,attributes[:],Attr_domains,Dataset_size,AddNoise,epsilon)
        print("Structure Learning is complete! (" + str(multiprocessing.current_process()) + ")")
    elif (str_choice == 2):
        ##print ("Please wait. This may take a few minutes.....")
        G,starting_attr = StructureLearning2(Datasets,k,attributes[:],Attr_domains,AddNoise1,epsilon1)
        print("Structure Learning is complete! (" + str(multiprocessing.current_process()) + ")")
    else:
        ##print ("Please wait. This may take a few minutes.....")
        Synth_Dataset = SharingNoisyModel(Datasets,k,attributes[:],Attr_domains,AddNoise1,epsilon1,AddNoise2,epsilon2,Iter_cnt)
        print("SharingNoisyModel is complete! (" + str(multiprocessing.current_process()) + ")")

    if (str_choice != 3):

        #Child_Parents = {x:[] for x in attributes}

        '''
        # Plot the Bayesian Network
        pos = nx.layout.circular_layout(G)
        #pos = nx.layout.planar_layout(G)
        nodes = nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='blue',with_labels=True)
        edges = nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=40, edge_color='black',edge_cmap=plt.cm.Blues, width=2)
        nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif',font_color='white')
        plt.title(str(k) + "-degree Bayesian Network with "+ str(NumOfAttrs) + " nodes/attributes\n  and the node <<" + starting_attr + ">> as starting point")
        plt.savefig("Outputs/BN1_" + str(Iter_cnt) +".png")
        plt.close()

        #plt.show()
        
        f = open("Outputs/SumofMutualInf.txt","a+")
        totalI = 0

        for X in Child_Parents.keys():
            Pa = Child_Parents[X]
            if (len(Pa) != 0):
                I = CalculateMutualInformation([X],Pa,Joint_Probabilities,Attr_domains)
                totalI = totalI + I

        f.write("Outputs/BN1_" + str(Iter_cnt) + "," + str(totalI))
        f.write('\n')
        
        '''

        logging.info("First Phase (Structure Learning) is complete!!")

        # Second Phase of PrivBayes: Parameter Learning

        for edge in G.edges():
             Child_Parents[edge[1]].append(edge[0])

        if (str_choice == 1):
            Conditional_Probabilities = ParameterLearning(Joint_Probabilities,Child_Parents,Attr_domains)

            if (AddNoise): # If we added noise, normalize conditional probabilites
                for key in Conditional_Probabilities.keys():

                    for pair in key: # Get the parent part of this conditional probability
                        if (pair[0] == 'Parents'):
                            parents = pair[1]
                            break

                    # Get the all the combinations of values that this set of parents can give
                    Parents_combs = FindFrozenSets(list(parents),Attr_domains,[])
                    #print("Parents_combs",Parents_combs)
                    for comb in Parents_combs:
                        prob_sum = 0

                        for attr_comb in Conditional_Probabilities[key].keys(): # Sum all cond probabilities where parents have same values and then divide those probs by the sum
                            if (comb.issubset(attr_comb)):
                                if (Conditional_Probabilities[key][attr_comb] != math.inf):
                                    prob_sum = Conditional_Probabilities[key][attr_comb] + prob_sum

                        for attr_comb in Conditional_Probabilities[key].keys(): # Sum all cond probabilities where parents have same values and then divide those probs by the sum
                            if (frozenset(comb).issubset(attr_comb)):
                                if (Conditional_Probabilities[key][attr_comb] != math.inf):
                                    #print(prob_sum)
                                    if (prob_sum != 0):
                                        Conditional_Probabilities[key][attr_comb] = Conditional_Probabilities[key][attr_comb] / prob_sum
                                    #print("Changed prob:",key,attr_comb,Conditional_Probabilities[key][attr_comb])


        elif (str_choice == 2): # Retrieved = False
            Joint_Probabilities = GetProbabilities(Child_Parents,Datasets,Dataset_size,NumOfAttrs,k,Attr_domains,attributes,AddNoise2,epsilon2)
            Conditional_Probabilities = ParameterLearning(Joint_Probabilities,Child_Parents,Attr_domains)


            if (AddNoise2): # Normalize conditional probabilites
                for key in Conditional_Probabilities.keys():

                    for pair in key: # Get the parent part of this conditional probability
                        if (pair[0] == 'Parents'):
                            parents = pair[1]
                            break

                    # Get the all the combinations of values that this set of parents can give
                    Parents_combs = FindFrozenSets(list(parents),Attr_domains,[])
                    #print("Parents_combs",Parents_combs)
                    for comb in Parents_combs:
                        prob_sum = 0

                        for attr_comb in Conditional_Probabilities[key].keys(): # Sum all cond probabilities where parents have same values and then divide those probs by the sum
                            if (comb.issubset(attr_comb)):
                                if (Conditional_Probabilities[key][attr_comb] != math.inf):
                                    prob_sum = Conditional_Probabilities[key][attr_comb] + prob_sum

                        for attr_comb in Conditional_Probabilities[key].keys(): # Sum all cond probabilities where parents have same values and then divide those probs by the sum
                            if (frozenset(comb).issubset(attr_comb)):
                                if (Conditional_Probabilities[key][attr_comb] != math.inf):
                                    #print(prob_sum)
                                    if (prob_sum != 0):
                                        Conditional_Probabilities[key][attr_comb] = Conditional_Probabilities[key][attr_comb] / prob_sum
                                        #print("Changed prob:",key,attr_comb,Conditional_Probabilities[key][attr_comb])

        del Datasets
        gc.collect()
        print("Parameter Learning is complete! (" + str(multiprocessing.current_process()) + ")")

        #with open("Outputs/" + str(Iter_cnt), 'wb') as fp:
            #pickle.dump(Joint_Probabilities, fp)

        '''
        
        print ("\nConditional Probabilities for the generated Bayesian Network are:")
        for child_parents,value_probabilites in Conditional_Probabilities.items():
            print("\nP(", child_parents , ") : ")
            for key,prob in value_probabilites.items():
                if (prob != math.inf):
                    print ("\t","P",sorted(key)," = ",round(prob,3))
                else:
                    print ("\t","P",sorted(key)," =  NOT DEFINED")
                    
        '''

        logging.info("Second Phase (Parameter Learning) is complete!!")

        '''

        methods = "\n1.Prior Sampling \n2.Rejection Sampling \n3.Likelihood Weighting\n"

        print(methods)
        choice = int(input("Choose data synthesis method: "))

        while (choice > 3) or (choice < 1) or (not isinstance(choice,int)):
            choice = input(int("Error!! Choose data synthesis method: "))          
        
        SizeOfSynthDataset = int(input("Choose the size of the synthetic dataset (>0): "))

        while (SizeOfSynthDataset < 1) or (not isinstance(SizeOfSynthDataset,int)):
            SizeOfSynthDataset = int(input("Choose the size of the synthetic dataset (>0): "))
            
        '''

        Weights = []

        choice = 1
        SizeOfSynthDataset = Dataset_size

        if (choice == 1):
            Synth_Dataset = PriorSampling(G,Conditional_Probabilities,SizeOfSynthDataset,Child_Parents,Attr_domains)

            FinalSynthDataset = []

            for i in range(0,len(Synth_Dataset)):

                tuple = []

                for attr in attributes:
                    tuple.append(Synth_Dataset[i][attr])
                FinalSynthDataset.append(tuple)

            Synth_Dataset = np.array(FinalSynthDataset).astype('float')

        elif (choice == 2):

            evidence = {}
            new_attr = ''
            attr_val = 0
            num_of_attrs = 0

            num_of_attrs = int(input("Enter number of attributes that the evidence contains: "))

            while (SizeOfSynthDataset < 1) or (not isinstance(SizeOfSynthDataset,int)):
                num_of_attrs = int(input("Enter number of attributes that the evidence contains: "))

            for i in range(0,num_of_attrs):

                new_attr = input("Enter attribute: ")
                while (not new_attr in attributes):
                    new_attr = input("Enter an EXISTING attribute: ")

                print("The attribute <<" + new_attr + ">> takes values",Attr_domains[new_attr])
                attr_val = int(input("Enter a value for the attribute: "))
                while (not attr_val in Attr_domains[new_attr]):
                    attr_val = int(input("Enter an valid value: "))

                evidence[new_attr] = attr_val

            print("Evidence is ",evidence)
            Synth_Dataset = RejectionSampling(G,Conditional_Probabilities,SizeOfSynthDataset,Child_Parents,evidence)
        else:
            evidence = {}
            new_attr = ''
            attr_val = 0
            num_of_attrs = 0

            num_of_attrs = int(input("Enter number of attributes that the evidence contains: "))

            while (SizeOfSynthDataset < 1) or (not isinstance(SizeOfSynthDataset,int)):
                num_of_attrs = int(input("Enter number of attributes that the evidence contains: "))

            for i in range(0,num_of_attrs):

                new_attr = input("Enter attribute: ")
                while (not new_attr in attributes):
                    new_attr = input("Enter an EXISTING attribute: ")

                print("The attribute <<" + new_attr + ">> takes values ",Attr_domains[new_attr])
                attr_val = int(input("Enter a value for the attribute: "))
                while (not attr_val in Attr_domains[new_attr]):
                    attr_val = int(input("Enter an valid value: "))

                evidence[new_attr] = attr_val

            print("Evidence is ",evidence)
            Synth_Dataset, Weights = LikelihoodWeighting(G,Conditional_Probabilities,SizeOfSynthDataset,Child_Parents,evidence)

        print("Data Synthesis is complete! (" + str(multiprocessing.current_process()) + ")")
        '''
        print("\nThe synthetic dataset is : \n")

        for i in range(0,len(Synth_Dataset)):
            print(sorted(Synth_Dataset[i].items()))
            if (len(Weights) !=0):
                print("Its weights are: ")
                print("\t",Weights[i])
                print()
                
        '''

        '''
        Classif_Lock.acquire()

        with open("Outputs/syntheticData_" + str(Iter_cnt) + ".csv", 'w') as f:
            for row in Synth_Dataset:
                line = ''
                for i in range(0,NumOfAttrs):
                    if (i != NumOfAttrs - 1):
                        line = line + str(row[attributes[i]]) + ','
                    else:
                        line = line + str(row[attributes[i]])

                f.write(line + '\n')

        Classif_Lock.release()
        
        '''

        del Conditional_Probabilities
        gc.collect()
        logging.info("Third Phase (Data Synthesis) is complete!!")
        #print("The algorithm satisfies",epsilon+epsilon1+epsilon2,"- differential privacy for each data holder")

    if (classfication_choice == 1):

        print("Starting classification of synthetic data (" + str(multiprocessing.current_process()) + ")")
        #print ("Please wait. This may take a few minutes..... (" + str(multiprocessing.current_process()) + ")")

        #f1 = open("Outputs/Results.csv","a")
        Result = str(Iter_cnt) + "," + str(dataset_choice) + "," + str(M) + "," + str(split_choice) + "," + str(str_choice) + "," + str(k) + "," + str(AddNoise) + "," + str(epsilon) + "," + str(AddNoise1) + "," + str(epsilon1) + "," + str(AddNoise2) + "," + str(epsilon2) + ","

        for i in range(0,len(Results)):
            if (i != len(Results) - 1):
                Result = Result + str(Results[i]) + ","
            else:
                Result = Result + str(Results[i]) + ","

        #print("Finished classification of input data! (" + str(multiprocessing.current_process()) + ")")

        del Results
        Results = Classification(Synth_Dataset,repetitions,NumOfAttrs,NumOfClasses,testing_data)

        for i in range(0,len(Results)):
            if (i != len(Results) - 1):
                Result = Result + str(Results[i]) + ","
            else:
                Result = Result + str(Results[i])

        print("Finished classification of synthetic data! (" + str(multiprocessing.current_process()) + ")")
        #print("Classification is complete! (" + str(multiprocessing.current_process()) + ")")

    Classif_Lock.acquire()
    f1 = open("Outputs/Results.csv","a+")
    f1.write(Result)
    f1.write('\n')
    f1.close()
    Classif_Lock.release()

    logging.info("The program is complete!!")
    #print("Thread is finished!")

def ReadDataset(dataset_choice):

    # Read the dataset
    if (dataset_choice == 1):
        filename = 'Datasets/cleveland.data'
        raw_data = open(filename,'rt')
        reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
        x = list(reader)
        raw_data.close()
        data = np.array(x).astype('float')
        del x
        gc.collect()
        np.random.shuffle(data) # Shuffle the tuples

        # Attribute management

        # Remove no attribute
        attributes = ['Age','Sex', 'Pain','Pressure','Cholesterol','Sugar','Results','Pulse','Angina','Oldpeak','Slope','BloodVessels','Thal','Class']

        #print_dataset(attributes,data)

        #choice = int(input("Convert continuous attributes to categorical? (0:No, 1:Yes) "))

        #while (choice < 0) or (choice > 1) or (not isinstance(choice,int)):
            #choice = int(input("Convert continuous attributes to categorical? (0:No, 1:Yes) "))

        choice = 1

        if (choice ==1):
            data = ContinuousToCategorical(data,[0,3,4,7,9])

        # Remove unnecessary columns (continuous attributes)
        #data = np.delete(data, [0,3,4,7,9], axis=1)
        #attributes = ['Sex', 'Pain','Sugar','Results', 'Angina','Slope','BloodVessels','Thal','Class']

        # Remove 'Pressure' attribute
        #data = np.delete(data, [3], axis=1)
        #attributes = ['Age','Sex', 'Pain','Cholesterol','Sugar','Results','Pulse','Angina','Oldpeak','Slope','BloodVessels','Thal','Class']

        NumOfAttrs = len(attributes)
        #print(NumOfAttrs)
        Child_Parents = {x:[] for x in attributes}

        Attr_domains = GetAttrDomains(attributes,data)
        #print(Attr_domains)
        #Attr_domains = {'Sex': [0,1], 'Pain': [0,1,2,3,4],'Sugar': [0,1],'Results':[0,1,2], 'Angina': [0,1],'Slope': [1,2,3],'Thal': [3,6,7],'Class': [0,1,2,3,4]}

        testing_data = data[:][-int(len(data)*0.1):-1]
        data = data[:-int(len(data)*0.1), :]

        '''
        print(data)
        print(len(data))
        print()
        print(testing_data)
        print(len(testing_data))        
        '''

    elif (dataset_choice == 2):

        # Read the dataset
        #choice = int(input("Old (big) version (0) or new (small) version (1) ? "))

        #while (choice < 0) or (choice > 1) or (not isinstance(choice,int)):
            #choice = int(input("Old-Bigger version (0) or new-smaller version (1) ? "))

        choice = 1

        if (choice == 1):
            filename = 'Datasets/poker-hand-training.data'
        else:
            filename = 'Datasets/poker-hand-training_old.data'

        raw_data = open(filename,'rt')
        reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)

        attributes = ['S1','C1','S2','C2','S3','C3','S4','C4','S5','C5','Class']
        NumOfAttrs = len(attributes)
        #print(NumOfAttrs)
        Child_Parents = {x:[] for x in attributes}
        x = list(reader)
        raw_data.close()
        data = np.array(x).astype('float')
        del x
        gc.collect()
        #print("data shape:", data.shape)

        Attr_domains = GetAttrDomains(attributes,data)
        #print(Attr_domains)
        #Attr_domains = {'S1': [1,2,3,4],'C1': [1,2,3,4,5,6,7,8,9,10,11,12,13],'S2': [1,2,3,4], 'C2': [1,2,3,4,5,6,7,8,9,10,11,12,13],'S3': [1,2,3,4],'C3': [1,2,3,4,5,6,7,8,9,10,11,12,13],'S4': [1,2,3,4], 'C4': [1,2,3,4,5,6,7,8,9,10,11,12,13],'S5': [1,2,3,4], 'C5': [1,2,3,4,5,6,7,8,9,10,11,12,13], 'Class' : [0,1,2,3,4,5,6,7,8,9] }

        '''
        filename1 = 'Datasets/poker-hand-testing.data'
        raw_data = open(filename,'rt')
        reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
        x = list(reader)
        data1 = np.array(x).astype('float')
        data = data + data1        
        '''

        np.random.shuffle(data) # Shuffle the tuples

        testing_data = data[:][-int(len(data)*0.1):-1]
        data = data[:-int(len(data)*0.1), :]

    elif (dataset_choice == 3):
        # Read the dataset
        filename = 'Datasets/adult.data_new'
        raw_data = open(filename,'rt')
        reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
        x = list(reader)
        raw_data.close()
        data = np.array(x).astype('float')
        del x
        gc.collect()
        #print(type(data))
        #print("data shape:", data.shape)
        np.random.shuffle(data) # Shuffle the tuples

        # Testing data
        filename = 'Datasets/adult.test_new'
        raw_data = open(filename,'rt')
        reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
        x = list(reader)
        raw_data.close()
        testing_data = np.array(x).astype('float')
        del x
        np.random.shuffle(testing_data) # Shuffle the tuples

        # Attribute management

        # Remove no attribute
        attributes = ['Age','Workclass', 'Fnlwgt','Education','EducationNum','Marital Status','Occupation','Relationship','Race','Sex','Capital-Gain','Capital-Loss','Hours-Per-Week','Native_Country','Class']

        #choice = int(input("Convert continuous attributes to categorical? (0:No, 1:Yes) "))

        #while (choice < 0) or (choice > 1) or (not isinstance(choice,int)):
            #choice = int(input("Convert continuous attributes to categorical? (0:No, 1:Yes) "))

        choice = 1

        if (choice ==1):
            data = ContinuousToCategorical(data,[0,2,4,10,11,12])

        # Remove continuous attributes attribute
        #data = np.delete(data, [0,2,4,10,11,12], axis=1)
        #attributes = ['Workclass','Education','Marital Status','Occupation','Relationship','Race','Sex','Native_Country','Class']

        NumOfAttrs = len(attributes)
        #print(NumOfAttrs)
        #print(data)
        Child_Parents = {x:[] for x in attributes}

        Attr_domains = GetAttrDomains(attributes,data)

    return data,testing_data,Attr_domains,Child_Parents,NumOfAttrs,attributes

def BayesNetClassification(ConditionalProbabilities,Data,attr_dict,Attr_domains):

    Labels = Data[:,-1]
    Results = [0 for i in range(0,len(Labels))]

    tupl = ('Child', 'Class')

    for key in ConditionalProbabilities.keys():
        if (key[0] == tupl):
            Class_key = key
            break

    row_cnt = 0

    parents = list(Class_key[1][1])
    print(parents)

    for row in Data: # We want every probability where Class is the child and the parents (whoever they are) have values that fit those of the row

        max_probability = 0
        max_class = 0

        prob_set = [(parent,row[attr_dict[parent]]) for parent in parents] # e.g {frozenset({('Class', 0.0), ('Cholesterol', 225.6), ('Oldpeak', 0.0)})

        for Class_val in Attr_domains["Class"]:
            new_prob_set = prob_set[:]
            new_prob_set.append(("Class",Class_val))
            new_prob_set = frozenset(new_prob_set)

            prob = ConditionalProbabilities[Class_key][new_prob_set]
            if (prob > max_probability) and (prob != math.inf):
                max_probability = prob
                max_class = Class_val

        Results[row_cnt] = max_class
        row_cnt = row_cnt + 1

    accuracy = accuracy_score(Labels,Results)
    print("The accuracy of the Bayesian Network Classifier is",round(accuracy*100,2),"%")

########################################

if __name__ == "__main__":

    cnt = 1

    '''
    # Empty Outputs folder
    folder = 'Outputs'
    for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)
    '''
    try:
        filename = 'Outputs/Results.csv'
        raw_data = open(filename,'rt')
        reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
        x = list(reader)
        raw_data.close()
        Completed_Outputs = [int(row[0]) for row in x]
    except:
        x = []
        print("No results available!")
        Completed_Outputs = []

    #print(Completed_Outputs)
    del x
    #print(Completed_Outputs)
    gc.collect()

    classfication_choice = 1
    repetitions = 1
    M = 3
    AddNoise = True

    epsilon_vals = [0.01,0.02,0.05,0.1,0.2,0.5,1,5,10]

    procs = []

    lock = Lock()
    for dataset_choice in range(1,4):
                    data,testing_data,Attr_domains,Child_Parents,NumOfAttrs,attributes = ReadDataset(dataset_choice)
                    NumOfClasses = len(Attr_domains['Class'])

                    if (os.path.exists('Outputs/Input_data_class_res_' + str(dataset_choice))):
                        print("Classification results for dataset {} exist! Loading ...".format(str(dataset_choice)))
                        with open ('Outputs/Input_data_class_res_' + str(dataset_choice), 'rb') as fp:
                            Results = pickle.load(fp)
                        print("Data loaded!")
                    else:
                        print("Performing classification dataset {} ...".format(str(dataset_choice)))
                        Results = Classification(data,repetitions,NumOfAttrs,NumOfClasses,testing_data)
                        with open('Outputs/Input_data_class_res_' + str(dataset_choice), 'wb') as fp:
                            pickle.dump(Results, fp)
                        print("Classification of dataset {} is complete!".format(str(dataset_choice)))

                    print()

                    Attr_dict = {attributes[i] : i for i in range (0,len(attributes))}
                    for split_choice in range(1,5):
                        Datasets,M = SplitDataset(data,M,Attr_domains,Attr_dict,split_choice)
                        for k in range(1,3):
                            for str_choice in range(1,4):
                                    #for AddNoise in [False,True]:
                                            #if (AddNoise == True):
                                                for epsilon in epsilon_vals:
                                                    AddNoise1 = AddNoise
                                                    AddNoise2 = AddNoise
                                                    epsilon1 = epsilon/2
                                                    epsilon2 = epsilon/2

                                                    if (cnt in Completed_Outputs):
                                                        pass
                                                    else:
                                                        proc = Process(target=PrivBayes, args=(dataset_choice,M,split_choice,k,str_choice,AddNoise,epsilon,AddNoise1,epsilon1,AddNoise2,epsilon2,classfication_choice,repetitions,cnt,data,testing_data,Attr_domains,Child_Parents,NumOfAttrs,attributes,lock,Datasets,Results))
                                                        procs.append(proc)

                                                    cnt = cnt + 1

    del Completed_Outputs
    gc.collect()

    NumOfThreads = multiprocessing.cpu_count() - 1
    low_bound = 0
    up_bound = NumOfThreads

    RepeatTimes = int(len(procs)/NumOfThreads)
    #print("Completed making the processes")

    for i in range(0,RepeatTimes):

        if (up_bound > len(procs)):
            up_bound = len(procs)

        sub_procs = procs[low_bound:up_bound]

        for proc in sub_procs:
            proc.start()
            gc.collect()

        for proc in sub_procs:
            proc.join()
            gc.collect()

        low_bound = up_bound
        up_bound = low_bound + NumOfThreads
