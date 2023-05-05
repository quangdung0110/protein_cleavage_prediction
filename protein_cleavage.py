## ==========================================================================
## Step 0: Load the necessary library 

import numpy as np
from sklearn import svm
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from Bio.Align import substitution_matrices
import time
start = time.time()

## ==========================================================================


p = 13
q = 2
## number of the proteins we choose
N = 10000


## ==========================================================================
## Step 1: Load the data from the data files

## This function return the list of the data
## Input: +) pre_data_list: the data read from the data file 
##        +) p: the number of amino acids before the position
##        +) q: the number of amino acids after the position
## Output: list of tuples (x,y), where 
##        +) x: a string the (p,q) neighborhood of the position
##        +) y: a boolean, represent that the position is a cleavage or not
def generated_data(pre_data_list, p, q):
    data_list = []
    for data_point in pre_data_list:
        protein_string = data_point[0]
        protein_length = len(protein_string)
        cleavage_position = data_point[1]
        for i in range(protein_length - p - q + 1):
            new_data_string = protein_string[i:i+p+q]
            new_label = False
            if (i+p == cleavage_position):
                new_label = True
            first_position = 0
            if (i == 0):
                first_position = 1
            ## why do we need first position here?
            data_list.append((new_data_string, new_label, first_position))
    return data_list

## Read the data file
with open('EUKSIG_13.red', 'r') as f:
    values = f.read().splitlines()

## number_data indicate the number of protein in the data file
## We must divide the number of lines by three, because for each data represent
## by three lines, the first line contains only the information of the organism,
## the second line contains the sequence of protein, the third line contains at
## which position the protein cleaves. 
number_data = len(values)//3

## The pre_data_list contains the list of the tuples (x,y), where
## +) x: the string of protein
## +) y: the position of cleavage
pre_data_list = []

for i in range(number_data):
    temp = values[3*i + 2]
    temp_length = len(temp)
    cleavage_position = 0
    for i in range(temp_length): 
        if (temp[i] == 'C'):
            cleavage_position = i
            break
    pre_data_list.append((values[3*i + 1], cleavage_position))


## data_list now contains the list of tuples (x,y), where x 
## is the (p,q) neighborhood of position, y the boolean that 
## the protein cleaves at this position or not
data_list = generated_data(pre_data_list, p, q)

## alignment_data: in fact, I don't know why I add the alignment_data here.
## It may be useless. 
# alignment_data = np.matrix([list(data_list[0][0])])
#for i in range(len(data_list)):
#    if (i > 0):
#        alignment_data = np.append(alignment_data, [list(data_list[i][0])], axis=0)

## ==========================================================================

## ==========================================================================
## Step 2: Emcoding the data into numerical vector.

## aa_order: the string that contains all 20 posible amino acids
aa_order = "ARNDCQEGHILKMFPSTWYV"

## char_to_vec: convert a character representing the amino acid to an one-hot 
## vector of length 20 
def char_to_vec(char):
    vec = np.zeros(20)
    vec[aa_order.index(char)] = 1
    return vec

## embedding_sequence: converting a sequence of amino acids of length n 
## into an one-hot vector of length 20n 
def embedding_sequence(sequence):
    result = char_to_vec(sequence[0])
    for i in range(len(sequence)):
        if (i > 0):
            result = np.concatenate([result, char_to_vec(sequence[i])])
    return result

## ==========================================================================

## ==========================================================================
## Step 3: Loading the data into the numerical form and splitting them into
## training set and test set. 

## Loading the data, where X be the feature, Y be the label (-1 and 1).
X = np.matrix([embedding_sequence(data_list[0][0])])

Y  = np.array(1)
if (data_list[0][1] == False):
    Y = np.array(0)
for i in range(N):
    if (i != 0):
        X = np.append(X, [embedding_sequence(data_list[i][0])], axis=0)
        if (data_list[i][1]):
            Y = np.append(Y, 1)
        else: 
            Y = np.append(Y, 0)
first_position = np.zeros((len(X), 1))
for i in range(len(X)):
    first_position[i,0] = data_list[i][2]
X = np.hstack([X, first_position])

## Split the data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
first_position_train = X_train[:, -1]
X_train = X_train[:, :-1]
X_test = X_test[:, :-1]


## ==========================================================================

## ==========================================================================
## Step 4: Construct the kernel function 

## 1. First kernel: scalar product kernel. 
## There are nothing to do with this kernel, because the standard library of 
## scikit-learn has already had this kernel.

## 2. Second kernel: kernel using substitution matrix 
## For construct this kernel, we use the substitution matrix, which is computed
## in the Bio library.

## sub_matrix: a numpy matrix which we store the substitution matrix
sub_matrix = np.zeros((20, 20))

## temp_sub_matrix: the substitution matrix computed in the Bio library.
temp_sub_matrix = substitution_matrices.load('BLOSUM62')

## We need to copy the coefficients of the temp_sub_matrix, which can be gotten 
## by get(name_of_amino_acid), to our sub_matrix submatrix, for the computational
## reason. 
for i in range(20):
    for j in range(20):
        sub_matrix[i, j] = temp_sub_matrix.get(aa_order[i]).get(aa_order[j])

## sub_mat_kernel: the kernel use substitution matrix 
## This kernel takes two vector with the same dimension of 20n, 
## then return the value of kernel function.
## In fact, in our calculation, the two arguments are two 2d matrices, 
## so we must write astutely the kernel function that the compilation can
## implement it with the matrices. 
def sub_mat_kernel(vec_a, vec_b):
    result = 0
    M = np.identity(20)
    length = len(vec_a)
    sequence_length = length//20
    for i in range(sequence_length):
        if (vec_a[:,20*i:20*i+20].shape[1] == sub_matrix.shape[0]):
            result += np.dot(np.dot(vec_a[:,20*i:20*i+20], sub_matrix.T), vec_b[:,20*i:20*i+20].T)
    return result

def rbf_sub_mat_kernel(vec_a, vec_b):
    result = 0
    M = np.identity(20)
    length = len(vec_a)
    sequence_length = length//20
    for i in range(sequence_length):
        if (vec_a[:,20*i:20*i+20].shape[1] == sub_matrix.shape[0]):
            result += np.dot(np.dot(vec_a[:,20*i:20*i+20], sub_matrix.T), vec_b[:,20*i:20*i+20].T)
    diag = np.diag(sub_matrix)
    norm_a = 0
    norm_b = 0
    for i in range(sequence_length):
         if (vec_a[:,20*i:20*i+20].shape[1] == sub_matrix.shape[0]):
            norm_a += np.dot(vec_a[:,20*i:20*i+20],diag)
            norm_b += np.dot(vec_b[:,20*i:20*i+20],diag)
    len_a = len(vec_a)
    len_b = len(vec_b)
    dummy_column_a = np.ones((len_a, 1))
    dummy_column_b  = np.ones((len_b, 1))

    temp_vec_a = np.hstack([norm_a.T, dummy_column_a])
    temp_vec_b = np.hstack([dummy_column_b, norm_b.T])
    result = np.matmul(temp_vec_a, temp_vec_b.T) - 2 * result
    result = np.exp(-result/(2*0.01))
    return result


## 3. Third kernel: probabilistic kernel
## For construct the kernel, firstly, we must construct the probabilistic matrix,
## which is mention in the paper of J.P.Vert

## proba_matrix: function that compute the probabilistic matrix from training set. 
def proba_matrix():
    result_matrix = np.ones((p+q, 20))
    n_train = len(y_train)
    counting_row = np.ones(20)
    for i in range(n_train):
        if y_train[i] == 1:
            for j in range(p+q):
                for i_k in range(20):
                    if X_train[i,20*j + i_k] == 1: 
                        result_matrix[j, i_k] += 1

                    result_matrix[j,:] = result_matrix[j,:] + X_train[i,20*j:20*j+20]
    return result_matrix

## fre_matrix: the matrix where we store the probabilistic matrix 
fre_matrix = proba_matrix()

## proba_kernel: the probabilistic kernel
def proba_kernel(vec_a, vec_b):
    result = 0
    len_a = len(vec_a)
    len_b = len(vec_b)
    for i in range(p+q):
        vec_a_test = vec_a[:,20*i:20*i+20]
        vec_b_test = vec_b[:,20*i:20*i+20]
        fre_vec = fre_matrix[i,:]
        modified_fre_vec = np.square(fre_vec) - fre_vec
        temp = np.matmul(np.matmul(vec_a_test, np.diag(modified_fre_vec)), vec_b_test.T)
        temp_vec_a = np.matmul(vec_a_test, fre_vec.T).T
        temp_vec_b = np.matmul(vec_b_test, fre_vec.T).T
        dummy_column_a = np.ones((len_a, 1))
        dummy_column_b  = np.ones((len_b, 1))
        temp_vec_a = np.hstack([temp_vec_a, dummy_column_a])
        temp_vec_b = np.hstack([dummy_column_b, temp_vec_b])
        temp += np.matmul(temp_vec_a, temp_vec_b.T)
        result += temp
    return result

## ==========================================================================

## ==========================================================================
## Step 5: Train model and get the precision score 

## Training model 
clf = svm.SVC(kernel=rbf_sub_mat_kernel)
clf.fit(X_train, y_train)


## y_pred: the predicted value for the test set 
y_pred = clf.predict(X_test)

## y_premitive: the predicted value for the training set 
y_premitive = clf.predict(X_train)

print("p =",p ,", q =",q,", N =", N)

## precision: precision score for test set 
precision = f1_score(y_test, y_pred)
print("F1-score for the test set: ", round(precision,3))

## precision_1: precision score for training set 
precision_1 = f1_score(y_train, y_premitive)
print("F1-score for the training set: ", precision_1)

## print the running time
end = time.time()
print("Running time: ",end - start, "s")
        