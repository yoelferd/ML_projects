### Imports
import glob
import os
import numpy as np
from astropy.table import Table, Column

### This function reads the data.

def upload_files(your_path):
    file_list = glob.glob(os.path.join(os.getcwd(), your_path))
    corpus_name =[]
    for file_path in file_list:
        with open(file_path) as f_input:
            corpus_name.append(f_input.read())
    return corpus_name


#Here I open my data into language train sets and test set.

data_input_A = upload_files("symbol/language-training-langA*")
data_input_B = upload_files("symbol/language-training-langB*")
data_input_C = upload_files("symbol/language-training-langC*")

test_corpus = upload_files("symbol/language-test*")


def MarkovChain(data_input):
    #"Using python, build a Markov model for each of the languages."
    #This function takes in a training corpus and outputs a table
    #transition matrix for that language stores count of how many
    #times letter i has been followed by letter j in nested loop
    #then normalizes those values to set as probabilities

    # Pull out the list of our symbols in the data
    letters = list(set(data_input[2]))
    # Initializes empty probability matrix
    prob_matrix = np.zeros((len(letters),len(letters)))

    # Loop through data counting each instance of each transition
    # and add count to matrix
    for s in range(len(data_input)):
        for ind in range(len(data_input[s])-1):
            for i in range(len(letters)):
                if data_input[s][ind] == letters[i]:
                    for j in range(len(letters)):
                        if data_input[s][ind+1] == letters[j]:
                            prob_matrix[i][j] +=1

    # Normalize the matrix to find probabilities:
    prob_matrix = prob_matrix / prob_matrix.sum(axis=1)[:,None]

    # Set matrix as log probabilities:
    for ind, row in enumerate(prob_matrix):
        for index, item in enumerate(row):
            if item != 0:
                prob_matrix[ind][index] = np.log(item)
            if item == 0:
                prob_matrix[ind][index] = -4  #set as -4 to make prob 0.0001

    # Make a pretty table:
    Markov_Model = Table(prob_matrix, names = letters)

    #Return the pretty table
    return Markov_Model

MarkovChain(data_input_B)



# This function returns the probability of a particular state shift
# (from i to j) given a transition probability matrix for a language.
# Note this is for one singular letter state change.
#Reads transition matrix, return probability that state j comes after i

def prob_of_letter(i,j,Transition):
    letter_dict = {'A':0, 'e':1, 'g':2, 'k':3, 'o':4,'p':5, 't':6}
    return Transition[j][letter_dict[i]]


# This function returns the log probability of a string given a
# language by returning the sum of all log probabilities of
# each transition in the string from Transition matrix.

def prob_of_string_language(string,data_input_language):
    log_prob_of_lang = 0
    for i in range(len(string)-1):
        log_prob_of_lang += prob_of_letter(string[i],string[i+1],
                                           MarkovChain(data_input_language))
    return log_prob_of_lang

# This uses Bayes to calculate P(language|string)
# because P(language|str) is proportional to P(str|language)
# P(language model) = 1/3 for all strings and P(str)
# is constant as well. P(language) and P(str) are the same
# for all languages given a string so we can ignore them
# for classification then normalize the probabilities later
# to get the final probabilities; they must sum to 1.

## Produce results:

classes = []
posteriors=[]

for i in range(len(test_corpus)):
    # probability that the ith string in test corpus is language A:
    p_A = prob_of_string_language(test_corpus[i], data_input_A)

    # probability that the ith string in test corpus is language B:
    p_B = prob_of_string_language(test_corpus[i], data_input_B)

    # probability that the ith string in test corpus is language B:
    p_C = prob_of_string_language(test_corpus[i], data_input_C)

    # construct classes list:
    if max(p_A,p_B,p_C) == p_A:
        classes.append('A')
    elif max(p_A,p_B,p_C) == p_B:
        classes.append('B')
    elif max(p_A,p_B,p_C) == p_C:
        classes.append('C')

    # construct posteriors list by converting log probabilities
    # back to regular probability and normalizing:

    #unlog
    p_A = 10**p_A
    p_B = 10**p_B
    p_C = 10**p_C

    #normalize
    p_sum = p_A + p_B + p_C
    p_A, p_B, p_C = p_A/p_sum, p_B/p_sum, p_C/p_sum

    #add to list
    posteriors.append((i,[p_A, p_B, p_C]))

posteriors = np.asarray(posteriors)
print 'test corpus classes are:', classes
print 'posteriors: ', posteriors


#_____#2_____#
'''
NOTES:
P(v_1:T,h_1:T) = P(v_1|h_1)*P(h_1)*np.product(p(v:t|h:t)*P(h:t|h:t-1)) -from t=2 to N
In English:
Joint probability of the observed variable v from initial time 1 to time T (the string)
and hidden variable h from time 1 to time T (the list of speakers at times 1 to time T)
is the product of:
1. probability of v1 given h1. (Given speaker 1 is the first speaker,
the probability that he says the first observed phoneme:'e')
2. probability of h1. (Probability speaker 1 is the first speaker. Prior is likely 1/3
but can try other options to see if I get a higher likelihood, however that might
lead to overfitting so must be conscious of that)
3. product from t=2 to T
    a. emission probabilities of observed phoneme at time t given hidden speaker at time t
        - Probability particular speaker says particular phoneme
        - 7 x 3 matrix
        - initialize this randomly (then update?)
    b. transition probability of hidden speaker at t coming after the hidden speaker before it
        -[[0.9,0.05,0.05]
          [0.05,0.9,0.05]
          [0.05,0.05,0.9]]
GOAL: list P(h_t | v_1:u)  from t=1 to len(string)
What parameters maximize: p(h1:T|v1:T)
Hidden var: Language
Observed: letters
Transtion: Letter to letter

Given speaker 1 is talking, what phonemes will he use
#alpha past: probability(speaker_t, string[0:t])
alpha(h_t) = p(v_t|h_t)* sum(p(h_t|h_t-1)*alpha(h_t-1))
alpha(h_1) = p(v_1|h_1)*p(h_1)

#ß future: P(string[t+1:]|speaker_t)
ß(h_t-1) = np.sum(p(v_t|h_t)*p(h_t|h_t-1)ß(h_t))
'''


from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from hmmlearn import hmm

text = 'eoggeggAeggepgpppoppogopppgoppoopegAAoAAAtAtttooepooppteeeeotpopppeeoepooopopgoooepoepotpoegogggggkeggpogopopeAtAttAoApAtttAggeAgegettttAAAAtoeeggeteoAopopotoktttpoepegpgtgAgAeeppeoooopgeggpAAAAgAtgegogoeepAtAtAAotAAAtttAtkAAAtAAktAAAtttAtAAoAtteeoopoAoAtoAAtAAApgeoeeeeoeeegteoAopeAkopgpeAgetAeeotAttAAeAAktttkAptAetAttAkAAAttAAkAAAttAAAAgAgkgogppgooApkpoAopopptotegoAppppAgettgtteAtttAAttAtpoooopopkeogeeettgtAAttAtAtttpopptoAokpopooooAooeoopopptoopgpAAootAtgtpgeeeeegegeAkeAgtoAoAooepgeegegeegekeegtoAAttttttggeegkeegggetgggggeggegeAgpoooktoppoopApoooAtAeAgegegoegeAgpeotppogpoppppoppoppoootAtAAAtApopoopooooopopppopoppoottoopopookAtAAAtettApAtttooAAtteeoAttppeAgtpeegoeeAtoAteeAeeppopekotktetppgpkgktopAAtkkgegttAAtoopopkeAApgoAotteegegeogkoggpAggpkAgAttttAAtAttAteeeopoetAttAtkeoAopgtAtktgtgttopooppgopppppppopeooAptoopopAookApoggtpttttoAoppoopAppoAoppooppptpAooppppppoooAAAtttttttAtteegggeeoeegeoeggkettkAAoAkAAteeggggkAgAtpAAAttAtAAAptAeppAAAopppAApkeeokpeegpppekpegeeeteoopoApoookoogggegekopo'

# Converts the string to integers:
X = np.asarray([ord(i) for i in text]).reshape(-1,1)

# Trains HMM using hmmlearn and gives prediction for speakers
hiddenMM = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100,algorithm="map").fit(X)
speakers = hiddenMM.predict(X)

# I plot 10 plots, one for each 100 strings (for readability) of the predicted speaker.
# This does not output the posterior probabilties of a speaker at each time.
# The process for doing this would require Expectation and Maximization steps for each phoneme.
for i in range(0,1000,100):
    plt.scatter(range(100),speakers[i:i+100])
    plt.show()













## Here is my attempt at the rest:
# See Initialization here:

symbol = 'eoggeggAeggepgpppoppogopppgoppoopegAAoAAAtAtttooepooppteeeeotpopppeeoepooopopgoooepoepotpoegogggggkeggpogopopeAtAttAoApAtttAggeAgegettttAAAAtoeeggeteoAopopotoktttpoepegpgtgAgAeeppeoooopgeggpAAAAgAtgegogoeepAtAtAAotAAAtttAtkAAAtAAktAAAtttAtAAoAtteeoopoAoAtoAAtAAApgeoeeeeoeeegteoAopeAkopgpeAgetAeeotAttAAeAAktttkAptAetAttAkAAAttAAkAAAttAAAAgAgkgogppgooApkpoAopopptotegoAppppAgettgtteAtttAAttAtpoooopopkeogeeettgtAAttAtAtttpopptoAokpopooooAooeoopopptoopgpAAootAtgtpgeeeeegegeAkeAgtoAoAooepgeegegeegekeegtoAAttttttggeegkeegggetgggggeggegeAgpoooktoppoopApoooAtAeAgegegoegeAgpeotppogpoppppoppoppoootAtAAAtApopoopooooopopppopoppoottoopopookAtAAAtettApAtttooAAtteeoAttppeAgtpeegoeeAtoAteeAeeppopekotktetppgpkgktopAAtkkgegttAAtoopopkeAApgoAotteegegeogkoggpAggpkAgAttttAAtAttAteeeopoetAttAtkeoAopgtAtktgtgttopooppgopppppppopeooAptoopopAookApoggtpttttoAoppoopAppoAoppooppptpAooppppppoooAAAtttttttAtteegggeeoeegeoeggkettkAAoAkAAteeggggkAgAtpAAAttAtAAAptAeppAAAopppAApkeeokpeegpppekpegeeeteoopoApoookoogggegekopo'
letters = ['A', 'e', 'g', 'k', 'o','p', 't']
speakers = [0,1,2]
transition = [[0.9,0.05,0.05],[0.05,0.9,0.05],[0.05,0.05,0.9]]
transition = np.asarray(transition)

initial = [float(1)/float(3), float(1)/float(3), float(1)/float(3)]
emission=[]
for i in range(3):
    emission_row = np.random.dirichlet(np.ones(7),size=1)
    emission_row_list = []
    for item in emission_row[0]:
        emission_row_list.append(item)
    emission.append(emission_row_list)
emission = Table(np.asarray(emission), names = letters)


# I would then calculate the liklelihood for each hidden state given the transition and emission matrices.
# P(initial)*P(transition)*P(emission)
# I would use this new probability to calculate the maximum likelihood for each parameter, emission + transition matrix and adjust accordingly.
# I repeat this step until the last step and the model converges.


# Attempt at forward step (alpha)
# I believe I should convert to log-probabilities to avoid setting values to zero. Also I need to be updating
# my transition and emission matrices in the Maximization step.


speakers = [0,1,2]
def alpha(speaker, string, time, transition, emission):
    if time == 0:
        return float(emission[speaker][string[time]])*float(float(1)/float(3))
    else:
        for q in speakers:
            return emission[speaker][string[time]]*np.sum(transition[q][speaker]*alpha(speaker, string, time-1, transition, emission))*[alpha(i ,symbol, time-1, transition, emission) for i in speakers][q]

print alpha(2 ,symbol, 2, transition, emission)
