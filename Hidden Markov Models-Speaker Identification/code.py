#_____#2_____#
### Imports
import glob
import os
import numpy as np
from astropy.table import Table, Column
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
