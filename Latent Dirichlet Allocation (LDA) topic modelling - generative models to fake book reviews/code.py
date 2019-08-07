###PCW #2
import numpy as np
from collections import OrderedDict
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from nltk.corpus import stopwords
import string




### Use Python to massage the data into a suitable format for
## processing by the Latent Dirichlet Allocation (LDA) model contained in Scikit.learn.
## This will include removing stop words and punctuation.
## Breaking into chapters

headings = ['Chapter 1', 'Chapter 2',
            'Chapter 3', 'Chapter 4', 'Chapter 5',
            'Chapter 6', 'Chapter 7', 'Chapter 8',
            'Chapter 9', 'Chapter 10', 'Chapter 11',
            'Chapter 12','Chapter 13','Chapter 14',
            'Chapter 15','Chapter 16','Chapter 17',]

terminate_code = "THE END"


def extract_chapters(path_to_book): #this function is adapted from Skye

    with open(path_to_book, 'r') as book:
        t = book.readlines()

    # We're going to put all the
    # sections/chapters in a dictionary
    chapter_dict = OrderedDict()

    # Initialize empty sections
    chapter_text = ''
    chapter_name = None

    for line in t:

        if not chapter_name and any([heading in line for heading in headings]):
            chapter_name = line.replace('\r\n', '')

        if chapter_name:

            # Populate section string with line
            if not any([heading in line for heading in headings]):
                chapter_text += line

            # If a heading line, we've hit a new
            # chapter/section; throw the text string into
            # the dictionary and start a new section
            if any([heading in line for heading in headings]):
                chapter_dict[chapter_name] = [chapter_text]
                chapter_name = line.replace('\r\n', '')
                chapter_text = ''

            # If we hit the end of the book,
            # throw everything into last chapter
            if terminate_code in line:
                chapter_dict[chapter_name] = [chapter_text]

    # Make dataframe to store chapters
    df = pd.DataFrame.from_dict(chapter_dict, orient='index')
    df.columns = ['raw_text']

    return df

#Clean up the text!

df = extract_chapters('pan.txt')

# Establish stopwords
sw = set(stopwords.words('english'))

# Remove punctuation & make lowercase
df['no_punctuation'] = df.raw_text.map(
    lambda x: x.translate(None, string.punctuation).lower())

# Remove stopwords
df['no_stopwords'] = df.no_punctuation.map(
    lambda x: [word for word in x.split() if word not in sw])

df['no_stopwords'] = df.no_stopwords.map(
    lambda x: " ".join(x))


df.head()

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

tf_vectorizer = CountVectorizer(max_df=0.95, min_df=0.1,stop_words='english')
tf = tf_vectorizer.fit_transform(df['no_stopwords'])
lda = LatentDirichletAllocation(n_components=10, max_iter=100, learning_method='online',learning_offset=5.,random_state=0)

### Train an LDA model on the corpus.
lda.fit(tf)
tf_feature_names = tf_vectorizer.get_feature_names()

###  Be sure to explain your choice of parameters for any parameters
###  that might have a significant effect on the model results.
'''
1. I removed n_features which set my 'max features' arugment in my count vectorizer
which 'builds a vocabulary that only consider the top max_features ordered by term frequency across the corpus.' assuming
that maybe there will be words that we want to consider that aren't used very much (although this is generally
negligible because the top 2000 words will usually have most influence).

2. I increased max_iter in LDA model from default 10 to 100 to increase number of iterations because the runtime was
still manageable.

3. I lowered learning offset to 5 from default 10 to allow for more influence from early iterations of the online method.

4. My max_df parameter in CountVectorizer is .95 instead of default
1 to  ignore terms that have a document frequency strictly higher than .95.

5. My min_df parameter in CountVectorizer is .1 instead of default 1 to ignore terms that have a
document frequency strictly lower than .1 as opposed to 1 to include more.

'''

### Print out the first ten words of the ten most common topics.
print_top_words(lda, tf_feature_names, 10)


### Now write a single paragraph on the themes contained in the book!
'''
Peter Pan is a fun children's book with main characters Hook, Peter and Wendy. Other notable characters include John,
Darling, Mother, Nana and Michael and George. The main themes in this book revolve around children,
crying, surprises, and dying. There are boats involved with captains of the boats.
Wendy and Peter always appear together. The pirates and the children come together. There are animals like birds nest
and their eggs.




OUTPUT:
Topic #0: hook peter wendy said cried boys moment john mother pirates
Topic #1: peter said wendy cried hook john time boys little mother
Topic #2: said peter wendy hook cried boys little time don john
Topic #3: wendy hook peter way saw water captain die island mother
Topic #4: peter said wendy hook john darling cried children mother michael
Topic #5: kennel ship beds home window head george inside peeped aired
Topic #6: hook peter children cried boys slightly form man pirate like
Topic #7: peter said wendy john hook long time way little quite
Topic #8: darling mrs wendy nana said michael mr children john mother
Topic #9: bird nest peter eggs hat lagoon piece paper hung shut
'''
