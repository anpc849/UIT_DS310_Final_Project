from .base import BaseTopicSystem
from models.model.smtopic import SMTopic
import pandas as pd
# from simcse import SimCSE
import gensim.corpora as corpora
# from flair.embeddings import TransformerDocumentEmbeddings
from gensim.models.coherencemodel import CoherenceModel

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from sklearn.feature_extraction.text import TfidfVectorizer

from ._base import BaseEmbedder


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
STOPWORDS = set(stopwords.words('english'))

MIN_WORDS = 4
MAX_WORDS = 200

def tokenizer(self, tokens_sen, min_words=MIN_WORDS, max_words=MAX_WORDS, stopwords=STOPWORDS, lemmatize=True):
    if lemmatize:
        stemmer = WordNetLemmatizer()
        tokens = [stemmer.lemmatize(w) for w in tokens_sen]
    else:
        tokens = [w for w in tokens_sen]
    token = [w for w in tokens if (len(w) > min_words and len(w) < max_words
                                                        and w not in stopwords)]
    return tokens

class SMTopicTM(BaseTopicSystem):
    def __init__(self, dataset, topic_model, num_topics, dim_size, word_select_method, embedding, seed):
        super().__init__(dataset, topic_model, num_topics)
        print(f'Initialize SMTopicTM with num_topics={num_topics}, embedding={embedding}')
        self.dim_size = dim_size
        self.word_select_method = word_select_method
        self.embedding = embedding
        self.seed = seed
        
        # make sentences and token_lists
        token_lists = self.dataset.get_corpus()
        # token_list = tokenizer(token_list)
        self.sentences = [' '.join(text_list) for text_list in token_lists]
        self.sentences = self.sentences[:50000]
        
        # embedding_model = TransformerDocumentEmbeddings(embedding)
        self.model = SMTopic(embedding_model=self.embedding,
                             nr_topics=num_topics, 
                             dim_size=self.dim_size, 
                             word_select_method=self.word_select_method, 
                             seed=self.seed)
    
    
    def train(self):
        self.topics = self.model.fit_transform(self.sentences)
    
    
    def evaluate(self):
        td_score = self._calculate_topic_diversity()
        cv_score, npmi_score = self._calculate_cv_npmi(self.sentences, self.topics)
        
        return td_score, cv_score, npmi_score
    
    
    def get_topics(self):
        return self.model.get_topics()
    
    
    def _calculate_topic_diversity(self):
        topic_keywords = self.model.get_topics()

        bertopic_topics = []
        for k,v in topic_keywords.items():
            temp = []
            for tup in v:
                temp.append(tup[0])
            bertopic_topics.append(temp)  

        unique_words = set()
        for topic in bertopic_topics:
            unique_words = unique_words.union(set(topic[:10]))
        td = len(unique_words) / (10 * len(bertopic_topics))

        return td


    def _calculate_cv_npmi(self, docs, topics): 

        doc = pd.DataFrame({"Document": docs,
                        "ID": range(len(docs)),
                        "Topic": topics})
        documents_per_topic = doc.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
        cleaned_docs = self.model._preprocess_text(documents_per_topic.Document.values)

        vectorizer = self.model.vectorizer_model
        analyzer = vectorizer.build_analyzer()

        tokens = [analyzer(doc) for doc in cleaned_docs]
        dictionary = corpora.Dictionary(tokens)
        corpus = [dictionary.doc2bow(token) for token in tokens]
        topic_words = [[words for words, _ in self.model.get_topic(topic)] 
                    for topic in range(len(set(topics))-1)]

        coherence_model = CoherenceModel(topics=topic_words, 
                                      texts=tokens, 
                                      corpus=corpus,
                                      dictionary=dictionary, 
                                      coherence='c_v')
        cv_coherence = coherence_model.get_coherence()

        coherence_model_npmi = CoherenceModel(topics=topic_words, 
                                      texts=tokens, 
                                      corpus=corpus,
                                      dictionary=dictionary, 
                                      coherence='c_npmi')
        npmi_coherence = coherence_model_npmi.get_coherence()

        return cv_coherence, npmi_coherence 
    