# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 16:52:08 2017

@author: E601
"""

 # import modules and set up logging
from gensim.models import word2vec
from gensim import models
import logging
def train(inputFile,modelFile):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # load up unzipped corpus from http://mattmahoney.net/dc/text8.zip
    sentences = word2vec.Text8Corpus(inputFile)
    # train the skip-gram model; default window=5
    model = word2vec.Word2Vec(sentences, size=200)
    # ... and some hours later... just as advertised...
     
    # pickle the entire model to disk, so we can load&resume training later
    model.save(modelFile)
def test(modelFile):
    model = models.Word2Vec.load(modelFile)
    # "boy" is to "father" as "girl" is to ...?
    girl_similar=model.most_similar(['girl', 'father'], ['boy'], topn=3)
    print(girl_similar)
    more_examples = ["he his she", "big bigger bad", "going went being"]
    for example in more_examples:
        a, b, x = example.split()
        predicted = model.most_similar([x, b],[a])[0][0]
        print("%s is to %s as %s is to %s" % (a, b, x, predicted))
 
    # which word doesn't go with the others?
    sentence="breakfast food dinner lunch"
    notMatch=model.doesnt_match(sentence.split())
    print(sentence,"中不是同一类的是",notMatch)
    res=model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
    print("man之于king，如woman之于%s" %res[0][0])
if  __name__ == "__main__":
    inputFile='data/English/text_Eng'
    modelFile='data/English/text_Eng200.model.bin'
#    #训练过程
#    train(inputFile,modelFile)
    #测试过程
    test(modelFile)