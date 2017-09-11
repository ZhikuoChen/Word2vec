# -*- coding: utf-8 -*-
'''
使用OpenCC将繁体字转化为简体字
cmd:opencc -i wiki.txt -o wiki_simplified.txt -c t2s.json
训练用时大概一小时
'''
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import jieba
from gensim.corpora import WikiCorpus
from gensim.models import word2vec
import time
#将wiki文件转化为txt文件
def wiki_to_txt(wiki_corpus,wikiToTetFile):
    wiki_corpus = WikiCorpus(wiki_corpus, dictionary={})
    texts_num = 0

    with open(wikiToTetFile,'w',encoding='utf-8') as output:
        for text in wiki_corpus.get_texts():
            output.write(' '.join(text) + '\n')
            texts_num += 1
            if texts_num % 10000 == 0:
               end=time.time()
               print("[", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), "] ", 
                         "已处理%d篇文章,已耗时%d秒" %(texts_num,(end-start)))
def cut(inputFile,stopwordsFile, cutResultFile):
    # jieba custom setting.
    jieba.set_dictionary('dict.txt')
    # load stopwords set
    stopwordset = set()
    with open(stopwordsFile,'r',encoding='utf-8') as sw:
        for line in sw:
            stopwordset.add(line.strip('\n'))

    texts_num = 0
    cutRes = open(cutResultFile,'w',encoding='utf-8')
    with open(inputFile,'r',encoding='utf-8') as content :
        for line in content:
            line = line.strip('\n')
            #利用jieba进行分词
            words = jieba.cut(line, cut_all=False)
            for word in words:
                #删除停用词
                if word not in stopwordset:
                    cutRes.write(word +' ')
            texts_num += 1

            if texts_num % 10000 == 0:
               end=time.time()
               print("[", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), "] ", 
                         "已完成前%d行的分词,已耗时%d秒" %(texts_num,(end-start)))
    cutRes.close()
def train(cutResultFile, modelFile):
    sentences = word2vec.Text8Corpus(cutResultFile)
#word2vec.Word2Vec(sentences=None, size=100, alpha=0.025, window=5, min_count=5,
#max_vocab_size=None, sample=1e-3, seed=1, workers=3, min_alpha=0.0001,sg=0, hs=0, negative=5, 
#cbow_mean=1, hashfxn=hash, iter=5, null_word=0,trim_rule=None, sorted_vocab=1, 
#batch_words=MAX_WORDS_IN_BATCH)本函数用于从“句子”的可迭代初始化模型。sentences为一句子列表,但对于
#大型数据要使用.BrownCorpus或.Text8Corpus预先处理下。size表示是特征向量的维数;alpha表示是初始学习率
#window表示句子中当前和预测词之间的最大距离;min_count表示忽略总频率低于此值的所有单词;
#max_vocab_size表示在词汇建立期间限制RAM大小;`sg`定义训练所使用的算法,sg=0表示使用CBOW;sg=1表示使用skip-gram 
    model = word2vec.Word2Vec(sentences, size=250)
    end=time.time()
    print("已耗时%d秒" %(end-start))
    #保存模型，供以后使用
    model.save(modelFile)
if __name__ == "__main__":
    start=time.time()
    wiki_corpus='data\wiki\zhwiki-latest-pages-articles.xml.bz2'
    wikiToTetFile="data\wiki\wiki.txt"  #此文件中字体有繁体也有简体
    wiki_simplifiedFile="data\wiki\wiki_simplified.txt"  #此文件为将繁体字转化为简体字后的文件
    stopwordsFile='data\wiki\stopwords.txt'
    cutResultFile="data\wiki\wiki_seg.txt"   #此为分好词的文件
    modelFile="data\wiki\wiki250.model.bin"
    #将网页wiki文件转化为.txt文件
#    wiki_to_txt(wiki_corpus,wikiToTetFile)
#    cut(wiki_simplifiedFile,stopwordsFile, cutResultFile)
    train(cutResultFile, modelFile)
