# -*- coding: utf-8 -*-
from gensim import models
def test1(words,model_file):
     #读取训练好的模型
    model = models.Word2Vec.load(model_file)
    word_list = words.split()
    #当输入词汇数等于1时，认为是求该词的相似词 
    if len(word_list) == 1:
    #.most_similar(positive=[], negative=[], topn=10, restrict_vocab=None, indexer=None)
    #此函数用于找到与给定单词最相似的N个单词;positive会对相似度有正向作用;negtive会对相似度有反向作用;
    #topn表示输出的最相似单词的个数;restrict_vocab为一个可选的整数，用于限制搜索最相似值的词汇范围;
    #restrict_vocab = 10000将只检查词汇顺序中的前10000个词的向量。
        res = model.most_similar(word_list[0],topn = 3)
#        #输出该单词的词向量，为一矩阵
#        word_vector=model.word2vec(word_list[0])
#        print(word_vector)
        print("与%s最相似的三个词为:" %word_list[0])
        for item in res:
            print(item[0]+","+str(item[1]))
    #当输入词汇数等于2时，认为是判断两个词的相似度        
    elif len(word_list) == 2:
         #.similarity(w1, w2)用于计算两个单词w1和w2的余弦相似度
         sim = model.similarity(word_list[0],word_list[1])
         print("%s和%s的Cosine相似度为：%f" %(word_list[0],word_list[1],sim))
    elif len(word_list) == 3:
         res = model.most_similar(positive=[word_list[0],word_list[1]],
                      negative=[word_list[2]], topn= 1)
         print("%s之于%s，如%s之于%s" % (word_list[2],word_list[1],word_list[0],res[0][0]))
    #当输入词汇数大于3时，认为是选择不匹配的选项
    else:
         notMatch=model.doesnt_match(word_list)
         print(word_list,"中不是同一类的是:",notMatch)
if  __name__ == "__main__":
    modelFile="data\wiki\wiki250.model.bin"
    test1('谷歌',modelFile)
    test1('苹果 iphone',modelFile)
    test1('新加坡 欧洲 瑞士',modelFile)
    test1('纽约 中国 上海',modelFile)
    test1('马化腾 百度 李彦宏',modelFile)
    test1('总统 首相 主席 总经理 总理',modelFile)
