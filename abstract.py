import numpy as np
import json
import re
from itertools import chain
import jieba
import time
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# 将目标文档进行分句
# print(time.asctime(time.localtime(time.time())))
sentences_list = []
dic = {'abstract':None}
fp_1 = open('test_data.json', 'r', encoding="utf8")
fp_2 = open('test_abstract_data.json', 'w', encoding="utf8")
for line in fp_1.readlines():
    dic = json.loads(line)
    text = dic['text']
    sent = str(text)
    if sent.strip():
        # 把元素按照[。！；？]进行分隔，得到句子。
        sent_split = re.split(r'[。！；？，]', sent.strip())
        sentences_list.append(sent_split)
    sentences_list = list(chain.from_iterable(sentences_list))

# 文本预处理，去除停用词和非汉词语，对分好的每个句子进行分词
    # 创建停用词列表
    stopwords = [line.strip() for line in open('./stopwords.txt', encoding='UTF-8').readlines()]

    # print(time.asctime(time.localtime(time.time())))

    sentence_word_list = []
    for sentence in sentences_list:
        # 去掉非汉字字符
        sentence = re.sub(r'[^\u4e00-\u9fa5]+', '', sentence)
        sentence_depart = jieba.cut(sentence.strip())
        word_list = []
        for word in sentence_depart:
            if word not in stopwords:
                word_list.append(word)
                # 如果句子整个被过滤掉了，如：'02-2717:56'被过滤，那就返回[],保持句子的数量不变
        sentence_word_list.append(word_list)

    # 保证处理后句子的数量不变，我们后面才好根据textrank值取出未处理之前的句子作为摘要。
    # if len(sentences_list) == len(sentence_word_list):
    #     print("\n数据预处理后句子的数量不变！")

# 加载word2vec向量，选取的是人民网的新闻
    word_embeddings = {}
    f = open('./sgns.renmin.char', 'r', encoding='utf-8')
    for line in f:
        # 把第一行的内容去掉
        if '467389 300\n' not in line:
            values = line.split()
            # 第一个元素是词语
            word = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            word_embeddings[word] = embedding
    f.close()

# 得到词语的embedding，用WordAVG作为句子的向量表示
    sentence_vectors = []
    for i in sentence_word_list:
        if len(i) != 0:
            # 如果句子中的词语不在字典中，那就把embedding设为300维元素为0的向量。
            # 得到句子中全部词的词向量后，求平均值，得到句子的向量表示
            v = sum([word_embeddings.get(w, np.zeros((300,))) for w in i]) / (len(i))
        else:
            # 如果句子为[]，那么就向量表示为300维元素为0个向量。
            v = np.zeros((300,))
        sentence_vectors.append(v)

    sim_mat = np.zeros([len(sentences_list), len(sentences_list)])

    for i in range(len(sentences_list)):
        for j in range(len(sentences_list)):
            if i != j:
                sim_mat[i][j] = \
                cosine_similarity(sentence_vectors[i].reshape(1, 300), sentence_vectors[j].reshape(1, 300))[
                    0, 0]

# 迭代得到句子的TEXTRANK值，排序并去除摘要
    # 利用句子相似度矩阵构建图结构，句子为节点，句子相似度为转移概率
    nx_graph = nx.from_numpy_array(sim_mat)

    # 得到所有句子的textrank值
    scores = nx.pagerank(nx_graph)

    # 根据textrank值对未处理的句子进行排序
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences_list)), reverse=True)

    # 取出得分最高的前10个句子作为摘要
    sentences = []
    sn = 7
    for i in range(sn):
        sentences.append(ranked_sentences[i][1])
    dic['abstract'] = sentences
    json.dump(dic, fp_2, ensure_ascii=False)
    fp_2.write('\n')


