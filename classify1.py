import re
import os
from jieba import cut
from itertools import chain
from collections import Counter
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer


def get_words(filename):
    """读取文本并过滤无效字符和长度为1的词"""
    words = []
    with open(filename, 'r', encoding='utf-8') as fr:
        for line in fr:
            line = line.strip()
            # 过滤无效字符
            line = re.sub(r'[.【】0-9、——。，！~\*]', '', line)
            # 使用jieba.cut()方法对文本切词处理
            line = cut(line)
            # 过滤长度为1的词
            line = filter(lambda word: len(word) > 1, line)
            words.extend(line)
    return words


def get_top_words(top_num):
    """遍历邮件建立词库后返回出现次数最多的词"""
    filename_list = ['邮件_files/{}.txt'.format(i) for i in range(151)]
    all_words = []
    # 遍历邮件建立词库
    for filename in filename_list:
        all_words.append(get_words(filename))
    # itertools.chain()把all_words内的所有列表组合成一个列表
    # collections.Counter()统计词个数
    freq = Counter(chain(*all_words))
    return [i[0] for i in freq.most_common(top_num)]


def extract_features(filename_list, top_words, feature_type='frequency'):
    if feature_type == 'frequency':
        vector = []
        for filename in filename_list:
            words = get_words(filename)
            word_map = list(map(lambda word: words.count(word), top_words))
            vector.append(word_map)
        vector = np.array(vector)
    elif feature_type == 'tfidf':
        texts = []
        for filename in filename_list:
            words = get_words(filename)
            text = " ".join(words)
            texts.append(text)
        # 修改：将lowercase参数设为False
        vectorizer = TfidfVectorizer(vocabulary=top_words, lowercase=False)
        vector = vectorizer.fit_transform(texts).toarray()
    else:
        raise ValueError("Invalid feature type. Choose 'frequency' or 'tfidf'.")
    return vector


top_words = get_top_words(100)
filename_list = ['邮件_files/{}.txt'.format(i) for i in range(151)]
# 可以修改这里的参数来切换特征提取方式
feature_type = 'tfidf'
vector = extract_features(filename_list, top_words, feature_type)

# 0 - 126.txt为垃圾邮件标记为1；127 - 151.txt为普通邮件标记为0
labels = np.array([1] * 127 + [0] * 24)
model = MultinomialNB()
model.fit(vector, labels)


def predict(filename, top_words, model, feature_type='frequency'):
    """对未知邮件分类"""
    if feature_type == 'frequency':
        # 构建未知邮件的词向量
        words = get_words(filename)
        current_vector = np.array(
            tuple(map(lambda word: words.count(word), top_words)))
    elif feature_type == 'tfidf':
        words = get_words(filename)
        text = " ".join(words)
        # 修改：将lowercase参数设为False
        vectorizer = TfidfVectorizer(vocabulary=top_words, lowercase=False)
        vectorizer.fit([" ".join(get_words(f)) for f in filename_list])
        current_vector = vectorizer.transform([text]).toarray().flatten()
    else:
        raise ValueError("Invalid feature type. Choose 'frequency' or 'tfidf'.")
    # 预测结果
    result = model.predict(current_vector.reshape(1, -1))
    return '垃圾邮件' if result == 1 else '普通邮件'


print('151.txt分类情况:{}'.format(predict('邮件_files/151.txt', top_words, model, feature_type)))
print('152.txt分类情况:{}'.format(predict('邮件_files/152.txt', top_words, model, feature_type)))
print('153.txt分类情况:{}'.format(predict('邮件_files/153.txt', top_words, model, feature_type)))
print('154.txt分类情况:{}'.format(predict('邮件_files/154.txt', top_words, model, feature_type)))
print('155.txt分类情况:{}'.format(predict('邮件_files/155.txt', top_words, model, feature_type)))
