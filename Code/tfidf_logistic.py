import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

DATA_PATH = '../data/weibo_senti_100k.csv'
STOPWORDS_PATH = '../data/cn_stopwords.txt'


def data_process(data_path=DATA_PATH):
    """
    主要包括中文分词，去除停用词，对于数据集需要进行数据的打乱，
    因为数据前半部分为正面类别数据，后半部分为负面类别数据
    另外对于打乱的数据进行9：1的划分，划分出训练集和测试集
    :param data_path: 数据集文件路径
    :return: train_data, test_data
    """
    data_df = pd.read_csv(data_path, sep=',')
    data_df = data_df.sample(frac=1.0, random_state=1)  # 设置random_state保证可重复
    data_review, data_label = chinese_segmetation(data_df)
    for i in range(20):
        print(data_review[i])
    train_data, test_data = build_data(0.9, data_review, data_label)

    return train_data, test_data


def chinese_segmetation(data_df):
    """
    对于文本语料库进行中文分词
    :param data_df: 打乱过后的原始数据
    :return: 分词之后的review列表和labe了列表
    """
    stopwords_list = [line.strip() for line in open(STOPWORDS_PATH, 'r').readlines()]
    data_review = []
    data_label = []

    for index, item in data_df.iterrows():
        text = item['review']
        label = item['label']
        text_seg = jieba.lcut(text)
        disposed_text = drop_stopwords(text_seg, stopwords_list)
        data_review.append(disposed_text)
        data_label.append(label)

    return data_review, data_label


def drop_stopwords(text: list, stopwords):
    """
    对一个文本进行去除停用词操作
    :param text:分词好的word列表
    :param stopwords: 停用词列表
    :return: 去除stopwords后的文本str
    """
    disposed_text = ""
    for word in text:
        if word not in stopwords:
            disposed_text = disposed_text + " " + word

    return disposed_text


def build_data(div_ratio, reviews_list, labels_list):
    """
    按照比例对数据集进行分割，一部分作为训练集，一部分作为测试集
    :param div_ratio: 划分比例
    :param reviews_list: review列表
    :param label_list: label类别，和review列表一一对应
    :return: train_data and test_data
    """
    division_index = int(len(reviews_list) * div_ratio)
    print("div_index ", division_index)
    train_reviews = reviews_list[:division_index]
    test_reviews = reviews_list[division_index:]
    train_labels = labels_list[:division_index]
    test_labels = labels_list[division_index:]

    train_data = {'review': train_reviews, 'label': train_labels}
    test_data = {'review': test_reviews, 'label': test_labels}

    return train_data, test_data


def tfidf_logisticRegression(train_data: dict, test_data: dict):
    """
    tf-idf + logistic regression 方法对文本进行分类
    :param train_data: 训练集
    :param test_data: 测试集
    :return: None
    """
    train_reviews = train_data['review']
    train_labels = train_data['label']
    test_reviews = test_data['review']
    test_labels = test_data['label']
    data_reviews = []
    data_reviews.extend(train_reviews)
    data_reviews.extend(test_reviews)
    div_index = len(train_reviews)

    for i in range(20):
        print(train_labels[i],train_reviews[i])

    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_features=20000)
    data_reviews = tfidf_vectorizer.fit_transform(data_reviews)

    lr_model = LogisticRegression(multi_class='auto', n_jobs=4,
                                  solver='saga')
    lr_model.fit(data_reviews[:div_index], train_labels)
    # predict the class
    val_pred = lr_model.predict(data_reviews[div_index:])
    print("f1_score: ", f1_score(test_labels, val_pred, average='macro'))
    print("precision_score:", precision_score(test_labels, val_pred))
    print("recall_score:", recall_score(test_labels, val_pred))

    # RidgeClassifier Model
    rc_model = RidgeClassifier(alpha=0.3)
    rc_model.fit(data_reviews[:div_index], train_labels)
    # predict the class
    val_pred = rc_model.predict(data_reviews[div_index:])
    print(f1_score(test_labels, val_pred, average='macro'))


if __name__ == "__main__":
    train_data, test_data = data_process()
    tfidf_logisticRegression(train_data, test_data)
