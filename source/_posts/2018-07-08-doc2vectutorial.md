---
layout: post
title: Doc2Vec Tutorial
date: 2018-07-08 12:10 +0800
categories: 深度学习
tags:
- Embedding
- 神经网络
mathjax: true
copyright: false
---

目录

* gensim简单使用
* Doc2Vec(PV-DM)
* Doc2Vec(PV-DBOW)


---------

### gensim简单使用

模型参数说明：
1. dm=1 PV-DM  dm=0 PV-DBOW。
2. size 所得向量的维度。
3. window 上下文词语离当前词语的最大距离。
4. alpha 初始学习率，在训练中会下降到min_alpha。
5. min_count 词频小于min_count的词会被忽略。
6. max_vocab_size 最大词汇表size，每一百万词会需要1GB的内存，默认没有限制。
7. sample 下采样比例。
8. iter 在整个语料上的迭代次数(epochs)，推荐10到20。
9. hs=1 hierarchical softmax ，hs=0(default) negative sampling。
10. dm_mean=0(default) 上下文向量取综合，dm_mean=1 上下文向量取均值。
11. dbow_words:1训练词向量，0只训练doc向量。

```python
# coding: utf-8

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

data = ["I love machine learning. Its awesome.",
        "I love coding in python",
        "I love building chatbots",
        "they chat amagingly well"]
tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[i]) for i, _d in enumerate(data)]

# for item in tagged_data:
#     print(item)
# TaggedDocument(['i', 'love', 'machine', 'learning', '.', 'its', 'awesome', '.'], ['0'])
# TaggedDocument(['i', 'love', 'coding', 'in', 'python'], ['1'])
# TaggedDocument(['i', 'love', 'building', 'chatbots'], ['2'])
# TaggedDocument(['they', 'chat', 'amagingly', 'well'], ['3'])

max_epochs = 100
vec_size = 10
alpha = 0.025
model = Doc2Vec(size=vec_size, alpha=alpha, min_alpha=0.00025, min_count=1, dm=1)
model.build_vocab(tagged_data)
for epoch in range(max_epochs):
    print("iteration {0}".format(epoch))
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.iter)
    model.alpha -= 0.0002
    model.min_alpha = model.alpha

model.save("d2v.model")
print("Model Saved!")


model = Doc2Vec.load("d2v.model")
test_data = word_tokenize("I love chatbots".lower())
v1 = model.infer_vector(test_data)
print("V1_infer:", v1, end="\n\n")
similar_doc = model.docvecs.most_similar(1)
print(similar_doc, end="\n\n")

for i in range(len(data)):
    print(i, model.docvecs[i], end="\n\n")

"""
V1_infer: [ 0.07642581  0.11099993 -0.02696064 -0.06895891  0.01907274 -0.08622721
 -0.00581482 -0.08242869 -0.02741096 -0.05718143]

[(0, 0.9964085817337036), (2, 0.9903678894042969), (3, 0.985236406326294)]

0 [ 1.22332275  0.93302435  0.0843188  -0.40848678 -0.26203951 -0.54852372
 -0.07185869 -0.72306669 -0.635378   -0.05744991]

1 [ 0.83533287  0.64208162  0.04288336 -0.28720176 -0.17140444 -0.43564293
 -0.0435797  -0.45327306 -0.46411869 -0.12211297]

2 [ 0.71188128  0.49025729 -0.03114104 -0.2172543  -0.18351653 -0.35265383
 -0.09494802 -0.47745392 -0.33393192 -0.1065111 ]

3 [ 0.86559838  0.77232999 -0.0108105  -0.179581   -0.10455605 -0.41468951
 -0.11108498 -0.59402496 -0.59637135 -0.22117028]
"""
```

> [DOC2VEC gensim tutorial](https://medium.com/@mishra.thedeepak/doc2vec-simple-implementation-example-df2afbbfbad5)
> [使用gensim的doc2vec生成文档向量](https://blog.csdn.net/weixin_39837402/article/details/80254868)

-----------

**需要说明的一点是这里的Paragraph Vector不是真的段落向量的意思，它可以根据需要的不同进行变化，可以是短语、句子甚至是文档。** 

### Doc2Vec(PV-DM)

PV-DM在模型的输入层新增了一个Paragraph id，用于表征输入上下文所在的Paragraph。
例如如果需要训练得到句子向量，那么Paragraph id即为语料库中的每个句子的表示。
Paragraph id其实也是一个向量，具有和词向量一样的维度，但是它们来自不同的向量空间，D和W，也就是来自于两个不同的矩阵。
剩下的思路和CBOW模型基本一样。在模型中值得注意的一点是，在同一个Paragraph中，进行窗口滑动时，Paragraph id是不变的。

Paragraph id本质上就是一个word，只是这个word唯一代表了这个paragraph，丰富了context vector。

- 模型的具体步骤如下：
  - 每个段落都映射到一个唯一的向量，由矩阵$D$中的一列表示，每个词也被映射到一个唯一的向量，表示为$W$ ;
  - 对**当前段落向量**和**当前上下文**所有词向量一起进行取平均值或连接操作，生成的向量用于输入到softmax层，以预测上下文中的下一个词: $$y=b+Uh(w_{t-k}, \dots, w_{t+k}; W; D)$$ 
- 这个段落向量可以被认为是另一个词。可以将它理解为一种记忆单元，记住**当前上下文所缺失的内容**或段落的**主题** ；
- 矩阵$D$ 和$W$ 的区别:
  - 通过当前段落的index，对$D$ 进行Lookup得到的段落向量，对于当前段落的所有上下文是共享的，但是其他段落的上下文并不会影响它的值，也就是说它**不会跨段落(not across paragraphs)** ；
  - 当时词向量矩阵$W$对于所有段落、所有上下文都是共享的。

![pv-dm](/posts_res/2018-07-08-doc2vectutorial/1.png)


--------

### Doc2Vec(PV-DBOW)

模型希望通过输入一个Paragraph id来预测该Paragraph中的单词的概率，和Skip-gram模型非常的类似。

- PV-DBOW模型的输入忽略了的上下文单词，但是关注模型从输出的段落中预测从段落中随机抽取的单词；
- PV-DBOW模型和训练词向量的**Skip-gram模型**非常相似。

![pv-dbow](/posts_res/2018-07-08-doc2vectutorial/2.png)


--------

### Doc2Vec的特点

- 可以从未标记的数据中学习，在没有足够多带标记的数据上仍工作良好；
- 继承了词向量的词的语义（semantics）的特点；
- 会考虑词的顺序（至少在某个小上下文中会考虑）


----------

> 
1. [Distributed representations of sentences and documents](https://arxiv.org/pdf/1405.4053.pdf)
2. [Distributed representations of sentences and documents总结 - paperweekly](https://www.paperweekly.site/papers/notes/135)
3. [用 Doc2Vec 得到文档／段落／句子的向量表达](https://blog.csdn.net/aliceyangxi1987/article/details/75097598)
4. [关于Gensim的初次见面 和 Doc2vec 的模型训练](https://blog.csdn.net/qq_36472696/article/details/77871723)
5. [论文笔记：Distributed Representations of Sentences and Documents](https://github.com/llhthinker/NLP-Papers/blob/master/distributed%20representations/2017-11/Distributed%20Representations%20of%20Sentences%20and%20Documents/note.md)
6. [paragraph2vec介绍](http://d0evi1.com/paragraph2vec/)
