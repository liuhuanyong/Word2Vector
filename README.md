#  Word2Vector  
Self complemented word embedding methods using CBOW，skip-Gram，word2doc matrix , word2word matrix ,基于CBOW、skip-gram、词-文档矩阵、词-词矩阵四种方法的词向量生成
# 简介：   
Harris 在1954 年提出的分布假说（distributional hypothesis）为这一设想提供了理论基础：上下文相似的词，其语义也相似。  
Firth 在1957 年对分布假说进行了进一步阐述和明确：词的语义由其上下文决定（a word is characterized by thecompany it keeps）。  
基于分布假说得到的表示均可称为分布表示（distributional representation），它们的核心思想也都由两部分组成：  
1）选择一种方式描述上下文；  
2）选择一种模型刻画某个词（下文称“目标词”）与其上下文之间的关系。  
根据建模的不同，主要可以分为三类：  
1）基于矩阵的分布表示  
这类方法需要构建一个“词-上下文”矩阵，从矩阵中获取词的表示。在“词-上下文”矩阵中，每行对应一个词，每列表示一种不同的上下文，矩阵中的每个元素对应相关词和上下文的共现次数。  
在这种表示下，矩阵中的一行，就成为了对应词的表示，这种表示描述了该词的上下文的分布。  
2）基于聚类的分布表示  
该方法以根据两个词的公共类别判断这两个词的语义相似度。最经典的方法是布朗聚类（Brown clustering）  
3）基于神经网络的分布表示  
基于神经网络的分布表示一般称作词向量、 词嵌入（word embedding）、分布式表示（distributed representation）  
# 项目介绍  
本项目主要实现1）基于矩阵的分布式表示和3）基于神经网络的分布表示两种方法。其中1）包括词-文档共现矩阵，词-词共现矩阵两种 ，2）包括cbow和skipgram两种模型。  
