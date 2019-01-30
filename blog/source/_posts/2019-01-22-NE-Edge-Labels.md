---
title: 《Enhanced Network Embeddings via Exploiting Edge Labels》阅读笔记
date: 2019-01-22 11:02:29
categories: [Network Embedding, paper]
tags: [Hawkes process, attention]
mathjax: true
---
论文地址: [Enhanced Network Embeddings via Exploiting Edge Labels](https://arxiv.org/abs/1809.05124?context=physics.soc-ph)

## Introduction
这是DeepWalk团队的一篇论文，目的是捕获网络中的边信息。传统的NE方法通常把节点简单关系当做（0,1）二值，然而边中所包含的丰富的语义信息。本文尝试做Network Embedding的同时保留网络结构和节点关系信息。举个例子来说，真实社交网络中，一个用户可能和他的同事，家人有关系，但是已有的NE方法不能同事捕获好友关系（有边连接），以及边的类型。

具体来说，本分的方法分为无监督部分和监督部分。其中无监督部分预测节点邻域， 监督部分预测边标签。所以本文模型是个**半监督NE模型**。

## Problem Definition
假定Network Graph $G=(V,E)$是无向图。$L=(l_1,l_2,...,l_{|V|})$是边的类型集。一个含有边类型的Graph可以被重新定义为$G=(V,E_L,E_U,Y_L)$,其中$E_L$是由label的边集，$E_U$是没有label的边集，$E_L \cup E_U =E$。$Y_L$表示$E_L$中边的关系类型集合。论文中假定一条边可以有多重关系，所以对于边$E_i$的label集$Y_L(i) \in Y_L$, $Y_L(i)$可能包含很多类型 所以$Y_L(i) \subseteq L$。目的还是一样，学习一个映射函数$\Phi \to \mathbb{R}^{|V| \times d}$, 其中$d \ll |V|$。

## Method
首先定义损失函数:  
$$\mathcal{L}=(1-\lambda)\mathcal{L}_s+\lambda\mathcal{L}_r$$  
其中$\mathcal{L}_s$表示预测节点邻域的损失。$\mathcal{L}_r$表示预测边label的损失。$\lambda$是两种损失的权重。

### Structural Loss
第一部分是最小化无监督网络结构损失，对于一个给定的节点$v$, 要最大化这个节点和它邻域可能性。其中，节点的邻域不一定要一定和该节点有边相连。文章先给出了结构损失的目标函数：  
$$\mathcal{L}_s=-\sum_{u \in C(v)} \log Pr(u|v)$$  
这个函数其实就是给定$v$,最大化$v$的邻域的极大似然。其中$Pr(u|v)$是一个softmax函数：  
$$Pr(u|v)=\frac{\exp(\Phi(u) \cdot \Phi'(v))}{\sum_{u' \in V} \exp(\Phi(u') \cdot \Phi'(v))}$$  
这其实和DeepWalk一样，一个节点$v$有两个表示向量，$\Phi(v)$和$\Phi'(v)$分别表示该节点作为中心节点和上下文节点的表示。由于计算复杂度较高，所以采用负采样的策略。  
剩下的问题就是如何构建节点$v$的邻域$C(v)$。一种直接的方式就是从邻接矩阵中选取他的邻居。然后由于现实网络的稀疏性，一个节点只有很少的邻居。为了缓解网络稀疏性的问题， 本文采取了类似于DeepWalk的randomwalk策略。 最终可以得到节点$v$的邻域：  
$$C(v)=\{v_{i-w},...,v_{i-1}\} \cup \{v_{i+1},...,v_{i+w}\}$$

### Relational Loss
由于label是为了预测边的，所以需要把每条边表示出来，所以对于边$e=(u,v) \in E$,可以用一下方法来表示这条边:  
$$\Phi(e)=g(\Phi(u),\Phi(v))$$  
其中，$g$是一个映射函数用来把两个节点的表示向量转化为他们之间边的表示向量，本文使用了简单的连接操作，即把两个向量直接拼接：  
$$\Phi(e)=\Phi(u) \oplus \Phi(v)$$  
这样我们就获得了edge embedding。直接将它输入前馈神经网络，前馈神经网络第$k$层的定义为:  
$$h^{(k)}=f(W^{(k)}h^{(k-1)}+b^{(k)})$$  
其中 $h^{(0)}=\Phi(e)$，$f$是除最后一层外采用relu激活函数，最后一层采用sigmoid函数激活,最后一层输出为$\hat{y_i}$。最后最小化二元交叉熵损失函数：

$$\mathcal{L}_r=\sum^{|L|}_{i=1} H(y_i,\hat{y_i}) + (1-y_i) \cdot \log (1-\hat{y_i})$$

### Conclusion
这篇论文从原理到方法实现都非常简单，稍后我也将尝试复现这篇论文，边的标签信息是以前NE方法所没有考虑到的，但这篇论问的局限性是没有考虑边的方向以及权重，这是可以拓展的方向。