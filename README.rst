.. image:: _img/mainpage/logo.gif

###################################################
Deep Learning for Natural Language Processing
###################################################

.. image:: https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat
    :target: https://github.com/astorfi/Deep-Learning-NLP/pulls
.. image:: https://badges.frapsoft.com/os/v2/open-source.png?v=103
    :target: https://github.com/ellerbrock/open-source-badge/
.. image:: https://img.shields.io/badge/Made%20with-Python-1f425f.svg
      :target: https://www.python.org/
.. image:: https://img.shields.io/pypi/l/ansicolortags.svg
      :target: https://github.com/astorfi/Deep-Learning-NLP/blob/master/LICENSE
.. image:: https://img.shields.io/github/contributors/Naereen/StrapDown.js.svg
      :target: https://github.com/astorfi/Deep-Learning-NLP/graphs/contributors



##################
Table of Contents
##################
.. contents::
  :local:
  :depth: 4

***************
Introduction
***************

The purpose of this project is to introduce a shortcut to developers and researcher
for finding useful resources about Deep Learning for Natural Language Processing.

============
Motivation
============

There are different motivations for this open source project.

.. --------------------
.. Why Deep Learning?
.. --------------------

------------------------------------------------------------
What's the point of this open source project?
------------------------------------------------------------

There other similar repositories similar to this repository and are very
comprehensive and useful and to be honest they made me ponder if there is
a necessity for this repository!

**The point of this repository is that the resources are being targeted**. The organization
of the resources is such that the user can easily find the things he/she is looking for.
We divided the resources to a large number of categories that in the beginning one may
have a headache!!! However, if someone knows what is being located, it is very easy to find the most related resources.
Even if someone doesn't know what to look for, in the beginning, the general resources have
been provided.


.. ================================================
.. How to make the most of this effort
.. ================================================

************
Papers
************

.. image:: _img/mainpage/article.jpeg

This chapter is associated with the papers published in NLP using deep learning.

====================
Data Representation
====================

-----------------------
One-hot representation
-----------------------

.. For continuous lines, the lines must be start from the same locations.
* **Character-level convolutional networks for text classification** :
  Promising results by the use of one-hot encoding possibly due to their character-level information.
  [`Paper link <http://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classifica>`_ ,
  `Torch implementation <https://github.com/zhangxiangxiao/Crepe>`_ ,
  `TensorFlow implementation <https://github.com/mhjabreel/CharCNN>`_ ,
  `Pytorch implementation <https://github.com/srviest/char-cnn-pytorch>`_]

  .. image:: _img/mainpage/progress-overall-80.png

.. @inproceedings{zhang2015character,
..   title={Character-level convolutional networks for text classification},
..   author={Zhang, Xiang and Zhao, Junbo and LeCun, Yann},
..   booktitle={Advances in neural information processing systems},
..   pages={649--657},
..   year={2015}
.. }

.. ################################################################################


.. ################################################################################

.. For continuous lines, the lines must be start from the same locations.
* **Effective Use of Word Order for Text Categorization with Convolutional Neural Networks** :
  Exploiting the 1D structure (namely, word order) of text data for prediction.
  [`Paper link <https://arxiv.org/abs/1412.1058>`_ ,
  `Code implementation <https://github.com/riejohnson/ConText>`_]

  .. image:: _img/mainpage/progress-overall-60.png

.. @article{johnson2014effective,
..   title={Effective use of word order for text categorization with convolutional neural networks},
..   author={Johnson, Rie and Zhang, Tong},
..   journal={arXiv preprint arXiv:1412.1058},
..   year={2014}
.. }

.. ################################################################################


.. ################################################################################
.. For continuous lines, the lines must be start from the same locations.
* **Neural Responding Machine for Short-Text Conversation** :
  Neural Responding Machine has been proposed to generate content-wise appropriate responses to input text.
  [`Paper link <https://arxiv.org/abs/1503.02364>`_ ,
  `Paper summary <https://isaacchanghau.github.io/2017/07/19/Neural-Responding-Machine-for-Short-Text-Conversation/>`_]

  .. image:: _img/mainpage/progress-overall-60.png

.. Please add bibtex here
.. @article{shang2015neural,
..   title={Neural responding machine for short-text conversation},
..   author={Shang, Lifeng and Lu, Zhengdong and Li, Hang},
..   journal={arXiv preprint arXiv:1503.02364},
..   year={2015}
.. }

.. ################################################################################


------------------------------
Continuous Bag of Words (CBOW)
------------------------------

.. ################################################################################
.. For continuous lines, the lines must be start from the same locations.
* **Distributed Representations of Words and Phrases and their Compositionality** :
  Not necessarily about CBOWs but the techniques represented in this paper
  can be used for training the continuous bag-of-words model.
  [`Paper link <http://papers.nips.cc/paper/5021-distributed-representations-of-words-andphrases>`_ ,
  `Code implementation 1 <https://code.google.com/archive/p/word2vec/>`_,
  `Code implementation 2 <https://github.com/deborausujono/word2vecpy>`_]


  .. image:: _img/mainpage/progress-overall-100.png

  .. @inproceedings{mikolov2013distributed,
  ..   title={Distributed representations of words and phrases and their compositionality},
  ..   author={Mikolov, Tomas and Sutskever, Ilya and Chen, Kai and Corrado, Greg S and Dean, Jeff},
  ..   booktitle={Advances in neural information processing systems},
  ..   pages={3111--3119},
  ..   year={2013}
  .. }

.. ################################################################################


---------------------
Word-Level Embedding
---------------------

.. ################################################################################
.. For continuous lines, the lines must be start from the same locations.
* **Efficient Estimation of Word Representations in Vector Space** :
  Two novel model architectures for computing continuous vector representations of words.
  [`Paper link <https://arxiv.org/abs/1301.3781>`_ ,
  `Official code implementation <https://code.google.com/archive/p/word2vec/>`_]

  .. image:: _img/mainpage/progress-overall-100.png

  .. @article{mikolov2013efficient,
  ..   title={Efficient estimation of word representations in vector space},
  ..   author={Mikolov, Tomas and Chen, Kai and Corrado, Greg and Dean, Jeffrey},
  ..   journal={arXiv preprint arXiv:1301.3781},
  ..   year={2013}
  .. }

.. ################################################################################

.. ################################################################################
.. For continuous lines, the lines must be start from the same locations.
* **GloVe: Global Vectors for Word Representation** :
  Combines the advantages of the two major models of global matrix
  factorization and local context window methods and efficiently leverages
  the statistical information of the content.
  [`Paper link <http://www.aclweb.org/anthology/D14-1162>`_ ,
  `Official code implementation <https://github.com/stanfordnlp/GloVe>`_]

  .. image:: _img/mainpage/progress-overall-100.png

  .. @inproceedings{pennington2014glove,
  ..   title={Glove: Global vectors for word representation},
  ..   author={Pennington, Jeffrey and Socher, Richard and Manning, Christopher},
  ..   booktitle={Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP)},
  ..   pages={1532--1543},
  ..   year={2014}
  .. }

.. ################################################################################

.. ################################################################################
.. For continuous lines, the lines must be start from the same locations.
* **Skip-Thought Vectors** :
  Skip-thought model applies word2vec at the sentence-level.
  [`Paper <http://papers.nips.cc/paper/5950-skip-thought-vectors>`_ ,
  `Code implementation <https://github.com/ryankiros/skip-thoughts>`_,
  `TensorFlow implementation <https://github.com/tensorflow/models/tree/master/research/skip_thoughts>`_]

  .. image:: _img/mainpage/progress-overall-100.png

  .. @inproceedings{kiros2015skip,
  ..   title={Skip-thought vectors},
  ..   author={Kiros, Ryan and Zhu, Yukun and Salakhutdinov, Ruslan R and Zemel, Richard and Urtasun, Raquel and Torralba, Antonio and Fidler, Sanja},
  ..   booktitle={Advances in neural information processing systems},
  ..   pages={3294--3302},
  ..   year={2015}
  .. }

.. ################################################################################

-------------------------
Character-Level Embedding
-------------------------

.. ################################################################################
.. For continuous lines, the lines must be start from the same locations.
* **Learning Character-level Representations for Part-of-Speech Tagging** :
  CNNs have successfully been utilized for learning character-level embedding.
  [`Paper link <http://proceedings.mlr.press/v32/santos14.pdf>`_ ]

  .. image:: _img/mainpage/progress-overall-60.png

  .. @inproceedings{santos2014learning,
  ..   title={Learning character-level representations for part-of-speech tagging},
  ..   author={Santos, Cicero D and Zadrozny, Bianca},
  ..   booktitle={Proceedings of the 31st International Conference on Machine Learning (ICML-14)},
  ..   pages={1818--1826},
  ..   year={2014}
  .. }

.. ################################################################################

.. ################################################################################
.. For continuous lines, the lines must be start from the same locations.
* **Deep Convolutional Neural Networks forSentiment Analysis of Short Texts** :
  A new deep convolutional neural network has been proposed for exploiting
  the character- to sentence-level information for sentiment analysis application on short texts.
  [`Paper link <http://www.aclweb.org/anthology/C14-1008>`_ ]

  .. image:: _img/mainpage/progress-overall-80.png

  .. @inproceedings{dos2014deep,
  ..   title={Deep convolutional neural networks for sentiment analysis of short texts},
  ..   author={dos Santos, Cicero and Gatti, Maira},
  ..   booktitle={Proceedings of COLING 2014, the 25th International Conference on Computational Linguistics: Technical Papers},
  ..   pages={69--78},
  ..   year={2014}
  .. }

.. ################################################################################

.. ################################################################################
.. For continuous lines, the lines must be start from the same locations.
* **Finding Function in Form: Compositional Character Models for Open Vocabulary Word Representation** :
  The usage of two LSTMs operate over the char-
  acters for generating the word embedding
  [`Paper link <https://arxiv.org/abs/1508.02096>`_ ]

  .. image:: _img/mainpage/progress-overall-60.png

  .. @article{ling2015finding,
  ..   title={Finding function in form: Compositional character models for open vocabulary word representation},
  ..   author={Ling, Wang and Lu{\'\i}s, Tiago and Marujo, Lu{\'\i}s and Astudillo, Ram{\'o}n Fernandez and Amir, Silvio and Dyer, Chris and Black, Alan W and Trancoso, Isabel},
  ..   journal={arXiv preprint arXiv:1508.02096},
  ..   year={2015}
  .. }

.. ################################################################################

.. ################################################################################
.. For continuous lines, the lines must be start from the same locations.
* **Improved Transition-Based Parsing by Modeling Characters instead of Words with LSTMs** :
  The effectiveness of modeling characters for dependency parsing.
  [`Paper link <https://arxiv.org/abs/1508.00657>`_ ]

  .. image:: _img/mainpage/progress-overall-40.png

  .. @article{ballesteros2015improved,
  ..   title={Improved transition-based parsing by modeling characters instead of words with lstms},
  ..   author={Ballesteros, Miguel and Dyer, Chris and Smith, Noah A},
  ..   journal={arXiv preprint arXiv:1508.00657},
  ..   year={2015}
  .. }

.. ################################################################################





====================
Applications
====================

-----------------------
Part-Of-Speech Tagging
-----------------------

.. ################################################################################
.. For continuous lines, the lines must be start from the same locations.
* **Learning Character-level Representations for Part-of-Speech Tagging** :
  A deep neural network (DNN) architecture that joins word-level and character-level representations to perform POS taggin
  [`Paper <http://proceedings.mlr.press/v32/santos14.pdf>`_]

  .. image:: _img/mainpage/progress-overall-100.png


* **Bidirectional LSTM-CRF Models for Sequence Tagging** :
  A variety of neural network based models haves been proposed for sequence tagging task.
  [`Paper <https://arxiv.org/abs/1508.01991>`_,
  `Code Implementation 1 <https://github.com/Hironsan/anago>`_,
  `Code Implementation 2 <https://github.com/UKPLab/emnlp2017-bilstm-cnn-crf>`_]



  .. image:: _img/mainpage/progress-overall-80.png


* **Globally Normalized Transition-Based Neural Networks** :
  Transition-based neural network model for part-of-speech tagging.
  [`Paper <https://arxiv.org/abs/1603.06042>`_]

  .. image:: _img/mainpage/progress-overall-80.png



-----------------------
Parsing
-----------------------

.. ################################################################################
.. For continuous lines, the lines must be start from the same locations.


* **A fast and accurate dependency parser using neural networks** :
  A novel way of learning a neural network classifier for use in a greedy, transition-based dependency parser.
  [`Paper <http://www.aclweb.org/anthology/D14-1082>`_,
  `Code Implementation 1 <https://github.com/akjindal53244/dependency_parsing_tf>`_]

  .. image:: _img/mainpage/progress-overall-100.png


* **Simple and Accurate Dependency Parsing Using Bidirectional LSTM Feature Representations** :
  A simple and effective scheme for dependency parsing which is based on bidirectional-LSTMs.
  [`Paper <https://arxiv.org/abs/1603.04351>`_]

  .. image:: _img/mainpage/progress-overall-60.png

* **Transition-Based Dependency Parsing with Stack Long Short-Term Memory** :
  A technique for learning representations of parser states in transition-based dependency parsers.
  [`Paper <https://arxiv.org/abs/1505.08075>`_]

  .. image:: _img/mainpage/progress-overall-80.png


* **Deep Biaffine Attention for Neural Dependency Parsing** :
  Using neural attention in a simple graph-based dependency parser.
  [`Paper <https://arxiv.org/abs/1611.01734>`_]

  .. image:: _img/mainpage/progress-overall-20.png

* **Joint RNN-Based Greedy Parsing and Word Composition** :
  A greedy parser based on neural networks, which leverages a new compositional sub-tree representation.
  [`Paper <https://arxiv.org/abs/1412.7028>`_]

  .. image:: _img/mainpage/progress-overall-20.png


-------------------------
Named Entity Recognition
-------------------------


* **Neural Architectures for Named Entity Recognition** :
  Bidirectional LSTMs and conditional random fields for NER.
  [`Paper <https://arxiv.org/abs/1603.01360>`_]

  .. image:: _img/mainpage/progress-overall-100.png

* **Boosting named entity recognition with neural character embeddings** :
  A language-independent NER system that uses automatically learned features.
  [`Paper <https://arxiv.org/abs/1505.05008>`_]

  .. image:: _img/mainpage/progress-overall-60.png

* **Named Entity Recognition with Bidirectional LSTM-CNNs** :
  A novel neural network architecture that automatically detects word- and character-level features.
  [`Paper <https://arxiv.org/abs/1511.08308>`_]

  .. image:: _img/mainpage/progress-overall-80.png


-------------------------
Semantic Role Labeling
-------------------------

* **End-to-end learning of semantic role labeling using recurrent neural networks** :
  The use of deep bi-directional recurrent network as an end-to-end system for SRL.
  [`Paper <http://www.aclweb.org/anthology/P15-1109>`_]

  .. image:: _img/mainpage/progress-overall-60.png


--------------------
Text classification
--------------------

.. ################################################################################
.. For continuous lines, the lines must be start from the same locations.
* **Convolutional Neural Networks for Sentence Classification** :
  By training the model on top of the pretrained word-vectors through finetuning, considerable improvement has been reported for learning task-specific vectors.
  [`Paper link <https://arxiv.org/abs/1408.5882>`_ ,
  `Code implementation 1 <https://github.com/yoonkim/CNN_sentence>`_,
  `Code implementation 2 <https://github.com/abhaikollara/CNN-Sentence-Classification>`_,
  `Code implementation 3 <https://github.com/Shawn1993/cnn-text-classification-pytorch>`_,
  `Code implementation 4 <https://github.com/mangate/ConvNetSent>`_]

  .. image:: _img/mainpage/progress-overall-100.png


  .. @article{kim2014convolutional,
  ..   title={Convolutional neural networks for sentence classification},
  ..   author={Kim, Yoon},
  ..   journal={arXiv preprint arXiv:1408.5882},
  ..   year={2014}
  .. }

.. ################################################################################



.. ################################################################################
.. For continuous lines, the lines must be start from the same locations.
* **A Convolutional Neural Network for Modelling Sentences** :
  Dynamic Convolutional Neural Network (DCNN) architecture, which technically is the CNN with a dynamic
  k-max pooling method, has been proposed for capturing the semantic modeling of the sentences.
  [`Paper link <https://arxiv.org/abs/1404.2188>`_ ,
  `Code implementation <https://github.com/FredericGodin/DynamicCNN>`_]

  .. image:: _img/mainpage/progress-overall-80.png

  .. @article{kalchbrenner2014convolutional,
  ..   title={A convolutional neural network for modelling sentences},
  ..   author={Kalchbrenner, Nal and Grefenstette, Edward and Blunsom, Phil},
  ..   journal={arXiv preprint arXiv:1404.2188},
  ..   year={2014}
  .. }

.. ################################################################################



.. ################################################################################
.. For continuous lines, the lines must be start from the same locations.
* **Very Deep Convolutional Networks for Text Classification** :
  The Very Deep Convolutional Neural
  Networks (VDCNNs) has been presented and employed at
  character-level with the demonstration of the effectiveness of
  the network depth on classification tasks
  [`Paper link <http://www.aclweb.org/anthology/E17-1104>`_ ]

  .. image:: _img/mainpage/progress-overall-20.png

  .. @inproceedings{conneau2017very,
  ..   title={Very deep convolutional networks for text classification},
  ..   author={Conneau, Alexis and Schwenk, Holger and Barrault, Lo{\"\i}c and Lecun, Yann},
  ..   booktitle={Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 1, Long Papers},
  ..   volume={1},
  ..   pages={1107--1116},
  ..   year={2017}
  .. }

.. ################################################################################


.. ################################################################################

* **Character-level convolutional networks for text classification** :
  The character-level
  representation using CNNs investigated which argues
  the power of CNNs as well as character-level representation for
  language-agnostic text classification.
  [`Paper link <http://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classifica>`_ ,
  `Torch implementation <https://github.com/zhangxiangxiao/Crepe>`_ ,
  `TensorFlow implementation <https://github.com/mhjabreel/CharCNN>`_ ,
  `Pytorch implementation <https://github.com/srviest/char-cnn-pytorch>`_]

  .. image:: _img/mainpage/progress-overall-80.png

  .. @inproceedings{zhang2015character,
  ..   title={Character-level convolutional networks for text classification},
  ..   author={Zhang, Xiang and Zhao, Junbo and LeCun, Yann},
  ..   booktitle={Advances in neural information processing systems},
  ..   pages={649--657},
  ..   year={2015}
  .. }

.. ################################################################################


.. ################################################################################

* **Multichannel Variable-Size Convolution for Sentence Classification** :
  Multichannel Variable Size Convolutional Neural Network (MV-CNN) architecture
  Combines different version of word-embeddings in addition to
  employing variable-size convolutional filters and is proposed
  in this paper for sentence classification.
  [`Paper link <https://arxiv.org/abs/1603.04513>`_]

  .. image:: _img/mainpage/progress-overall-20.png

  .. @article{yin2016multichannel,
  ..   title={Multichannel variable-size convolution for sentence classification},
  ..   author={Yin, Wenpeng and Sch{\"u}tze, Hinrich},
  ..   journal={arXiv preprint arXiv:1603.04513},
  ..   year={2016}
  .. }

.. ################################################################################


.. ################################################################################

* **A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification** :
  A practical sensitivity analysis of CNNs for exploring the effect
  of architecture on the performance, has been investigated in this paper.
  [`Paper link <https://arxiv.org/abs/1510.03820>`_]

  .. image:: _img/mainpage/progress-overall-60.png

  .. @article{zhang2015sensitivity,
  ..   title={A sensitivity analysis of (and practitioners' guide to) convolutional neural networks for sentence classification},
  ..   author={Zhang, Ye and Wallace, Byron},
  ..   journal={arXiv preprint arXiv:1510.03820},
  ..   year={2015}
  .. }

.. ################################################################################


* **Generative and Discriminative Text Classification with Recurrent Neural Networks** :
  RNN-based discriminative and generative models have been investigated for
  text classification and their robustness to the data distribution shifts has been
  claimed as well.
  [`Paper link <https://arxiv.org/abs/1703.01898>`_]

  .. image:: _img/mainpage/progress-overall-20.png

  .. @article{yogatama2017generative,
  ..   title={Generative and discriminative text classification with recurrent neural networks},
  ..   author={Yogatama, Dani and Dyer, Chris and Ling, Wang and Blunsom, Phil},
  ..   journal={arXiv preprint arXiv:1703.01898},
  ..   year={2017}
  .. }

.. ################################################################################


.. ################################################################################


* **Deep sentence embedding using long short-term memory networks: Analysis and application to information retrieval** :
  An LSTM-RNN architecture has been utilized
  for sentence embedding with special superiority in
  a defined web search task.
  [`Paper link <https://dl.acm.org/citation.cfm?id=2992457>`_]

  .. image:: _img/mainpage/progress-overall-60.png

  .. .. image:: _img/mainpage/progress-overall-20.png
  ..
  .. @article{palangi2016deep,
  ..   title={Deep sentence embedding using long short-term memory networks: Analysis and application to information retrieval},
  ..   author={Palangi, Hamid and Deng, Li and Shen, Yelong and Gao, Jianfeng and He, Xiaodong and Chen, Jianshu and Song, Xinying and Ward, Rabab},
  ..   journal={IEEE/ACM Transactions on Audio, Speech and Language Processing (TASLP)},
  ..   volume={24},
  ..   number={4},
  ..   pages={694--707},
  ..   year={2016},
  ..   publisher={IEEE Press}
  .. }

.. ################################################################################


* **Hierarchical attention networks for document classification** :
  Hierarchical
  Attention Network (HAN) has been presented and utilized to
  capture the hierarchical structure of the text by two word-
  level and sentence-level attention mechanism.
  [`Paper link <http://www.aclweb.org/anthology/N16-1174>`_ ,
  `Code implementation 1 <https://github.com/richliao/textClassifier>`_ ,
  `Code implementation 2 <https://github.com/ematvey/hierarchical-attention-networks>`_ ,
  `Code implementation 3 <https://github.com/EdGENetworks/attention-networks-for-classification>`_,
  `Summary 1 <https://richliao.github.io/supervised/classification/2016/12/26/textclassifier-HATN/>`_,
  `Summary 2 <https://medium.com/@sharaf/a-paper-a-day-25-hierarchical-attention-networks-for-document-classification-dd76ba88f176>`_]

  .. image:: _img/mainpage/progress-overall-80.png

  .. @inproceedings{yang2016hierarchical,
  ..   title={Hierarchical attention networks for document classification},
  ..   author={Yang, Zichao and Yang, Diyi and Dyer, Chris and He, Xiaodong and Smola, Alex and Hovy, Eduard},
  ..   booktitle={Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  ..   pages={1480--1489},
  ..   year={2016}
  .. }

.. ################################################################################


.. ################################################################################


* **Recurrent Convolutional Neural Networks for Text Classification** :
  The combination of both RNNs and CNNs is used for text classification which technically
  is a recurrent architecture in addition to max-pooling with
  an effective word representation method and demonstrates
  superiority compared to simple windows-based neural network
  approaches.
  [`Paper link <http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9745/9552>`_ ,
  `Code implementation 1 <https://github.com/airalcorn2/Recurrent-Convolutional-Neural-Network-Text-Classifier>`_ ,
  `Code implementation 2 <https://github.com/knok/rcnn-text-classification>`_ ,
  `Summary <https://medium.com/paper-club/recurrent-convolutional-neural-networks-for-text-classification-107020765e52>`_]

  .. image:: _img/mainpage/progress-overall-60.png

  .. @inproceedings{lai2015recurrent,
  ..   title={Recurrent Convolutional Neural Networks for Text Classification.},
  ..   author={Lai, Siwei and Xu, Liheng and Liu, Kang and Zhao, Jun},
  ..   booktitle={AAAI},
  ..   volume={333},
  ..   pages={2267--2273},
  ..   year={2015}
  .. }

.. ################################################################################

* **A C-LSTM Neural Network for Text Classification** :
  A unified architecture proposed for sentence and document modeling for classification.
  [`Paper link <https://arxiv.org/abs/1511.08630>`_ ]

  .. image:: _img/mainpage/progress-overall-20.png

  .. @article{zhou2015c,
  ..   title={A C-LSTM neural network for text classification},
  ..   author={Zhou, Chunting and Sun, Chonglin and Liu, Zhiyuan and Lau, Francis},
  ..   journal={arXiv preprint arXiv:1511.08630},
  ..   year={2015}
  .. }

.. ################################################################################

--------------------
Sentiment Analysis
--------------------

.. ################################################################################
.. For continuous lines, the lines must be start from the same locations.
* **Domain adaptation for large-scale sentiment classification: A deep learning approach** :
  A deep learning approach which learns to extract a meaningful representation for each online review.
  [`Paper link <http://www.iro.umontreal.ca/~lisa/bib/pub_subject/language/pointeurs/ICML2011_sentiment.pdf>`_]

  .. image:: _img/mainpage/progress-overall-80.png


  .. @inproceedings{glorot2011domain,
  ..   title={Domain adaptation for large-scale sentiment classification: A deep learning approach},
  ..   author={Glorot, Xavier and Bordes, Antoine and Bengio, Yoshua},
  ..   booktitle={Proceedings of the 28th international conference on machine learning (ICML-11)},
  ..   pages={513--520},
  ..   year={2011}
  .. }

* **Sentiment analysis: Capturing favorability using natural language processing** :
  A sentiment analysis approach to extract sentiments associated with polarities of positive or negative for specific subjects from a document.
  [`Paper link <https://dl.acm.org/citation.cfm?id=945658>`_]

  .. image:: _img/mainpage/progress-overall-80.png


  .. @inproceedings{nasukawa2003sentiment,
  ..   title={Sentiment analysis: Capturing favorability using natural language processing},
  ..   author={Nasukawa, Tetsuya and Yi, Jeonghee},
  ..   booktitle={Proceedings of the 2nd international conference on Knowledge capture},
  ..   pages={70--77},
  ..   year={2003},
  ..   organization={ACM}
  .. }


* **Document-level sentiment classification: An empirical comparison between SVM and ANN** :
  A comparison study. [`Paper link <https://dl.acm.org/citation.cfm?id=945658>`_]

  .. image:: _img/mainpage/progress-overall-60.png


  .. @article{moraes2013document,
  ..   title={Document-level sentiment classification: An empirical comparison between SVM and ANN},
  ..   author={Moraes, Rodrigo and Valiati, Jo{\~a}O Francisco and Neto, Wilson P Gavi{\~a}O},
  ..   journal={Expert Systems with Applications},
  ..   volume={40},
  ..   number={2},
  ..   pages={621--633},
  ..   year={2013},
  ..   publisher={Elsevier}
  .. }

* **Learning semantic representations of users and products for document level sentiment classification** :
  Incorporating of user- and product- level information into a neural network approach for document level sentiment classification.
  [`Paper <http://www.aclweb.org/anthology/P15-1098>`_]

  .. image:: _img/mainpage/progress-overall-40.png


* **Document modeling with gated recurrent neural network for sentiment classification** :
  A a neural network model has been proposed to learn vector-based document representation.
  [`Paper <http://www.aclweb.org/anthology/D15-1167>`_,
  `Implementation <https://github.com/NUSTM/DLSC>`_]

  .. image:: _img/mainpage/progress-overall-60.png


* **Semi-supervised recursive autoencoders for predicting sentiment distributions** :
  A novel machine learning framework based on recursive autoencoders for sentence-level prediction.
  [`Paper <https://dl.acm.org/citation.cfm?id=2145450>`_]

  .. image:: _img/mainpage/progress-overall-80.png


* **A convolutional neural network for modelling sentences** :
  A convolutional architecture adopted for the semantic modelling of sentences.
  [`Paper <https://arxiv.org/abs/1404.2188>`_]

  .. image:: _img/mainpage/progress-overall-80.png


* **Recursive deep models for semantic compositionality over a sentiment treebank** :
  Recursive Neural Tensor Network for sentiment analysis.
  [`Paper <http://www.aclweb.org/anthology/D13-1170>`_]

  .. image:: _img/mainpage/progress-overall-60.png


* **Adaptive recursive neural network for target-dependent twitter sentiment classification** :
  AdaRNN adaptively propagates the sentiments of words to target depending on the context and syntactic relationships.
  [`Paper <http://www.aclweb.org/anthology/P14-2009>`_]

  .. image:: _img/mainpage/progress-overall-20.png

* **Aspect extraction for opinion mining with a deep convolutional neural network** :
  A deep learning approach to aspect extraction in opinion mining.
  [`Paper <https://www.sciencedirect.com/science/article/pii/S0950705116301721>`_]

  .. image:: _img/mainpage/progress-overall-20.png


--------------------
Machine Translation
--------------------

.. ################################################################################
.. For continuous lines, the lines must be start from the same locations.
* **Learning phrase representations using RNN encoder-decoder for statistical machine translation** :
  The proposed RNN Encoder–Decoder with a novel hidden unit has been empirically evaluated on the task of machine translation.
  [`Paper <https://arxiv.org/abs/1406.1078>`_,
  `Code <https://github.com/pytorch/tutorials/blob/master/intermediate_source/seq2seq_translation_tutorial.py>`_,
  `Blog post <https://medium.com/@gautam.karmakar/learning-phrase-representation-using-rnn-encoder-decoder-for-machine-translation-9171cd6a6574>`_]


  .. image:: _img/mainpage/progress-overall-100.png

* **Sequence to Sequence Learning with Neural Networks** :
  A showcase of NMT system is comparable to the traditional pipeline by Google.
  [`Paper <http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural>`_,
  `Code <https://github.com/farizrahman4u/seq2seq>`_]

  .. image:: _img/mainpage/progress-overall-100.png


* **Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation** :
  This work presents the design and implementation of GNMT, a production NMT system at Google.
  [`Paper <https://arxiv.org/pdf/1609.08144.pdf>`_,
  `Code <https://github.com/tensorflow/nmt>`_]

  .. image:: _img/mainpage/progress-overall-100.png


* **Neural Machine Translation by Jointly Learning to Align and Translate** :
  An extension to the encoder–decoder model which learns to align and translate jointly by attention mechanism.
  [`Paper <https://arxiv.org/abs/1409.0473>`_]

  .. image:: _img/mainpage/progress-overall-100.png


* **Effective Approaches to Attention-based Neural Machine Translation** :
  Improvement of attention mechanism for NMT.
  [`Paper <https://arxiv.org/abs/1508.04025>`_,
  `Code <https://github.com/mohamedkeid/Neural-Machine-Translation>`_]

  .. image:: _img/mainpage/progress-overall-60.png

* **On the Properties of Neural Machine Translation: Encoder-Decoder Approaches** :
  Analyzing the properties of the neural machine translation using two models; RNN Encoder--Decoder and a newly proposed gated recursive convolutional neural network.
  [`Paper <https://arxiv.org/abs/1409.12595>`_]

  .. image:: _img/mainpage/progress-overall-60.png


* **On Using Very Large Target Vocabulary for Neural Machine Translation** :
  A method that allows to use a very large target vocabulary without increasing training complexity.
  [`Paper <https://arxiv.org/abs/1412.2007>`_]

  .. image:: _img/mainpage/progress-overall-40.png

* **Convolutional sequence to sequence learning** :
  An architecture based entirely on convolutional neural networks.
  [`Paper <https://arxiv.org/abs/1705.03122>`_,
  `Code[Torch] <https://github.com/facebookresearch/fairseq>`_,
  `Code[Pytorch] <https://github.com/pytorch/fairseq>`_,
  `Post <https://code.facebook.com/posts/1978007565818999/a-novel-approach-to-neural-machine-translation/>`_]

  .. image:: _img/mainpage/progress-overall-60.png


* **Attention Is All You Need** :
  The Transformer: a novel neural network architecture based on a self-attention mechanism.
  [`Paper <https://arxiv.org/abs/1706.03762>`_,
  `Code <https://github.com/tensorflow/tensor2tensor>`_,
  `Accelerating Deep Learning Research with the Tensor2Tensor Library  <https://ai.googleblog.com/2017/06/accelerating-deep-learning-research.html>`_,
  `Transformer: A Novel Neural Network Architecture for Language Understanding  <https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html>`_]

  .. image:: _img/mainpage/progress-overall-100.png


--------------------
Summarization
--------------------

.. ################################################################################
.. For continuous lines, the lines must be start from the same locations.

* **A Neural Attention Model for Abstractive Sentence Summarization** :
  A fully data-driven approach to abstractive sentence summarization based on a local attention model.
  [`Paper <https://arxiv.org/abs/1509.00685>`_,
  `Code <https://github.com/facebookarchive/NAMAS>`_,
  `A Read on "A Neural Attention Model for Abstractive Sentence Summarization" <http://thegrandjanitor.com/2018/05/09/a-read-on-a-neural-attention-model-for-abstractive-sentence-summarization-by-a-m-rush-sumit-chopra-and-jason-weston/>`_,
  `Blog Post <https://medium.com/@supersonic_ss/paper-a-neural-attention-model-for-abstractive-sentence-summarization-a6fa9b33f09b>`_,
  `Paper notes <https://github.com/dennybritz/deeplearning-papernotes/blob/master/notes/neural-attention-model-for-abstractive-sentence-summarization.md>`_,]

  .. image:: _img/mainpage/progress-overall-100.png

* **Get To The Point: Summarization with Pointer-Generator Networks** :
  A novel architecture that augments the standard sequence-to-sequence attentional model by using a hybrid pointer-generator network that may copy words from the source text via pointing and using coverage to keep track of what has been summarized.
  [`Paper <https://arxiv.org/abs/1704.04368>`_,
  `Code <https://github.com/abisee/pointer-generator>`_,
  `Video <https://www.coursera.org/lecture/language-processing/get-to-the-point-summarization-with-pointer-generator-networks-RhxPO>`_,
  `Blog Post <http://www.abigailsee.com/2017/04/16/taming-rnns-for-better-summarization.html>`_]

  .. image:: _img/mainpage/progress-overall-100.png

* **Abstractive Sentence Summarization with Attentive Recurrent Neural Networks** :
  A  conditional  recurrent  neural  network (RNN) based on convolutional attention-based encoder which generates a summary of an input sentence.
  [`Paper <http://www.aclweb.org/anthology/N16-1012>`_]

  .. image:: _img/mainpage/progress-overall-60.png

* **Abstractive Text Summarization Using Sequence-to-Sequence RNNs and Beyond** :
  Abstractive text summarization using Attentional Encoder-Decoder Recurrent Neural Networks
  [`Paper <https://arxiv.org/abs/1602.06023>`_]

  .. image:: _img/mainpage/progress-overall-60.png

* **A Deep Reinforced Model for Abstractive Summarization** :
  A neural network model with a novel intra-attention that attends over the input and continuously generated output separately, and a new training method that combines standard supervised word prediction and reinforcement learning (RL).
  [`Paper <https://arxiv.org/abs/1705.04304>`_]

  .. image:: _img/mainpage/progress-overall-60.png

--------------------
Question Answering
--------------------

* **Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks** :
  An argue for the usefulness of a set of proxy tasks that evaluate reading comprehension via question answering.
  [`Paper <https://arxiv.org/abs/1502.05698>`_]

  .. image:: _img/mainpage/progress-overall-60.png


* **Teaching Machines to Read and Comprehend** :
  addressing the lack of real natural language training data by introducing a novel approach to building a supervised reading comprehension data set.
  [`Paper <http://papers.nips.cc/paper/5945-teaching-machines-to-read-and-comprehend.pdf>`_]

  .. image:: _img/mainpage/progress-overall-80.png

* **Ask Me Anything Dynamic Memory Networks for Natural Language Processing** :
  Introducing the dynamic memory network (DMN), a neural network architecture which processes input sequences and questions, forms episodic memories, and generates relevant answers
  [`Paper <http://proceedings.mlr.press/v48/kumar16.pdf>`_]

  .. image:: _img/mainpage/progress-overall-80.png




..   * ``#`` with overline, for parts
..   * ``*`` with overline, for chapters
..   * ``=``, for sections
..   * ``-``, for subsections
..   * ``^``, for subsubsections
..   * ``"``, for paragraphs
..
.. ************
.. Heading 1
.. ************
..
.. ====================
.. Heading 2
.. ====================
..
.. -----------------------
.. Heading 3
.. -----------------------

************
Courses
************

.. image:: _img/mainpage/online.png

* **Natural Language Processing with Deep Learning** by Stanford :
  [`Link <http://web.stanford.edu/class/cs224n/>`_]

* **Deep Natural Language Processing** by the University of Oxford:
  [`Link <https://www.cs.ox.ac.uk/teaching/courses/2016-2017/dl/>`_]

* **Natural Language Processing with Deep Learning in Python** by Udemy:
  [`Link <https://www.udemy.com/natural-language-processing-with-deep-learning-in-python/?siteID=QhjctqYUCD0-KJsvUG2M8PW2kOmJ0nwFPQ&LSNPUBID=QhjctqYUCD0>`_]

* **Natural Language Processing with Deep Learning** by Coursera:
  [`Link <https://www.coursera.org/learn/language-processing>`_]


************
Books
************

.. image:: _img/mainpage/books.jpg

* **Speech and Language Processing** by Dan Jurafsky and James H. Martin at stanford:
  [`Link <https://web.stanford.edu/~jurafsky/slp3/>`_]

* **Neural Network Methods for Natural Language Processing** by Yoav Goldberg:
  [`Link <https://www.morganclaypool.com/doi/abs/10.2200/S00762ED1V01Y201703HLT037>`_]

* **Deep Learning with Text: Natural Language Processing (Almost) from Scratch with Python and spaCy** by Patrick Harrison, Matthew Honnibal:
  [`Link <https://www.amazon.com/Deep-Learning-Text-Approach-Processing/dp/1491984414>`_]

* **Natural Language Processing with Python** by Steven Bird, Ewan Klein, and Edward Loper:
  [`Link <http://www.nltk.org/book/>`_]


************
Blogs
************

.. image:: _img/mainpage/Blogger_icon.png

* **Understanding Convolutional Neural Networks for NLP** by Denny Britz:
  [`Link <http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/>`_]

* **Deep Learning, NLP, and Representations** by Matthew Honnibal:
  [`Link <http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/>`_]

* **Embed, encode, attend, predict: The new deep learning formula for state-of-the-art NLP models** by Sebastian Ruder:
  [`Link <https://explosion.ai/blog/deep-learning-formula-nlp>`_]

* **Embed, encode, attend, predict: The new deep learning formula for state-of-the-art NLP models** by Sebastian Ruder:
  [`Link <https://explosion.ai/blog/deep-learning-formula-nlp>`_]

* **Natural Language Processing** by Sebastian Ruder:
  [`Link <http://blog.aylien.com/12-of-the-best-free-natural-language-processing-and-machine-learning-educational-resources/>`_]

* **Probably Approximately a Scientific Blog** by Vered Schwartz:
  [`Link <http://veredshwartz.blogspot.com/>`_]

* **NLP news** by Sebastian Ruder:
  [`Link <http://newsletter.ruder.io/>`_]

* **Deep Learning for Natural Language Processing (NLP): Advancements & Trends**:
  [`Link <https://tryolabs.com/blog/2017/12/12/deep-learning-for-nlp-advancements-and-trends-in-2017/>`_]

* **Neural Language Modeling From Scratch**:
  [`Link <http://ofir.io/Neural-Language-Modeling-From-Scratch/?a=1>`_]


************
Tutorials
************

.. image:: _img/mainpage/tutorial.png

* **Understanding Natural Language with Deep Neural Networks Using Torch** by NVIDIA:
  [`Link <https://devblogs.nvidia.com/understanding-natural-language-deep-neural-networks-using-torch/>`_]

* **Deep Learning for NLP with Pytorch** by Pytorch:
  [`Link <https://pytorch.org/tutorials/beginner/deep_learning_nlp_tutorial.html>`_]

* **Deep Learning for Natural Language Processing: Tutorials with Jupyter Notebooks** by Jon Krohn:
  [`Link <https://insights.untapt.com/deep-learning-for-natural-language-processing-tutorials-with-jupyter-notebooks-ad67f336ce3f>`_]


************
Datasets
************

=====================
General
=====================

* **1 Billion Word Language Model Benchmark**: The purpose of the project is to make available a standard training and test setup for language modeling experiments:
  [`Link <http://www.statmt.org/lm-benchmark/>`_]

* **Common Crawl**: The Common Crawl corpus contains petabytes of data collected over the last 7 years. It contains raw web page data, extracted metadata and text extractions:
  [`Link <http://commoncrawl.org/the-data/get-started/>`_]

* **Yelp Open Dataset**: A subset of Yelp's businesses, reviews, and user data for use in personal, educational, and academic purposes:
  [`Link <https://www.yelp.com/dataset>`_]


=====================
Text classification
=====================

* **20 newsgroups** The 20 Newsgroups data set is a collection of approximately 20,000 newsgroup documents, partitioned (nearly) evenly across 20 different newsgroups:
  [`Link <http://qwone.com/~jason/20Newsgroups/>`_]

* **Broadcast News** The 1996 Broadcast News Speech Corpus contains a total of 104 hours of broadcasts from ABC, CNN and CSPAN television networks and NPR and PRI radio networks with corresponding transcripts:
  [`Link <https://catalog.ldc.upenn.edu/LDC97S44>`_]

* **The wikitext long term dependency language modeling dataset**: A collection of over 100 million tokens extracted from the set of verified Good and Featured articles on Wikipedia. :
  [`Link <https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset>`_]

=======================
Question Answering
=======================

* **Question Answering Corpus** by Deep Mind and Oxford which is two new corpora of roughly a million news stories with associated queries from the CNN and Daily Mail websites.
  [`Link <https://github.com/deepmind/rc-data>`_]

* **Stanford Question Answering Dataset (SQuAD)** consisting of questions posed by crowdworkers on a set of Wikipedia articles:
  [`Link <https://rajpurkar.github.io/SQuAD-explorer/>`_]

* **Amazon question/answer data** contains Question and Answer data from Amazon, totaling around 1.4 million answered questions:
  [`Link <http://jmcauley.ucsd.edu/data/amazon/qa/>`_]



=====================
Sentiment Analysis
=====================

* **Multi-Domain Sentiment Dataset** TThe Multi-Domain Sentiment Dataset contains product reviews taken from Amazon.com from many product types (domains):
  [`Link <http://www.cs.jhu.edu/~mdredze/datasets/sentiment/>`_]

* **Stanford Sentiment Treebank Dataset** The Stanford Sentiment Treebank is the first corpus with fully labeled parse trees that allows for a complete analysis of the compositional effects of sentiment in language:
  [`Link <https://nlp.stanford.edu/sentiment/>`_]

* **Large Movie Review Dataset**: This is a dataset for binary sentiment classification:
  [`Link <http://ai.stanford.edu/~amaas/data/sentiment/>`_]


=====================
Machine Translation
=====================

* **Aligned Hansards of the 36th Parliament of Canada** dataset contains 1.3 million pairs of aligned text chunks:
  [`Link <https://www.isi.edu/natural-language/download/hansard/>`_]

* **Europarl: A Parallel Corpus for Statistical Machine Translation** dataset extracted from the proceedings of the European Parliament:
  [`Link <http://www.statmt.org/europarl/>`_]


=====================
Summarization
=====================

* **Legal Case Reports Data Set** as a textual corpus of 4000 legal cases for automatic summarization and citation analysis.:
  [`Link <https://archive.ics.uci.edu/ml/datasets/Legal+Case+Reports>`_]

************
Contributing
************


*For typos, unless significant changes, please do not create a pull request. Instead, declare them in issues or email the repository owner*. Please note we have a code of conduct, please follow it in all your interactions with the project.

========================
Pull Request Process
========================

Please consider the following criterions in order to help us in a better way:

1. The pull request is mainly expected to be a link suggestion.
2. Please make sure your suggested resources are not obsolete or broken.
3. Ensure any install or build dependencies are removed before the end of the layer when doing a
   build and creating a pull request.
4. Add comments with details of changes to the interface, this includes new environment
   variables, exposed ports, useful file locations and container parameters.
5. You may merge the Pull Request in once you have the sign-off of at least one other developer, or if you
   do not have permission to do that, you may request the owner to merge it for you if you believe all checks are passed.

========================
Final Note
========================

We are looking forward to your kind feedback. Please help us to improve this open source project and make our work better.
For contribution, please create a pull request and we will investigate it promptly. Once again, we appreciate
your kind feedback and support.
