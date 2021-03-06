# Learning-to-Borrow-for-KGs
Learning to Borrow – Relation Representation for Without-Mention Entity-Pairs for Knowledge Graph Completion

This repository is for learning-to-Borrow (SuperBorrow) model that integrates a text corpus with a Knowledge Graph (KG) to improve Knowledge Graph Embeddings (KGEs). In particular, SuperBorrow borrows Lexicalised Dependency Paths (LDPs), that are extracted from a text corpus, from the entity-pairs that co-occur in sentencs in the corpus (with-mentions) to represent entity-pairs that do not co-occur in any sentence in the corpus (without-mentions). The augmented KG is used to train several well-known KGE methods. The learnt KGEs have shown superior performance for link/relation prediction. 

If you use SuperBorrow for any published research, please include the following citation:

"Learning to Borrow – Relation Representation for Without-Mention Entity-Pairs for Knowledge Graph Completion"
Huda Hakami, Mona Hakami, Angrosh Mandya and Danushka Bollegala, Proc. of the Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL), Seattle, USA, 2022. 

# Overview
In Knowledge Graph Completion (KGC) if two entities co-occur in the same sentence, it is relatively easy to extract semantic relations between them even using simple methods such as Lexicalised Dependency Paths (LDPs). But what do we do when they don't? These so-called "without-mention" entity pairs cannot be appended to a knowledge graph because they do not have relational edges. This prevents us from learning KGEs for such entities. In our paper, we propose a method to borrow LDPs from with-mention entity pairs to represent the relations that exist between without-mention entity pairs. The proposed method (SuperBorrow) learns how to best borrow LDPs such that we can use those borrowed LDPs to append the without-mention entity pairs to KGs. This borrowing technique is a preprocessing step to KGE learning, which works directly on the KG and is independent of any KGE learning method. This enables you to run your favourite KGE method on the processed KG. 
 

# Prerequisites 

To train SuperBorrow model you require:


     - python 
     - tensorflow
     - sentence_transformers
     - sklearn

The KGEs for TransE, DistMult, ComplEx and RotatE in this work are trained using [OpenKE](https://github.com/thunlp/OpenKE) that is implemented with Pytorch. 
