# Learning-to-Borrow-for-KGs
Learning to Borrow – Relation Representation for Without-Mention Entity-Pairs for Knowledge Graph Completion

This repository is for learning-to-Borrow (SuperBorrow) model that integrates a text corpus with a Knowledge Graph (KG) to improve Knowledge Graph Embeddings (KGEs). In particular, SuperBorrow borrows Lexicalised Dependency Paths (LDPs), that are extracted from a text corpus, from the entity-pairs that co-occur in sentencs in the corpus (with-mentions) to represent entity-pairs that do not co-occur in any sentence in the corpus (without-mentions). The augmented KG is used to train several well-known KGE methods. The learnt KGEs have shown superior performance for link/relation prediction. 

If you use SuperBorrow for any published research, please include the following citation:

"Learning to Borrow – Relation Representation for Without-Mention Entity-Pairs for Knowledge Graph Completion"
Huda Hakami, Mona Hakami, Angrosh Mandya and Danushka Bollegala, Proc. of the Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL), Seattle, USA, 2022. 

# Prerequisites 

To train SuperBorrow model you require:


     - python 
     - tensorflow
     - sentence_transformers
     - sklearn

The KGEs for TransE, DistMult, ComplEx and RotatE in this work are trained using [OpenKE](https://github.com/thunlp/OpenKE) that is implemented with Pytorch. 
