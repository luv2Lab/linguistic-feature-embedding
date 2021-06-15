```shell script
TransE
│  count_corr.py
│  list.txt
│  model_TransE_newLoss.py
│  model_TransE_oldLoss.py
│  README.md
│  
├─sample
│      Cambridge_33feature.csv
│      Cambridge_33feature_entity2id.csv
│      Cambridge_33feature_TransE_train2id0.7-P_gaussian_retrofitting.txt
│      Cambridge_33feature_TransE_train2id0.7.csv
│      relation2id.csv
│      
└─TransE_output
        Cambridge_33feature_TransE_300dim_0.7_ent_embed_gaussian.txt
        Cambridge_33feature_TransE_300dim_0.7_rel_embed_gaussian.txt
```
This is an implementation of the TransE model by Kunxun Qi and Yuan Chen.
count_corr.py is used to get the correlation function, and it will generate a file called xxx_train2id.csv.
model_TransE_newLoss.py is the TransE model with new loss function by us.
model_TransE_oldLoss.py is the TransE model with the initial loss function

This project is the code to train TransE to get its project vector. 
You can copy the TransE_output to ../Graph-based_feature_embedding_computation/TransE_output and run project_embedding_TransE.py for projecting feature.

The TransE paper is:
Bordes, A. et al. 2013. Translating embeddings for modeling multi-relational data. In Advances in Neural Information Processing Systems. Curran Associates, Inc, pages 2787–2795.
