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

This project is the code to train TransE to get its project vector. 
You can copy the TransE output to ../Graph-based_feature_embedding_computation and run project_embedding_TransE.py for projecting feature.