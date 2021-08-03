# Learning Syntactic Dense Embedding with Correlation Graph For Automatic Readability Assessment

This repository contains the code and resources from the following [paper](https://www.aclanthology.org/2021.acl-long.235)

Please cite if you use the above resources for your research:

 ```
@inproceedings{qiu-etal-2021-learning,
    title = "Learning Syntactic Dense Embedding with Correlation Graph for Automatic Readability Assessment",
    author = "Qiu, Xinying and Chen, Yuan and Chen, Hanwu and Nie, Jian-Yun and Shen, Yuming and Lu, Dawei",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    year = "2021",
    pages = "3013--3025",
 ```

------
### Data Preparation
- GFE_Retrofitting: to produce syntactic dense embeddings with retrofitting
  - run: Syntactic_Dense_Embedding/project_embedding_retrofitting.py，
  - Results will be saved in: Syntactic_Dense_Embedding/GFE/GFE_Retrofitting 

- GFE_TransE: to produce syntactic dense embeddings with TransE
  - run: Syntactic_Dense_Embedding/project_embedding_TransE.py，
  - Results will be saved in: Syntactic_Dense_Embedding/GFE/GFE_TransE 

- G-Doc: to produce document representation with Gaussian binning applied to linguistic feature vectors
  - In: Readability_Assessment/data/G_Doc

- Raw: original linguistic feature vectors
  - In: Readability_Assessment/data/Raw_data

- [bert](https://github.com/google-research/bert)  
Download: BERT-Base, Multilingual Cased (New, recommended)，
Unzip and save to: Readability_Assessment/multilingual_L-12_H-768_A-12

### Environment configuration   
```python
pip install -r requirements.txt
```
### Train

Dual Model:
```python
python Readability_Assessment/src/doubleMultiModel.py
```
Results will be saved to: Readability_Assessment/result
