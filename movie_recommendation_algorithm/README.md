# MovieRecommender Algorithm

This is the algorithm for movie recommender.
### Example

`Recommender: "Hello, I hear that you are seeking a movie recommendation. Tell me about your movie preferences?"`

`User: "I like thriller movies. What thriller or horror movies would you suggest watching that came out in 2019?"`

#### Step 1. Generate the responses, and predict the positions of movie placeholders.
`"<movie> might be a good choice for you. It is a thriller movie came out in 2019."`
#### Step 2. Predict the movies.
`Joker (2019 film)`
#### Step 3. Put the movies into the placeholders, and recommend to users.
`Recommender: "Joker (2019 film) might be a good choice for you. It is a thriller movie came out in 2019."`

## Three subparts:

1. Knowledge Graph
2. Responses Generation
3. Movies Generation

## Requirements

- python >= 3.6
- pytorch == 1.9.0
- transformers == 4.15.0


## Dataset processing
### Download Dataset
Two conversational recommendation dataset: [Redial](https://redialdata.github.io/website/) and [Inspired](https://github.com/sweetpeach/Inspired)
### Download Knowledge Graph
Download DBpedia Knowledge Graph from the [link](https://databus.dbpedia.org/dbpedia/mappings/mappingbased-objects/2021.09.01/mappingbased-objects_lang=en.ttl.bz2), after unzipping, move it into data/dbpedia.
```
python Data/dbpedia/extract_kg.py

# inspired
python Data/inspired/extract_subkg.py
python Data/inspired/remove_entity.py
```

## 1. Knowledge Graph Pretraining
Pretrain Knowledge Graph Encoder with R-GCN
```
# inspired
python Data/inspired/process.py
bash KnowledgeGraph/train_pre.sh 
```
## 2. NLP Pretraining
Pretrain NLP Encoder with DialoGPT
```
# inspired
python Data/inspired/process_mask.py
bash NLP/train_conv.sh
bash NLP/infer_conv.sh
```

## 3. Recommendation Pretraining
Pretrain recommendation Encoder with RoBERTa
```
python Data/inspired_gen/merge.py
bash Recommendation/train_rec.py
```
## Reference
```
@inproceedings{hayati-etal-2020-inspired,
    title = "INSPIRED: Toward Sociable Recommendation Dialog Systems",
    author = "Hayati, Shirley Anugrah  and Kang, Dongyeop  and Zhu, Qingxiaoyang  and Shi, Weiyan  and Yu, Zhou",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.654",
    pages = "8142--8152",
}

@inproceedings{wang2022towards,
  title={Towards Unified Conversational Recommender Systems via Knowledge-Enhanced Prompt Learning},
  author={Wang, Xiaolei and Zhou, Kun and Wen, Ji-Rong and Zhao, Wayne Xin},
  booktitle={Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={1929--1937},
  year={2022}
}

@article{wang2019dgl,
    title={Deep Graph Library: A Graph-Centric, Highly-Performant Package for Graph Neural Networks},
    author={Minjie Wang and Da Zheng and Zihao Ye and Quan Gan and Mufei Li and Xiang Song and Jinjing Zhou and Chao Ma and Lingfan Yu and Yu Gai and Tianjun Xiao and Tong He and George Karypis and Jinyang Li and Zheng Zhang},
    year={2019},
    journal={arXiv preprint arXiv:1909.01315}
}

@inproceedings{zhang2019dialogpt,
    title={DialoGPT: Large-Scale Generative Pre-training for Conversational Response Generation},
    author={Yizhe Zhang and Siqi Sun and Michel Galley and Yen-Chun Chen and Chris Brockett and Xiang Gao and Jianfeng Gao and Jingjing Liu and Bill Dolan},
    year={2020},
    booktitle={ACL, system demonstration}
}

@article{liu2019roberta,
    title = {RoBERTa: A Robustly Optimized BERT Pretraining Approach},
    author = {Yinhan Liu and Myle Ott and Naman Goyal and Jingfei Du and
              Mandar Joshi and Danqi Chen and Omer Levy and Mike Lewis and
              Luke Zettlemoyer and Veselin Stoyanov},
    journal={arXiv preprint arXiv:1907.11692},
    year = {2019}
}
    
@inproceedings{li2018conversational,
  title={Towards Deep Conversational Recommendations},
  author={Li, Raymond and Kahou, Samira Ebrahimi and Schulz, Hannes and Michalski, Vincent and Charlin, Laurent and Pal, Chris},
  booktitle={Advances in Neural Information Processing Systems 31 (NIPS 2018)},
  year={2018}
}
```
