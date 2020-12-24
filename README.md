# Intro

I want to research how BERT is useful in products search for e-commerce sites. Traditionally for product search uses term-matching algorithms builtin into ElasticSearch or Solr. But these systems may fail when queries and product descriptions use different terms to describe the same meaning. Semantic search based on BERT models could help in this case.
The main question is how the domain adaptation of BERT model improves search relevancy. 

## Prelimitary results
[My report for Huawei NLP course project.](docs/huawei_course_report/main.pdf)

## Dataset

My initial plan is to use Home Depot dataset for finetuning/training and testing (https://www.kaggle.com/c/home-depot-product-search-relevance).

## Runing expreriments
* Install python package using pip, and run command with default config:
  <pre>
  eval_encoder
  </pre>

* You can override any config option using command line, and use multirun feature to run many experiments:
  <pre>
  eval_encoder -m models.senttrans.loss="SoftmaxLoss,CosineSimilarityLoss" \
    models.senttrans.base_model="distilroberta-base-msmarco-v2,distilbert-base-nli-stsb-quora-ranking,sentence-transformers/LaBSE" \
    hydra.sweep.dir="eval_runs_results"
  </pre>

* To collect single report from all runs. use command:
  <pre>
  ./collect_results.sh path/to/eval_runs_results
  </pre>


## Local development

* Run bash script:
  <pre>
  ./start_development.sh
  </pre>

It will create virtualenv for you, install all dependencies.