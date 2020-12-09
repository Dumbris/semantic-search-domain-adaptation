# Intro

I want to research how BERT is useful in products search for e-commerce sites. Traditionally for product search uses term-matching algorithms builtin into ElasticSearch or Solr. But these systems may fail when queries and product descriptions use different terms to describe the same meaning. Semantic search based on BERT models could help in this case.
The main question is how the domain adaptation of BERT model improves search relevancy. 

## Dataset

My initial plan is to use Home Depot dataset for finetuning/training and testing (https://www.kaggle.com/c/home-depot-product-search-relevance).

## Runing expreriments
* Install python package using pip, and run command:
  <pre>
  eval_cli
  </pre>

## Local development

* Run bash script:
  <pre>
  ./start_development.sh
  </pre>

It will create virtualenv for you, install all dependencies.