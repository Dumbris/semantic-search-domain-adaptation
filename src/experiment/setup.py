# coding: utf-8
import sys

from setuptools import setup, find_packages

install_requires = [
    'numpy',
    'pandas',
    'spacy',
    'rank_bm25',
    'hydra-core',
    'sentence_transformers', 
    'scipy',
    'hnswlib',
    'sklearn'
]

if sys.version_info < (2, 7):
    install_requires.append('importlib')
    install_requires.append('logutils')
    install_requires.append('ordereddict')

with open('README.md') as f:
    long_description = f.read()

setup(
    name='experiment',
    python_requires='>3.6.0',
    version='1.0.1',
    url='https://localhost',
    license='Apache License 2.0',
    description=('Package for search evaluation'),
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    install_requires=install_requires,
    extras_require={
    },
    zip_safe=False,
    platforms='any',
    classifiers=(
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ),
    test_suite='tests',
    entry_points = {
        'console_scripts': [
            'eval_encoder=experiment.eval_encoder:entry',
            'eval_reranker=experiment.eval_reranker:entry',
            'eval_bm25=experiment.eval_bm25:entry',
            'eval_uset=experiment.eval_uset:entry'
        ]
    }
)
