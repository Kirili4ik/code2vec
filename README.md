This is my version of [code2vec](https://github.com/tech-srl/code2vec) work for Python3. It works only on keras implementation as for now. Some basic changes done:

0) Added Jupyter notebook with [preprocessing](pre-preprocessing.ipynb) of code snippets
1) Support of Python3 code thx to [JB parser](https://github.com/JetBrains-Research/astminer/tree/master-dev/astminer-cli)
2) Support of code embeddings (a.k.a. before the last dense layer, witch originally works only for TF implementation)
3) Getting target and token embeddings by running .sh
4) Getting top 10 synonyms for given label.


The rest of the README is almost the same with the original [code2vec](https://github.com/tech-srl/code2vec), but with some changes considering my implemetation. You should understand that the original work has a lot more opportunities (including already trained on Java models) so I really recommend working with it. Here I leave some dependencies on file and folder names, but anyone can get through them. 


# Code2vec
A neural network for learning distributed representations of code.
This is made on top of the implementation of the model described in:

[Uri Alon](http://urialon.cswp.cs.technion.ac.il), [Meital Zilberstein](http://www.cs.technion.ac.il/~mbs/), [Omer Levy](https://levyomer.wordpress.com) and [Eran Yahav](http://www.cs.technion.ac.il/~yahave/),
"code2vec: Learning Distributed Representations of Code", POPL'2019 [[PDF]](https://urialon.cswp.cs.technion.ac.il/wp-content/uploads/sites/83/2018/12/code2vec-popl19.pdf)

_**October 2018** - The paper was accepted to [POPL'2019](https://popl19.sigplan.org)_!

_**April 2019** - The talk video is available [here](https://www.youtube.com/watch?v=EJ8okcxL2Iw)_.

_**July 2019** - Add `tf.keras` model implementation

An **online demo** is available at [https://code2vec.org/](https://code2vec.org/).

#### Only keras version for now.

<center style="padding: 40px"><img width="70%" src="https://github.com/tech-srl/code2vec/raw/master/images/network.png" /></center>

Table of Contents
=================
  * [Requirements](#requirements)
  * [Quickstart](#quickstart)
  * [Configuration](#configuration)
  * [Features](#features)
  * [Citation](#citation)

## Requirements
On Ubuntu:
  * [Python3](https://www.linuxbabe.com/ubuntu/install-python-3-6-ubuntu-16-04-16-10-17-04) (>=3.6). To check the version:
> python3 --version
  * TensorFlow - version 2.0.0 ([install](https://www.tensorflow.org/install/install_linux)).
  To check TensorFlow version:
> python3 -c 'import tensorflow as tf; print(tf.\_\_version\_\_)'
  * If you are using a GPU, you will need CUDA 10.0
  ([download](https://developer.nvidia.com/cuda-10.0-download-archive-base)) 
  as this is the version that is currently supported by TensorFlow. To check CUDA version:
> nvcc --version
  * For GPU: cuDNN (>=7.5) ([download](http://developer.nvidia.com/cudnn)) To check cuDNN version:
> cat /usr/include/cudnn.h | grep CUDNN_MAJOR -A 2
  * For creating a new dataset (any operation that requires parsing of a new code example) - [JetBrains astminer](https://github.com/JetBrains-Research/astminer/tree/master-dev/astminer-cli) (their cli is already [here](cd2vec/cli.jar)

## Quickstart
### Step 0: Cloning this repository
```
git clone https://github.com/Kirili4ik/code2vec
cd code2vec
```

### Step 1: Creating a new dataset from java sources
In order to have a preprocessed dataset to train a network on you should create a new dataset of your own. It consists from 3 folders train, test and validation.

#### Creating and preprocessing a new Python dataset
In order to create and preprocess a new dataset (for example, to compare code2vec to another model on another dataset):
  * Edit the file [preprocess.sh](preprocess.sh) using the instructions there, pointing it to the correct training, validation and test directories.
  * Run the preprocess.sh file:
> source preprocess.sh

### Step 2: Training a model
You should train a new model using a preprocessed dataset.

#### Training a model from scratch
To train a model from scratch:
  * Edit the file [train.sh](train.sh) to point it to the right preprocessed data. By default, 
  it points to my "my_dataset" dataset that was preprocessed in the previous step.
  * Before training, you can edit the configuration hyper-parameters in the file [config.py](config.py),
  as explained in [Configuration](#configuration).
  * Run the [train.sh](train.sh) script:

> source train.sh


##### Notes:
  1. By default, the network is evaluated on the validation set after every training epoch.
  2. The newest 10 versions are kept (older are deleted automatically). This can be changed, but will be more space consuming.
  3. By default, the network is training for 20 epochs.
These settings can be changed by simply editing the file [config.py](config.py). You may need lots and lots of data because of the simplicity of the model.

### Step 3: Evaluating a trained model
Once the score on the validation set stops improving over time, you can stop the training process (by killing it)
and pick the iteration that performed the best on the validation set.
Suppose that iteration #8 is our chosen model, run:
```
python3 code2vec.py --framework keras --load models/my_first_model/saved_model --test data/my_dataset/my_dataset.test.c2v
```

### Step 4: Manual examination of a trained model
To manually examine a trained model, run:
```
source my_predict.sh
```
After the model loads, follow the instructions and edit the file [Input.py](pred_files/Input.py) and enter a Python 
method or code snippet, and examine the model's predictions and attention scores.

## Step 5: Getting embeddings
Follow Step 4 and embedding for your snippet will be in [EMBEDDINGS.txt](cd2vec/EMBEDDINGS.txt) file.

## Step 6: Look at synonyms
Run command:
>python3 my_find_synonim.py --label 'linear|algebra'
Or any other tag and look at the closest to it.

## Configuration
Changing hyper-parameters is possible by editing the file
[config.py](config.py).

Here are some of the parameters and their description:
#### config.NUM_TRAIN_EPOCHS = 20
The max number of epochs to train the model. Stopping earlier must be done manually (kill).
#### config.SAVE_EVERY_EPOCHS = 1
After how many training iterations a model should be saved.
#### config.TRAIN_BATCH_SIZE = 1024 
Batch size in training.
#### config.TEST_BATCH_SIZE = config.TRAIN_BATCH_SIZE
Batch size in evaluating. Affects only the evaluation speed and memory consumption, does not affect the results.
#### config.TOP_K_WORDS_CONSIDERED_DURING_PREDICTION = 10
Number of words with highest scores in $ y_hat $ to consider during prediction and evaluation.
#### config.NUM_BATCHES_TO_LOG_PROGRESS = 100
Number of batches (during training / evaluating) to complete between two progress-logging records.
#### config.NUM_TRAIN_BATCHES_TO_EVALUATE = 100
Number of training batches to complete between model evaluations on the test set.
#### config.READER_NUM_PARALLEL_BATCHES = 4
The number of threads enqueuing examples to the reader queue.
#### config.SHUFFLE_BUFFER_SIZE = 10000
Size of buffer in reader to shuffle example within during training.
Bigger buffer allows better randomness, but requires more amount of memory and may harm training throughput.
#### config.CSV_BUFFER_SIZE = 100 * 1024 * 1024  # 100 MB
The buffer size (in bytes) of the CSV dataset reader.

#### config.MAX_CONTEXTS = 200
The number of contexts to use in each example.
#### config.MAX_TOKEN_VOCAB_SIZE = 1301136
The max size of the token vocabulary.
#### config.MAX_TARGET_VOCAB_SIZE = 261245
The max size of the target words vocabulary.
#### config.MAX_PATH_VOCAB_SIZE = 911417
The max size of the path vocabulary.
#### config.DEFAULT_EMBEDDINGS_SIZE = 128
Default embedding size to be used for token and path if not specified otherwise.
#### config.TOKEN_EMBEDDINGS_SIZE = config.EMBEDDINGS_SIZE
Embedding size for tokens.
#### config.PATH_EMBEDDINGS_SIZE = config.EMBEDDINGS_SIZE
Embedding size for paths.
#### config.CODE_VECTOR_SIZE = config.PATH_EMBEDDINGS_SIZE + 2 * config.TOKEN_EMBEDDINGS_SIZE
Size of code vectors.
#### config.TARGET_EMBEDDINGS_SIZE = config.CODE_VECTOR_SIZE
Embedding size for target words.
#### config.MAX_TO_KEEP = 10
Keep this number of newest trained versions during training.
#### config.DROPOUT_KEEP_RATE = 0.75
Dropout rate used during training.
#### config.SEPARATE_OOV_AND_PAD = False
Whether to treat `<OOV>` and `<PAD>` as two different special tokens whenever possible.

## Features
Code2vec supports the following features: 

### Releasing the model (not sure)
If you wish to keep a trained model for inference only (without the ability to continue training it) you can
release the model using:
```
python3 code2vec.py --load models/my_first_model/saved_model --release
```
This will save a copy of the trained model with the '.release' suffix.
A "released" model usually takes 3x less disk space.

### Exporting the trained token vectors and target vectors
These saved embeddings are saved without subtoken-delimiters ("*toLower*" is saved as "*tolower*").

In order to export embeddings from a trained model, use:

> source my_get_embeddings.sh

This creates 2 files [tokens.txt](models/my_first_model/tokens.txt) and [targets.txt](models/my_first_model/targets.txt)

This saves the tokens/targets embedding matrices in word2vec format to the specified text file, in which:
the first line is: \<vocab_size\> \<dimension\>
and each of the following lines contains: \<word\> \<float_1\> \<float_2\> ... \<float_dimension\>

These word2vec files can be manually parsed or easily loaded and inspected using the [gensim](https://radimrehurek.com/gensim/models/word2vec.html) python package:
```python
python3
>>> from gensim.models import KeyedVectors as word2vec
>>> vectors_text_path = 'models/java14_model/targets.txt' # or: `models/java14_model/tokens.txt'
>>> model = word2vec.load_word2vec_format(vectors_text_path, binary=False)
>>> model.most_similar(positive=['equals', 'to|lower']) # or: 'tolower', if using the downloaded embeddings
>>> model.most_similar(positive=['download', 'send'], negative=['receive'])
```

## Citation

[code2vec: Learning Distributed Representations of Code](https://urialon.cswp.cs.technion.ac.il/wp-content/uploads/sites/83/2018/12/code2vec-popl19.pdf)

```
@article{alon2019code2vec,
 author = {Alon, Uri and Zilberstein, Meital and Levy, Omer and Yahav, Eran},
 title = {Code2Vec: Learning Distributed Representations of Code},
 journal = {Proc. ACM Program. Lang.},
 issue_date = {January 2019},
 volume = {3},
 number = {POPL},
 month = jan,
 year = {2019},
 issn = {2475-1421},
 pages = {40:1--40:29},
 articleno = {40},
 numpages = {29},
 url = {http://doi.acm.org/10.1145/3290353},
 doi = {10.1145/3290353},
 acmid = {3290353},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {Big Code, Distributed Representations, Machine Learning},
}
```
