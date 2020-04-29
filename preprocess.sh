#!/usr/bin/env bash
###########################################################
# Change the following values to preprocess a new dataset.
# TRAIN_DIR, VAL_DIR and TEST_DIR should be paths to      
#   directories containing sub-directories with .java files
#   each of {TRAIN_DIR, VAL_DIR and TEST_DIR} should have sub-dirs,
#   and data will be extracted from .java files found in those sub-dirs).
# DATASET_NAME is just a name for the currently extracted 
#   dataset.                                              
# MAX_CONTEXTS is the number of contexts to keep for each 
#   method (by default 200).                              
# WORD_VOCAB_SIZE, PATH_VOCAB_SIZE, TARGET_VOCAB_SIZE -   
#   - the number of words, paths and target words to keep 
#   in the vocabulary (the top occurring words and paths will be kept). 
#   The default values are reasonable for a Tesla K80 GPU 
#   and newer (12 GB of board memory).
# NUM_THREADS - the number of parallel threads to use. It is 
#   recommended to use a multi-core machine for the preprocessing 
#   step and set this value to the number of cores.
# PYTHON - python3 interpreter alias.

# cd2vec/java-small_raw/test/my_test
# cd2vec/python/my_test
TRAIN_DIR=cd2vec/python/my_train/
VAL_DIR=cd2vec/python/my_val/
TEST_DIR=cd2vec/python/my_test/
DATASET_NAME=my_dataset
MAX_CONTEXTS=200
WORD_VOCAB_SIZE=50000
PATH_VOCAB_SIZE=50000
TARGET_VOCAB_SIZE=50
NUM_THREADS=64
PYTHON=python3
JAVA=java 
###########################################################

TRAIN_DATA_FILE=${DATASET_NAME}_train
VAL_DATA_FILE=${DATASET_NAME}_val
TEST_DATA_FILE=${DATASET_NAME}_test
EXTRACTOR_JAR=cd2vec/cli.jar

mkdir -p data
mkdir -p data/${DATASET_NAME}

echo "Extracting paths from validation set..."
${JAVA} -jar ${EXTRACTOR_JAR} code2vec --lang py --project ${VAL_DIR} --output ${VAL_DATA_FILE} --maxH 8 --maxW 2 --maxContexts ${MAX_CONTEXTS} --maxTokens ${WORD_VOCAB_SIZE} --maxPaths ${PATH_VOCAB_SIZE}
echo "Finished extracting paths from validation set"
echo "Extracting paths from test set..."
${JAVA} -jar ${EXTRACTOR_JAR} code2vec --lang py --project ${TEST_DIR} --output ${TEST_DATA_FILE} --maxH 8 --maxW 2 --maxContexts ${MAX_CONTEXTS} --maxTokens ${WORD_VOCAB_SIZE} --maxPaths ${PATH_VOCAB_SIZE}
echo "Finished extracting paths from test set"
echo "Extracting paths from training set..."
${JAVA} -jar ${EXTRACTOR_JAR} code2vec --lang py --project ${TRAIN_DIR} --output ${TRAIN_DATA_FILE} --maxH 8 --maxW 2 --maxContexts ${MAX_CONTEXTS} --maxTokens ${WORD_VOCAB_SIZE} --maxPaths ${PATH_VOCAB_SIZE}
echo "Finished extracting paths from training set"

cat ${VAL_DATA_FILE}/path_*csv > ${VAL_DATA_FILE}/combined.csv
cat ${TEST_DATA_FILE}/path_*csv > ${TEST_DATA_FILE}/combined.csv
cat ${TRAIN_DATA_FILE}/path_*csv > ${TRAIN_DATA_FILE}/combined.csv

VAL_DATA_FILE=${VAL_DATA_FILE}/combined.csv
TEST_DATA_FILE=${TEST_DATA_FILE}/combined.csv
TRAIN_DATA_FILE=${TRAIN_DATA_FILE}/combined.csv


TARGET_HISTOGRAM_FILE=data/${DATASET_NAME}/${DATASET_NAME}.histo.tgt.c2v
ORIGIN_HISTOGRAM_FILE=data/${DATASET_NAME}/${DATASET_NAME}.histo.ori.c2v
PATH_HISTOGRAM_FILE=data/${DATASET_NAME}/${DATASET_NAME}.histo.path.c2v

echo "Creating histograms from the training data"
cat ${TRAIN_DATA_FILE} | cut -d' ' -f1 | awk '{n[$0]++} END {for (i in n) print i,n[i]}' > ${TARGET_HISTOGRAM_FILE}
cat ${TRAIN_DATA_FILE} | cut -d' ' -f2- | tr ' ' '\n' | cut -d',' -f1,3 | tr ',' '\n' | awk '{n[$0]++} END {for (i in n) print i,n[i]}' > ${ORIGIN_HISTOGRAM_FILE}
cat ${TRAIN_DATA_FILE} | cut -d' ' -f2- | tr ' ' '\n' | cut -d',' -f2 | awk '{n[$0]++} END {for (i in n) print i,n[i]}' > ${PATH_HISTOGRAM_FILE}




${PYTHON} preprocess.py --train_data ${TRAIN_DATA_FILE} --test_data ${TEST_DATA_FILE} --val_data ${VAL_DATA_FILE} \
  --max_contexts ${MAX_CONTEXTS} --word_vocab_size ${WORD_VOCAB_SIZE} --path_vocab_size ${PATH_VOCAB_SIZE} \
  --target_vocab_size ${TARGET_VOCAB_SIZE} --word_histogram ${ORIGIN_HISTOGRAM_FILE} \
  --path_histogram ${PATH_HISTOGRAM_FILE} --target_histogram ${TARGET_HISTOGRAM_FILE} --output_name data/${DATASET_NAME}/${DATASET_NAME}
    
# If all went well, the raw data files can be deleted, because preprocess.py creates new files 
# with truncated and padded number of paths for each example.
rm ${TARGET_HISTOGRAM_FILE} ${ORIGIN_HISTOGRAM_FILE} ${PATH_HISTOGRAM_FILE}

