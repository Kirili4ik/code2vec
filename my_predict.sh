#!/usr/bin/env bash
##############################################
# Make predictions for files in pred_files directory
# cli.jar is in cd2vec directory and output (4 files from JB cli, deleted after all)
# for another model change /my_first_model/saved_model

java -jar cd2vec/cli.jar code2vec --lang py --project pred_files --output cd2vec --maxH 8 --maxW 1
python3 code2vec.py --framework keras --load models/my_first_model/saved_model --predict
rm -v cd2vec/node_types.csv cd2vec/path_contexts_*.csv cd2vec/paths.csv cd2vec/tokens.csv
rm -d cd2vec/py
