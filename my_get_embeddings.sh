#!/usr/bin/env bash
####################################################################
# saves targets and tokens embeddings to models directory

python3 code2vec.py --framework keras --load models/my_first_model/saved_model --save_t2v models/my_first_model/targets.txt
python3 code2vec.py --framework keras --load models/my_first_model/saved_model --save_w2v models/my_first_model/tokens.txt

