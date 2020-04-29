from gensim.models import KeyedVectors as word2vec
from argparse import ArgumentParser

def arguments_parser():
    parser = ArgumentParser()
    parser.add_argument("--label", dest="label",
                            help="label str to work with", required=True)
    return parser

args = arguments_parser().parse_args()

vectors_text_path = 'models/my_first_model/targets.txt'
model = word2vec.load_word2vec_format(vectors_text_path, binary=False)
print(model.most_similar(positive=[args.label]))
