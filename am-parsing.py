
import pickle
from sentence_transformers import SentenceTransformer


def get_gen_sents():
    sents = []
    for n in range(1,21):
        f = open(f'dstc6_t2_evaluation/hypotheses/S_{n}.txt', 'r')
        text = f.read()
        lines = text.split('\n')
        sents.append(lines)
        f.close()
    return sents # 20 x 2000


def get_ref_sents():
    sents = []
    fnames = ['original_refs.txt'] + [f'refgen_result{n}.txt' for n in range(1,11)]
    for fname in fnames:
        f = open(f'dstc6_t2_evaluation/references/{fname}', 'r')
        text = f.read()
        lines = text.split('\n')
        sents.append(lines)
        f.close()
    return sents # 11 x 2000


if __name__ == '__main__':
    gen_sents = get_gen_sents() # per system
    ref_sents = get_ref_sents() # 11 reference sentences per generated sentence

    model = SentenceTransformer('all-MiniLM-L6-v2')
    gen_embs = [model.encode(sents, convert_to_tensor=True) for sents in gen_sents]
    ref_embs = [model.encode(sents, convert_to_tensor=True) for sents in ref_sents]

    db = dict()
    db['gen_embs'] = gen_embs
    db['ref_embs'] = ref_embs
    dbfile = open('dstc6-embs', 'ab')
    pickle.dump(db, dbfile)
    dbfile.close()
