
import pickle
from sentence_transformers import util


def score_system(embs, ref_embs):
    score_sum = 0
    for i, emb in enumerate(embs[:-1]): # ignore last embeddings (blank lines)
        cos_sims = []
        for embs_ in ref_embs:
            cos_sim = float( util.cos_sim(emb, embs_[i])[0][0] )
            cos_sims.append(cos_sim)
        score_sum += max(cos_sims)
    return score_sum / 2000


if __name__ == '__main__':
    dbfile = open('dstc6-embs', 'rb')
    db = pickle.load(dbfile)
    gen_embs = db['gen_embs']
    ref_embs = db['ref_embs']
    dbfile.close()
    
    for i, embs in enumerate(gen_embs):
        score = score_system(gen_embs[i], ref_embs)
        print(f'S_{i+1} score: {score:.2f}')


# S_1 score:    0.50
# S_2 score:    0.50
# S_3 score:    0.50
# S_4 score:    0.49
# S_5 score:    0.52
# S_6 score:    0.50
# S_7 score:    0.49
# S_8 score:    0.53
# S_9 score:    0.50
# S_10 score:   0.52
# S_11 score:   0.51
# S_12 score:   0.46
# S_13 score:   0.45
# S_14 score:   0.51
# S_15 score:   0.48
# S_16 score:   0.45
# S_17 score:   0.49
# S_18 score:   0.48
# S_19 score:   0.51
# S_20 score:   0.49
