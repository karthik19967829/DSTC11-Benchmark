# score am/fm per args
# [20221020] (air) added minimal comments
#


import json
import torch
import numpy as np
import argparse
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
from transformers import BertModel, BertTokenizer, GPT2Tokenizer, GPT2LMHeadModel
import re
from sentence_transformers import SentenceTransformer, util

torch.manual_seed(2000)
np.random.seed(2000)


dataset_meta_info = {
    "dstc6": {"num_references": 11, "annotations": ["overall"], "aggregation": np.mean},
    "humod": {
        "num_references": 3,
        "annotations": ["language_usage", "relevance"],
        "aggregation": np.mean,
    },
    "fed-turn": {
        "annotations": [
            "Correct",
            "Engaging",
            "Fluent",
            "Interesting",
            "Overall",
            "Relevant",
            "Semantically appropriate",
            "Specific",
            "Understandable",
        ],
        "aggregation": np.mean,
    },
    "convai2-grade": {"annotations": ["relevance"], "aggregation": np.mean},
    "empathetic-grade": {"annotations": ["relevance"], "aggregation": np.mean},
    "jsalt": {"annotations": ["appropriateness"], "aggregation": np.mean},
    "dailydialog-grade": {"annotations": ["relevance"], "aggregation": np.mean},
    "dailydialog-predictive": {"annotations": ["overall"], "aggregation": np.mean},
    "dailydialog-holistic": {
        "annotations": ["relevance"],
        "aggregation": lambda x: x[0],
    },
    "dailydialog-gupta": {"annotations": ["overall"], "aggregation": lambda x: x[0]},
    "esl": {"annotations": ["appropriateness"], "aggregation": lambda x: x[0]},
    "ncm": {"annotations": ["appropriateness"], "aggregation": lambda x: x[0]},
    "dstc10-topical": {
        "annotations": ["content", "grammar", "appropriateness", "relevance"],
        "aggregation": np.mean,
    },
    "dstc10-persona": {
        "annotations": ["content", "grammar", "appropriateness", "relevance"],
        "aggregation": np.mean,
    },
    "dailydialog-zhao": {
        "num_references": 1,
        "annotations": ["content", "grammar", "appropriateness", "relevance"],
        "aggregation": np.mean,
    },
    "dstc7": {
        "num_references": 1,
        "annotations": ["relevance", "informativeness", "overall"],
        "aggregation": np.mean,
    },
    "persona-zhao": {
        "num_references": 1,
        "annotations": ["appropriateness"],
        "aggregation": np.mean,
    },
    "persona-usr": {
        "num_references": 1,
        "annotations": [
            "Understandable",
            "Natural",
            "Maintains Context",
            "Engaging",
            "Uses Knowledge",
            "Overall",
        ],
        "aggregation": np.mean,
    },
    "topical-usr": {
        "annotations": [
            "Understandable",
            "Natural",
            "Maintains Context",
            "Engaging",
            "Uses Knowledge",
            "Overall",
        ],
        "aggregation": np.mean,
    },
}


def compute_fm_score(x, y):
    return max([x, y]) / min([x, y])


def normalize_df(dataset_name, df, ds_meta):
    dataset_meta = ds_meta[dataset_name]
    for annotation in dataset_meta["annotations"]:
        df["annotations." + annotation] = df["annotations." + annotation].apply(
            dataset_meta["aggregation"]
        )
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="up")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--am_model_path", type=str, default="embedding_models/persona_am/"
    )
    parser.add_argument(
        "--fm_model_path", type=str, default="language_models/persona_fm"
    )
    args = parser.parse_args()
    print(args)
    globals().update(args.__dict__)

    bert_model = BertModel.from_pretrained(am_model_path).to(device)
    bert_tokenizer = BertTokenizer.from_pretrained(am_model_path)
    bert_model.eval()

    gpt2_tokenizer = GPT2Tokenizer.from_pretrained(fm_model_path)
    gpt2_model = GPT2LMHeadModel.from_pretrained(fm_model_path).to(device)
    gpt2_model.eval()

    # xlmr_sentbert_model
    xlmr_sentbert_model = SentenceTransformer(
        "sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens"
    )

    # get data
    with open("human_evaluation_data/{}_eval.json".format(dataset)) as f:
        df = pd.json_normalize(json.load(f))
    df = normalize_df(dataset, df, dataset_meta_info)
    response_list = df.response.to_list()
    response_list = [item if item != "" else "no response" for item in response_list]

    context_list = [item.split("\n")[-1] for item in df.context.to_list()]

    annotations = [
        "annotations." + _ for _ in dataset_meta_info[dataset]["annotations"]
    ]
    human_scores = {}
    for k in annotations:
        human_scores[k] = list(df[k])

    response_embedding_list = []
    with torch.no_grad():
        for r in tqdm(response_list):
            r_sent_embedding = xlmr_sentbert_model.encode(r, convert_to_tensor=True)
            response_embedding_list.append(r_sent_embedding.cpu().numpy())

    reference_embedding_list = []
    with torch.no_grad():
        for r in tqdm(context_list):
            r_sent_embedding = xlmr_sentbert_model.encode(r, convert_to_tensor=True)
            reference_embedding_list.append(r_sent_embedding.cpu().numpy())

    print("\nAM scores")
    am_scores = []
    for idx, (x, y) in enumerate(
        zip(response_embedding_list, reference_embedding_list)
    ):
        single_am_score = float(util.cos_sim(x, y)[0][0])
        am_scores.append(single_am_score)

    cutoff = np.quantile(am_scores, 0.05)
    modified_rating = np.array([cutoff if t < cutoff else t for t in am_scores])
    normed_am_scores = (modified_rating - cutoff) / np.abs(cutoff)
    for k, v in human_scores.items():
        pear, p = pearsonr(v, normed_am_scores)
        kf = re.match(r"(.+?)\.(.+)", k).group(2)  # strip off redundant info
        print(f"AM -- Pearson:  {kf:s}=> {pear:6.3f}  p={p:6.4f}")
        spear, p = spearmanr(v, normed_am_scores)
        print(f"AM -- Spearman: {kf:s}=> {spear:6.3f}  p={p:6.4f}")

    df["am_scores"] = normed_am_scores

    print("\nFM scores")
    fm_scores = []
    with torch.no_grad():
        for prev, cur in tqdm(zip(context_list, response_list)):
            joint_enc = gpt2_tokenizer.encode(str(prev) + " " + str(cur)) + [50256]
            q = gpt2_tokenizer.encode(str(prev)) + [50256]
            batch_joint = torch.tensor([joint_enc]).to(device)
            batch_q = torch.tensor([q]).to(device)
            loss_joint = gpt2_model(batch_joint, labels=batch_joint)[0]
            loss_q = gpt2_model(batch_q, labels=batch_q)[0]
            p_joint = -loss_joint * (len(joint_enc) - 1)
            p_q = -loss_q * (len(q) - 1)
            score = (p_joint - (p_q)) / ((len(joint_enc) - 1) - (len(q) - 1))
            fm_scores.append(score.item())

    cutoff = np.quantile(fm_scores, 0.05)
    modified_rating = np.array([cutoff if t < cutoff else t for t in fm_scores])

    normed_fm_scores = (modified_rating - cutoff) / np.abs(cutoff)
    for k, v in human_scores.items():
        kf = re.match(r"(.+?)\.(.+)", k).group(2)  # strip off redundant info
        pear, p = pearsonr(v, normed_fm_scores)
        print(f"AM -- Pearson:  {kf:s}=> {pear:6.3f}  p={p:6.4f}")
        spear, p = spearmanr(v, normed_fm_scores)
        print(f"FM -- Spearman: {kf:s}=> {spear:6.3f} p={p:6.4f}")

    df["fm_scores"] = normed_fm_scores

    print("\nAM-FM scores")
    amfm_pear_sum = amfm_pear_count = 0.0
    amfm_spear_sum = amfm_spear_count = 0.0
    amfm_count = 0
    am_fm_scores = [np.mean([x, y]) for x, y in zip(normed_am_scores, normed_fm_scores)]
    for k, v in human_scores.items():
        kf = re.match(r"(.+?)\.(.+)", k).group(2)  # strip off redundant info
        pear, p = pearsonr(v, am_fm_scores)
        print(f"_wor FM -- Pearson:  {kf:s}=> {pear:6.3f}  p={p:6.4f}")
        spear, p = spearmanr(v, am_fm_scores)
        print(f"_wor FM -- Spearman: {kf:s}=> {spear:6.3f} p={p:6.4f}")
        amfm_pear_sum = amfm_pear_sum + pear
        amfm_spear_sum = amfm_spear_sum + spear
        amfm_count = amfm_count + 1

    df["am_fm_scores"] = am_fm_scores
    amfm_pear_mean = amfm_pear_sum / amfm_count
    print(f"\nmean r: {amfm_pear_mean:6.4f}")
    amfm_spear_mean = amfm_spear_sum / amfm_count
    print(f"mean r: {amfm_spear_mean:6.4f}")

    # write file with all computed scores
    df.to_csv(dataset + "_wor_" + "_results.csv", index=None)

#
