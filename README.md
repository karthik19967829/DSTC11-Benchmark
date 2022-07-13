# DSTC11-Benchmark
The purpose of this project is to identify a baseline classifier for DSTC-11. The default choice is Zhang et al's AM-FM tool (used for DSTC-10 and previously). 
This project will investigate more recent approaches, based on fine-tuned large language models. Xhang et al note that their approach may be limited due to domain specificity. On the other hand LLMs are trained from large corpora that in priciple are less domain-dependent. This is an empirical question.

## Work Plan
Investigate benckmarking:
- replicate dstc-10 setup; baseline code (Deep AM-FM) + data = result
- define dstc11 eanchmark system: using metrics from dstc10 *and possiby others*
- build model (e.g. OPT-300k, T5, etc): fine tune
- benchmark candidate and Deep AM-FM (dstc-10 benchmark data) to verify
- propose a DSTC11 candidate 

Agree on DSTT11 training/dev/test datasets (with UMadrid)

## Materials
- [readings](readings/README.md) on the topic
