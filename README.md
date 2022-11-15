# DSTC11-Benchmark
The purpose of this project is to identify a baseline classifier for DSTC-11. The default choice is Zhang et al's (2020) [AM-FM tool](readings/IWSDS_2020_paper_11.pdf) (used for DSTC-10 and previously). 
This project will investigate more recent approaches, based on fine-tuned large language models. Xhang et al note that their approach may be limited due to domain specificity. On the other hand LLMs are trained from large corpora that in priciple are less domain-dependent. This is an empirical question.

## Installation and setup
```conda create -n dstc11-env --python=3.8.6  
conda activate dstc11-env <br>
pip install -r requirements.txt  
```
```mkdir human_evaluation_data``` (add all the dataset JSONs within this) 
## Usage
```bash test_sentbert.sh -d <dataset_name> cuda -s wor```
