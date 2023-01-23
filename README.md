# Moore Followup Training and Inference

This is the training source code for a set of transformer (RoBERTa)-based model to predict whether after receiving a CT in the ED, a potential incidental lung nodule finding (ILN) will require a followup or not. We classify into three classes NO_FOLLOWUP, CONDITIONAL_FOLLOWUP, and HARD_FOLLOWUP. The formal definitions are below:

**HARD_FOLLOWUP**: The patient definitively should get a followup visit for a possible ILN
**CONDITION_FOLLOWUP**: The patient may or may not need a followup visit for a possible ILN
**NO_FOLLOWUP**: The patient does not need a followup visit for a possible ILN

The model was trained on Yale-New Haven Health System ED CT reports from 2014-2021.

# Model Performance

Our best performing model achieves state-of-the-art performance compared to all other models found in the literature. We specifically ran it against the iScout model found in [Information from Searching Content with an Ontology-Utilizing Toolkit (iSCOUT) ](https://pubmed.ncbi.nlm.nih.gov/22349993/).   

Please check the [iScout.ipynb](https://github.com/vsocrates/moore-followup/blob/main/notebooks/iScout.ipynb) notebook to see these results.


## Results

|                      | Precision    | Recall       | F1           |
|----------------------|--------------|--------------|--------------|
| NO FOLLOWUP          | 0.9880952381 | 0.9431818182 | 0.9651162791 |
| CONDITIONAL FOLLOWUP | 0.9117647059 | 0.8857142857 | 0.8985507246 |
| HARD FOLLOWUP        | 0.8181818182 | 0.9642857143 | 0.8852459016 |

## Docker Container

We also developed a Docker container for non-computational scientists to recreate this pipeline with an easy-to-use UI located at [Docker Hub](https://hub.docker.com/repository/docker/vsocrates/moore/general). The code and instructions can be found at https://github.com/vsocrates/moore-followup-docker. 
