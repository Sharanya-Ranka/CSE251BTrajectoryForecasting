from EnsembledAttentionAndNN.config import config, getExpertConfig, getRouterConfig
from EnsembledAttentionAndNN.train_router import RouterTrainer
from EnsembledAttentionAndNN.train_expert import ExpertTrainer
import torch
import os
import numpy as np
import pandas as pd
import utilities as utils


def getExpertPredictions(dataloader, expert_cohort, expert_trainer):
    expert_individual_predictions = []
    for model in expert_cohort:
        preds = expert_trainer.predictFromOutside(dataloader, model)
        expert_individual_predictions.append(preds)

    # breakpoint()
    return np.mean(expert_individual_predictions, axis=0)

def getRouterPredictions(dataloader, router_cohort, router_trainer):
    router_individual_predictions = []
    for model in router_cohort:
        preds = router_trainer.predictFromOutside(dataloader, model)
        router_individual_predictions.append(preds)

    # breakpoint()
    return np.mean(router_individual_predictions, axis=0)


def getFinalPredictions(expert_predictions, router_predictions):
    hard_predictions_ind = np.argmax(router_predictions, axis=-1)
    hard_predictions = np.zeros_like(router_predictions)
    hard_predictions[np.indices((2100,))[0], hard_predictions_ind] = 1
    
    rt_preds = hard_predictions.reshape(2100, 1, 1, -1)
    final_predictions = np.sum(expert_predictions * rt_preds, axis=-1)
    breakpoint()
    return final_predictions

def convertAndSavePredictions(predictions):
    assert tuple(predictions.shape) == (2100, 60, 2)

    pred_output = predictions.reshape(-1, 2)
    output_df = pd.DataFrame(pred_output, columns=["x", "y"])

    output_df.index.name = "index"
    output_df.to_csv(os.path.join(utils.SUBMISSION_DIR, "ensembled_attention_and_nn.csv"))

def main():
    dl = None
    router_cohort = []
    for multiplicity_ind in range(config["ROUTER_MULTIPLICITY"]):
        router_config = getRouterConfig(config, multiplicity_ind)
        rt = RouterTrainer(router_config)
        router_model = rt.loadModel()
        if dl is None:
            dl = rt.loadPredictionData()
        router_cohort.append(router_model)
        
    router_predictions = getRouterPredictions(dl, router_cohort, rt)


    expert_preds = []
    dl = None
    
    for expert_ind in range(len(config['REGIMES'])-1):
        expert_cohort = []
        for multiplicity_ind in range(config["EXPERTS_MULTIPLICITY"][expert_ind]):
            expert_config = getExpertConfig(config, expert_ind, multiplicity_ind)
            expert_trainer = ExpertTrainer(expert_config)
            expert_model = expert_trainer.loadModel()
            if dl is None:
                dl = expert_trainer.loadPredictionData()

            expert_cohort.append(expert_model)

        expert_preds.append(getExpertPredictions(dl, expert_cohort, expert_trainer))

    expert_preds = np.stack(expert_preds, axis=-1)

    final_preds = getFinalPredictions(expert_preds, router_predictions)
    # breakpoint()
    convertAndSavePredictions(final_preds)

    