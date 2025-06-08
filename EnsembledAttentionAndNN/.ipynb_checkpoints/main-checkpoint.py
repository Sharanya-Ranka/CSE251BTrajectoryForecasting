from EnsembledAttentionAndNN.config import config, getExpertConfig, getRouterConfig
from EnsembledAttentionAndNN.train_router import RouterTrainer
from EnsembledAttentionAndNN.train_expert import ExpertTrainer
import torch
import os


def main():
    router_config = getRouterConfig(config)
    rt = RouterTrainer(router_config)
    # breakpoint()
    rt.performPipeline()

    # for expert_ind in range(len(config['REGIMES']) - 1):
    #     print(f"\n\nTraining expert {expert_ind}\n\n")
    #     expert_config = getExpertConfig(config, expert_ind)
    #     expert = ExpertTrainer(expert_config)
    #     expert.performPipeline()
    # expert_ind = 2
    # print(f"\n\nTraining expert {expert_ind}\n\n")
    # expert_config = getExpertConfig(config, expert_ind)
    # expert = ExpertTrainer(expert_config)
    # expert.performPipeline()

