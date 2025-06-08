import torch
import os

config = {
        "BATCH_SIZE": 64,
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
        "TEST_SIZE": 0.1,
        "NUM_SAMPLES": 10000,
    
        # Router specific config
        "REGIMES": [0, 0.0478100, 0.08058, 1],
        "ROUTER_LR": 0.001,
        "ROUTER_PATIENCE": 10,
        "ROUTER_EPOCHS": 100,
        "ROUTER_MULTIPLICITY" : 2,

        # Expert specific config
        "EXPERTS_MULTIPLICITY": [1, 1, 2],
        "EXPERTS_LR" : [0.001] * 3,
        "EXPERTS_PATIENCE" : [10, 10, 10], #[50, 50, 25, 10, 10],
        "EXPERTS_EPOCHS" : [100, 200, 200], #[500, 500, 250, 100, 100],
    
        # Transformer specific parameters
        "NUM_QUERIES" : 20,
        # "D_INPUT": 50 * 9 + 50 * 6 + 7,
        "D_INPUT": 50 * 5,
        "D_MODEL": 100,
        "N_HEAD": 4,
        "NUM_LAYERS": 15,
        "DROPOUT": 0.2,
    
        # FFNN Specific parameters
        "D_EGO_FFNN_INPUT" : 6,
        "FFNN_D_HIDDEN" : 500,
        "FFNN_NUM_HIDDEN_LAYERS":7,
        "D_OUTPUT": 60 * 2,
        
        # Analysis parameters (optional based on ANALYZE flag)
        "ANALYZE": True,
        "ANALYZE_NUM_BATCHES": 50,
        "MODEL_SAVE_PATH": "Models"
    }




def getExpertConfig(org_config, data_based_ind, multiplicity_ind):
    config = {}
    copy_vals = ["BATCH_SIZE", "DEVICE", "TEST_SIZE", "NUM_SAMPLES", "REGIMES", "NUM_QUERIES", "D_INPUT", "D_MODEL", "N_HEAD", "NUM_LAYERS", "DROPOUT", "D_EGO_FFNN_INPUT", "FFNN_D_HIDDEN", "FFNN_NUM_HIDDEN_LAYERS", "D_OUTPUT"]
    for key in copy_vals:
        config[key] = org_config[key]

    num_regimes = len(org_config['REGIMES']) - 1

    addn_config = dict( 
        LEARNING_RATE=org_config["EXPERTS_LR"][data_based_ind],
        PATIENCE=org_config["EXPERTS_PATIENCE"][data_based_ind],
        EPOCHS=org_config["EXPERTS_EPOCHS"][data_based_ind],
        MIN_METRIC=org_config["REGIMES"][data_based_ind],
        MAX_METRIC=org_config["REGIMES"][data_based_ind+1],
        MODEL_SAVE_PATH=os.path.join(org_config['MODEL_SAVE_PATH'], f"expert_{num_regimes}_{data_based_ind+1}_{multiplicity_ind}.pth")
        )
    config.update(addn_config)
    
    return config

def getRouterConfig(org_config, multiplicity_ind):
    config = {}
    copy_vals = ["BATCH_SIZE", "DEVICE", "TEST_SIZE", "NUM_SAMPLES", "NUM_QUERIES", "D_INPUT", "D_MODEL", "N_HEAD", "NUM_LAYERS", "DROPOUT", "D_EGO_FFNN_INPUT", "FFNN_D_HIDDEN", "FFNN_NUM_HIDDEN_LAYERS"]
    for key in copy_vals:
        config[key] = org_config[key]

    num_regimes = len(org_config['REGIMES']) - 1

    addn_config = dict( 
        LEARNING_RATE=org_config["ROUTER_LR"],
        PATIENCE=org_config["ROUTER_PATIENCE"],
        EPOCHS=org_config["ROUTER_EPOCHS"],
        D_OUTPUT=len(org_config["REGIMES"]) - 1,
        REGIMES=list(zip(org_config['REGIMES'][:-1], org_config['REGIMES'][1:])),
        MODEL_SAVE_PATH=os.path.join(org_config['MODEL_SAVE_PATH'], f"router_{num_regimes}_{multiplicity_ind}.pth")
        )
    config.update(addn_config)
        
    return config
