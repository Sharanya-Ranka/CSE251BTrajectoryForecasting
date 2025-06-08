import pandas as pd
from utilities import *
from functools import reduce


def getSubmission(submission_name):
    df = pd.read_csv(os.path.join(SUBMISSION_DIR, submission_name))
    # df["index"] = df["index"].astype(int)
    df.set_index("index", inplace=True)
    return df


submissions_to_ensemble = [
    # "lstm_submission.csv",
    # "simple_nn_submission.csv",
    # "attention_and_nn (10).csv",
    # "attention_and_nn (11).csv",
    # "attention_and_nn (3).csv",
     "simple_nn_submission_high_accel.csv",
    "simple_nn_submission_low_accel.csv"
   
]

dfs = [getSubmission(sn) for sn in submissions_to_ensemble]

avg_df = reduce(lambda x, y: x + y, dfs) / len(dfs)

for i in range(len(dfs)):
    for j in range(i + 1, len(dfs)):
        print(f"{i} {j} {((dfs[i] - dfs[j])**2).mean()}")
        
# avg_df.to_csv(os.path.join(SUBMISSION_DIR, "join3.csv"))
