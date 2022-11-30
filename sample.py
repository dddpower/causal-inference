import logging
import logging.config
import warnings

from dowhy import CausalModel
import dowhy.datasets
import numpy as np
import pandas as pd
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)

# Config dict to set the logging level

DEFAULT_LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'loggers': {
        '': {
            'level': 'WARN',
        },
    }
}

logging.config.dictConfig(DEFAULT_LOGGING)
logging.info("Getting started with DoWhy. Running notebook...")


def run_sample1():
    data = dowhy.datasets.linear_dataset(beta=10,
            num_common_causes=5,
            num_instruments = 2,
            num_effect_modifiers=1,
            num_samples=5000, 
            treatment_is_binary=True,
            stddev_treatment_noise=10,
            num_discrete_common_causes=1)
    df = data["df"]
    print(df.head())
    print(data["dot_graph"])
    print("\n")
    print(data["gml_graph"])
    return

class Sample:
    def __init__(self):
        # Load some sample data
        data = dowhy.datasets.linear_dataset(
            beta=10,
            num_common_causes=5,
            num_instruments=2,
            num_samples=10000,
            treatment_is_binary=True)
        # I. Create a causal model from the data and given graph.
        model = CausalModel(
            data=data["df"],
            treatment=data["treatment_name"],
            outcome=data["outcome_name"],
            graph=data["gml_graph"])
        # II. Identify causal effect and return target estimands
        identified_estimand = model.identify_effect()
        
        # III. Estimate the target estimand using a statistical method.
        estimate = model.estimate_effect(identified_estimand,
                                         method_name="backdoor.propensity_score_matching")
        
        # IV. Refute the obtained estimate using multiple robustness checks.
        refute_results = model.refute_estimate(identified_estimand, estimate,
                                               method_name="random_common_cause")
        self.data = data
        self.model = model
        self.identified_estimand = identified_estimand
        self.estimate = estimate
        self.refute_results = refute_results
if __name__ == "__main__":
    test = Sample()
