# %%
import krippendorff
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

agreement = pd.read_parquet("../../data/annotation_agreement_all_its.parquet.gzip")
# %%
agreement["implicit_layer_enc"] = agreement["implicit_layer"].replace({"Implicit HS": 1, "Explicit HS": 0})
agreement["subtlety_layer_enc"] = agreement["subtlety_layer"].replace({"Subtle": 1, "Non-Subtle": 0})
# %%
nof_its = agreement["iteration"].nunique()

its = []
imp_cohen_kappas = []
subt_cohen_kappas = []
krips = []

for i in range(nof_its):
    coder_1 = agreement[(agreement["iteration"] == i)
                        & (agreement["coder"] == 1)]
    coder_2 = agreement[(agreement["iteration"] == i)
                        & (agreement["coder"] == 2)]

    y1_imp = coder_1["implicit_layer_enc"]
    y2_imp = coder_2["implicit_layer_enc"]

    y1_subt = coder_1["subtlety_layer_enc"]
    y2_subt = coder_2["subtlety_layer_enc"]

    its.append(i)
    imp_cohen_kappas.append(cohen_kappa_score(y1_imp, y2_imp))
    subt_cohen_kappas.append(cohen_kappa_score(y1_subt, y2_subt))

    reliability_data = np.vstack([y1_subt + 2 * y1_imp, y2_subt + 2 * y2_imp])

    krips.append(
        krippendorff.alpha(reliability_data=reliability_data,
                           level_of_measurement="ordinal"))

agreement_scores = pd.DataFrame({
    "it": its,
    "imp_cohen_kappa": imp_cohen_kappas,
    "subt_cohen_kappa": subt_cohen_kappas,
    "krip": krips
})
# %%
agreement_scores
# %%
