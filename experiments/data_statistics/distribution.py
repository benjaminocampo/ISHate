# %%
import pandas as pd

train = pd.read_parquet("../../data/ishate_train.parquet.gzip")
dev = pd.read_parquet("../../data/ishate_dev.parquet.gzip")
test = pd.read_parquet("../../data/ishate_test.parquet.gzip")
# %%
train["set"] = "train"
dev["set"] = "dev"
test["set"] = "test"

ishate = pd.concat([train, dev, test])
# %%
pd.crosstab(ishate["implicit_layer"].fillna("Non-HS"), ishate["set"])
# %%
pd.crosstab(ishate["subtlety_layer"].fillna("Non-HS"), ishate["set"])
# %%
pd.crosstab(ishate["implicit_layer"].fillna("Non-HS"), ishate["source"])
# %%
pd.crosstab(ishate["subtlety_layer"].fillna("Non-HS"), ishate["source"])
# %%
pd.crosstab(ishate["implicit_layer"].fillna("Non-HS"),
            ishate["set"],
            normalize="columns").round(4)
# %%
pd.crosstab(ishate["subtlety_layer"].fillna("Non-HS"),
            ishate["set"],
            normalize="columns").round(4)
# %%
imp_train = pd.read_parquet("../../data/implicit_task/train.parquet.gzip")
imp_dev = pd.read_parquet("../../data/implicit_task/dev.parquet.gzip")
imp_test = pd.read_parquet("../../data/implicit_task/test.parquet.gzip")
# %%
imp_aav = pd.read_parquet("../../data/implicit_task/aug_data/aav.parquet.gzip")
imp_bt = pd.read_parquet("../../data/implicit_task/aug_data/bt.parquet.gzip")
imp_eda = pd.read_parquet("../../data/implicit_task/aug_data/eda.parquet.gzip")
imp_gm_revised = pd.read_parquet("../../data/implicit_task/aug_data/gm_revised.parquet.gzip")
imp_gm = pd.read_parquet("../../data/implicit_task/aug_data/gm.parquet.gzip")
imp_ra = pd.read_parquet("../../data/implicit_task/aug_data/ra.parquet.gzip")
imp_ri = pd.read_parquet("../../data/implicit_task/aug_data/ri.parquet.gzip")
imp_rne = pd.read_parquet("../../data/implicit_task/aug_data/rne.parquet.gzip")
imp_rsa = pd.read_parquet("../../data/implicit_task/aug_data/rsa.parquet.gzip")
imp_all = pd.read_parquet("../../data/implicit_task/aug_data/all.parquet.gzip")

imp_aav["aug_method"] = "aav"
imp_bt["aug_method"] = "bt"
imp_eda["aug_method"] = "eda"
imp_gm_revised["aug_method"] = "gm_revised"
imp_gm["aug_method"] = "gm"
imp_ra["aug_method"] = "ra"
imp_ri["aug_method"] = "ri"
imp_rne["aug_method"] = "rne"
imp_rsa["aug_method"] = "rsa"
imp_all["aug_method"] = "all"

imp_train["aug_method"] = "real"

imp_train_w_aug = pd.concat([
    imp_train, imp_aav, imp_bt, imp_eda, imp_gm_revised, imp_gm, imp_ra,
    imp_ri, imp_rne, imp_rsa, imp_all
])
# %%
imp_train_w_aug.loc[imp_train_w_aug["label"] == 2, "aug_method"].value_counts()
# %%


# %%
subt_train = pd.read_parquet("../../data/subtle_task/train.parquet.gzip")
subt_dev = pd.read_parquet("../../data/subtle_task/dev.parquet.gzip")
subt_test = pd.read_parquet("../../data/subtle_task/test.parquet.gzip")
# %%
subt_aav = pd.read_parquet("../../data/subtle_task/aug_data/aav.parquet.gzip")
subt_bt = pd.read_parquet("../../data/subtle_task/aug_data/bt.parquet.gzip")
subt_eda = pd.read_parquet("../../data/subtle_task/aug_data/eda.parquet.gzip")
subt_gm_revised = pd.read_parquet("../../data/subtle_task/aug_data/gm_revised.parquet.gzip")
subt_gm = pd.read_parquet("../../data/subtle_task/aug_data/gm.parquet.gzip")
subt_ra = pd.read_parquet("../../data/subtle_task/aug_data/ra.parquet.gzip")
subt_ri = pd.read_parquet("../../data/subtle_task/aug_data/ri.parquet.gzip")
subt_rne = pd.read_parquet("../../data/subtle_task/aug_data/rne.parquet.gzip")
subt_rsa = pd.read_parquet("../../data/subtle_task/aug_data/rsa.parquet.gzip")
subt_all = pd.read_parquet("../../data/subtle_task/aug_data/all.parquet.gzip")

subt_aav["aug_method"] = "aav"
subt_bt["aug_method"] = "bt"
subt_eda["aug_method"] = "eda"
subt_gm_revised["aug_method"] = "gm_revised"
subt_gm["aug_method"] = "gm"
subt_ra["aug_method"] = "ra"
subt_ri["aug_method"] = "ri"
subt_rne["aug_method"] = "rne"
subt_rsa["aug_method"] = "rsa"
subt_all["aug_method"] = "all"

subt_train["aug_method"] = "real"

subt_train_w_aug = pd.concat([
    subt_train, subt_aav, subt_bt, subt_eda, subt_gm_revised, subt_gm, subt_ra,
    subt_ri, subt_rne, subt_rsa, subt_all
])
# %%
subt_train_w_aug.loc[subt_train_w_aug["label"] == 2, "aug_method"].value_counts()