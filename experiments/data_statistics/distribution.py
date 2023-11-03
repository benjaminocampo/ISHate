# %% [markdown]
# This code is a data analysis script in Python using the Pandas library. It
# loads several datasets in parquet format and performs data manipulations and
# exploratory data analysis.
# %% [markdown]
# First we import the Pandas library and create an alias pd. Then we load three
# datasets: train, dev, and test from parquet files. The path of each file
# starts with "../../data/" and ends with ".parquet.gzip".
# %%
import pandas as pd

train = pd.read_parquet("../../data/ishate_train.parquet.gzip")
dev = pd.read_parquet("../../data/ishate_dev.parquet.gzip")
test = pd.read_parquet("../../data/ishate_test.parquet.gzip")
# %% [markdown]
# Then we add a new column called "set" to each of the three
# datasets, indicating the set to which each observation belongs: "train",
# "dev", or "test". After that, we  concatenate the three datasets into a single
# data frame ishate.
# %%
train["set"] = "train"
dev["set"] = "dev"
test["set"] = "test"

ishate = pd.concat([train, dev, test])
# %% [markdown]
# The next lines calculate and display cross-tabulations
# between the "implicit_layer" and "set" columns and between the
# "subtlety_layer" and "set" columns, respectively. The missing values in the
# "implicit_layer" and "subtlety_layer" columns are filled with "Non-HS"
# %%
pd.crosstab(ishate["implicit_layer"].fillna("Non-HS"), ishate["set"])
# %%
pd.crosstab(ishate["subtlety_layer"].fillna("Non-HS"), ishate["set"])
# %% [markdown]
# We perform the same crosstabs but this time comparing the column "source"
# %%
pd.crosstab(ishate["implicit_layer"].fillna("Non-HS"), ishate["source"])
# %%
pd.crosstab(ishate["subtlety_layer"].fillna("Non-HS"), ishate["source"])

# %% [markdown]
# We calculate and display cross-tabulations between the "implicit_layer" and
# "set" columns and between the "subtlety_layer" and "set" columns,
# respectively, with columns normalized. The results are rounded to 4 decimal
# places.
# %%
pd.crosstab(ishate["implicit_layer"].fillna("Non-HS"),
            ishate["set"],
            normalize="columns").round(4)
# %%
pd.crosstab(ishate["subtlety_layer"].fillna("Non-HS"),
            ishate["set"],
            normalize="columns").round(4)
# %% [markdown]
# The lines load several datasets for the implicit task and augment them with
# additional datasets for data augmentation. The datasets are loaded from
# parquet files with similar path structures as above.
#
# The next several lines add a new column "aug_method" to each of the datasets,
# indicating the augmentation method used. The method "real" is used for the
# original "imp_train" dataset.
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
], ignore_index=True)
# %% [markdown]
# We count the number of implicit messages grouped by augmentation method.
# %%
imp_train_w_aug.loc[imp_train_w_aug["label"] == 2, "aug_method"].value_counts()
# %% [markdown]
# We do the same with the subtle task.
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
imp_train_w_aug["set"] = "train"
#imp_train_w_aug.loc[(imp_train_w_aug["aug_method"] == "aav") & (imp_train_w_aug["label"] == 1), "label"]
real_col = pd.crosstab(imp_train_w_aug["label"], [imp_train_w_aug["set"], imp_train_w_aug["aug_method"]])["train"][["real"]]
# %%
tab = pd.crosstab(imp_train_w_aug["label"], [imp_train_w_aug["set"], imp_train_w_aug["aug_method"]]).copy()
# %%
for c1, c2 in tab:
    if c2 != "real":
        tab.loc[:, (c1, c2)] = tab[c1][c2] + tab["train"]["real"]
# %%
tab2 = tab.copy()
for c1, c2 in tab2:
    tab2.loc[:, (c1, c2)] = tab2[c1][c2] / tab2[c1][c2].sum()
# %%
for i, row in tab.iterrows():
    for c1, c2 in row:
        if row[c2] != "real":
            tab.loc[i, c1] = row[c1] + tab.loc[i, "train"]["real"]
# %%
subt_train_w_aug.loc[subt_train_w_aug["label"] == 2, "aug_method"].value_counts()
# %%
subt_train_w_aug["set"] = "train"
#real_col = pd.crosstab(subt_train_w_aug["label"], [subt_train_w_aug["set"], subt_train_w_aug["aug_method"]])["train"][["real"]]
subt_tab = pd.crosstab(subt_train_w_aug["label"], [subt_train_w_aug["set"], subt_train_w_aug["aug_method"]]).copy()
for c1, c2 in subt_tab:
    if c2 != "real":
        subt_tab.loc[:, (c1, c2)] = subt_tab[c1][c2] + subt_tab["train"]["real"]
# %%
subt_tab2 = subt_tab.copy()
for c1, c2 in subt_tab2:
    subt_tab2.loc[:, (c1, c2)] = subt_tab2[c1][c2] / subt_tab2[c1][c2].sum()

subt_tab3 = subt_tab2["train"][["real", "rsa", "aav", "rne", "ri", "ra", "eda", "bt", "gm", "gm_revised", "all"]].round(3)
# %%
