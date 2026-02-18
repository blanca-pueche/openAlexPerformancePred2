from utils.pipeline import *

import pandas as pd
import streamlit as st

fnAll = "dfMultInst.p"
if os.path.exists(fnAll):
    dfAll = pickle.load(open(fnAll, "rb"))
else:
    dfAll={}

# Get all institutions authors (test)
allInstitutions = [
    "i4210087039", # 1 Instituto Investig. Biomedicas
    "i4210130807", # 2 Instituto Astrofisico de Canarias
    "i4210107147", # 3 Instituto Catalan de Oncologia
    "i4210120109", # 4 Museo Nac. Ciencias Naturales
    "i4210151127", # 5 Inst. Geociencias
    "i4210126640", # 6 Inst. Filosofia
    "i4210165411", # 7 CIEMAT
    "i4210113665", # 8 IIB Granada
    "i4210086614", # 9 Inst. Invest. 12 de octubre
    "i4210105802", # 10 Inst. Historia
    "i4210105141", # 11 Bioingenieria de Zaragoza
    "i4210118429", # 12 Ciencia de materiales de Madrid
    "i4210146061", # 13 CBM
    "i2799803557", # 14 BSC
    "i4210102407", # 15 Inst. Invest. Vall d'Hebron
    "i4210147680", # 16 CIB
    "i4210151560", # 17 Ciencias del Mar
    "i4210159146", # 18 Inst. Astrofisica de Andalucia
    "i4210148332", # 19 Barcelona Global Health
    "i4210129656", # 20 Ecologia
]
year=2020 # todo use this in the search!?

# All institutions
#for inst_id in allInstitutions:
#    if inst_id not in dfAll:
#       aids = authors_working_at_institution_in_year(inst_id, year)
#       print(inst_id,len(aids))
#       df = build_author_df_and_unique_work_distributions(aids, Y=year, mailto=MAILTO, sleep_s=0.25)
#       df = df[(df["count1"] > 0)].reset_index(drop=True)
#       dfAll[inst_id]=df
#       pickle.dump(dfAll, open(fnAll, "wb"))

# Selected institutions
inst_id = ['i4210105802','i4210129656']
for i, inst in enumerate(inst_id):
    aids = authors_working_at_institution_in_year(inst, year)
    print(inst, len(aids))
    df = build_author_df_and_unique_work_distributions(
        aids, Y=year, mailto=MAILTO, sleep_s=0.05
    )
    df = df[(df["count1"] > 0)].reset_index(drop=True)
    dfAll[inst] = df
    pickle.dump(dfAll, open(fnAll, "wb"))
print(dfAll)

# Selected authors
#aids = ['A5050710342', 'A5071564228', 'A5039659064']
#inst_ids = get_inst_ids_from_authors(aids)
#for aid, inst_id in zip(aids, inst_ids):
#    df = build_author_df_and_unique_work_distributions([aid], Y=year, mailto=MAILTO, sleep_s=1)
#    df = df[(df["count1"] > 0)].reset_index(drop=True)
#    if df.empty:
#        print("EMPTY DF for author", aid)
#        continue
#    if inst_id not in dfAll:
#        dfAll[inst_id] = df
#    else:
#        dfAll[inst_id] = pd.concat([dfAll[inst_id], df], ignore_index=True)

#pickle.dump(dfAll, open(fnAll, "wb"))

# Add citations columns
cols = ["count", "citationAvg", "maxCitation"]
dfClean = {}
parts = []

for inst_id, df in dfAll.items():
    d = df.loc[df["count1"] > 1].copy()

    d["citationAvg1"] = d["citations1"] / d["count1"]
    d["citationAvg2"] = d["citations2"] / d["count2"]

    for suffix in ["1","2"]:
      for c in cols:
          col = f"{c}{suffix}"
          d[f"{col}Perc"] = d[col].rank(pct=True)
    d["avgPerc1"]=(d["citationAvg1Perc"]+d["count1Perc"]+d["maxCitation1Perc"])/3
    d["avgPerc2"]=(d["citationAvg2Perc"]+d["count2Perc"]+d["maxCitation2Perc"])/3

    # collect only percentile columns
    perc_cols = [f"{c}{suffix}Perc" for c in cols for suffix in ["1", "2"]]
    out = d.loc[:, ["authorID","avgPerc1","avgPerc2"]+perc_cols].copy()

    parts.append(out)
    dfClean[inst_id] = d

df = pd.concat(parts, axis=0, ignore_index=True)
df = (
    df
    .sort_values(by="avgPerc1", ascending=False)
    .reset_index(drop=True)
)

print(df.head())

# -------------------------
# Example usage
# -------------------------
B = 1                      # total budget
score_col = "avgPerc1"
target_col = "avgPerc2"

# todo: elegir entre best_params, usar lo de por defecto, o con unas definidas por el user
best_params, results = grid_search_hyperparams(
    df=df, B=B, score_col=score_col,
    target_col=target_col,
    alphas=np.linspace(0.0, 1.0, 21),
    lambdas=np.linspace(0.0, 1.0, 21),
    gammas=np.linspace(0.1, 3.0, 21),
    # Optional bounds
    b_min=0.0, b_max=np.inf
)

print("Best params:", best_params)
print(results.head(10))

# default params
#alpha = 0.3
#lambda = 0.8
#gamma = 1.5
#b_min = 0.00
#b_max = inf

# EXAMPLE
alloc = allocate_budget(
        df=df,
        B=1,
        score_col='avgPerc1',
        alpha=best_params['alpha'],
        lambda_uniform=best_params['lambda'],
        gamma=best_params['gamma'],
        id_col='authorID',
        add_columns=True
    )

print(alloc.head(10))
