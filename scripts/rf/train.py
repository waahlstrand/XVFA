#%%
import sklearn.ensemble
import lightning as L
from data.vertebra import VertebraDataModule
from models.vertebra.models import SingleVertebraClassifier
from models.vertebra.classifiers import VertebraParameters
import numpy as np
import torch
from utils.evaluate import classification_metrics
import pandas as pd
import joblib
# from utils.evaluate import *

def train():

    for fold in range(5):
        dm = VertebraDataModule(
                    height=224,
                    width=224,
                    source="/home/vicska/research/superb/superb-detection/folds.csv",
                    batch_size=32,
                    removed=[
                        "1977-05-15",
                        "360410-4848",
                        "MO1704",
                        "MO1863",
                        "MO2004",
                        "MO2005",
                        "MO2129",
                        "MO2335",
                        "MO2585",
                        "MO2799",
                        "MO2806",
                        "MO3018",
                        "MO2154",
                        "MO0181", 
                        "MO0996"
                        ],
                    bbox_expansion=0.35,
                    bbox_jitter=10,
                    bbox_normalization=True,
                    bbox_format="cxcywh",
                    n_classes=3,
                    fold=fold,
                    n_workers=4,
        )

        dm.prepare_data()
        dm.setup("fit")

        vp = VertebraParameters()

        # Collect data into a matrix
        X_train = {
            "ha": [],
            "hp": [],
            "hm": [],
            "apr": [],
            "mpr": [],
            "mar": [],
        }
        grades_train = []
        types_train = []

        for batch in dm.train_dataloader():
            y = batch.y
            keypoints   = torch.stack([target.keypoints for target in y]).squeeze(1)
            types       = torch.stack([target.labels for target in y]).squeeze(1)
            grades      = torch.stack([target.visual_grades for target in y]).squeeze(1)

            params = vp(keypoints) # Dict[str, Tensor]

            for key in X_train.keys():
                X_train[key].append(params[key])

            grades_train.append(grades)
            types_train.append(types)

        for key in X_train.keys():
            X_train[key] = torch.cat(X_train[key], dim=0)

        X_train = torch.stack([X_train["ha"], X_train["hp"], X_train["hm"], X_train["apr"], X_train["mpr"], X_train["mar"]], dim=1).cpu()
        grades_train = torch.cat(grades_train, dim=0).cpu()
        types_train = torch.cat(types_train, dim=0).cpu()

        rf_types = sklearn.ensemble.RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
        rf_types.fit(X_train, types_train)

        rf_grades = sklearn.ensemble.RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
        rf_grades.fit(X_train, grades_train)

        # Save the models
        joblib.dump(rf_types, f"rf_types_fold_{fold}.pkl")
        joblib.dump(rf_grades, f"rf_grades_fold_{fold}.pkl")

#%%
def test():

    model_dict = {
        0: "/home/vicska/research/superb/superb-detection/logs/vertebra/version_6/checkpoints/epoch=371-val_stage/distance=0.0184.ckpt",
        1: "/home/vicska/research/superb/superb-detection/logs/vertebra/version_5/checkpoints/epoch=597-val_stage/distance=0.0185.ckpt",
        2: "/home/vicska/research/superb/superb-detection/logs/vertebra/version_4/checkpoints/epoch=329-val_stage/distance=0.0177.ckpt",
        3: "/home/vicska/research/superb/superb-detection/logs/vertebra/version_3/checkpoints/epoch=651-val_stage/distance=0.0179.ckpt",
        4: "/home/vicska/research/superb/superb-detection/logs/vertebra/version_2/checkpoints/epoch=695-val_stage/distance=0.0176.ckpt"
    }

    records = []

    for fold in range(5):

        vertebra_model = SingleVertebraClassifier.load_from_checkpoint(model_dict[fold])
        vertebra_model.eval()

        # Load models
        rf_types = joblib.load(f"rf_types_fold_{fold}.pkl")
        rf_grades = joblib.load(f"rf_grades_fold_{fold}.pkl")

        dm = VertebraDataModule(
                    height=224,
                    width=224,
                    source="/home/vicska/research/superb/superb-detection/folds.csv",
                    batch_size=32,
                    removed=[
                        "1977-05-15",
                        "360410-4848",
                        "MO1704",
                        "MO1863",
                        "MO2004",
                        "MO2005",
                        "MO2129",
                        "MO2335",
                        "MO2585",
                        "MO2799",
                        "MO2806",
                        "MO3018",
                        "MO2154",
                        "MO0181", 
                        "MO0996"
                        ],
                    bbox_expansion=0.35,
                    bbox_jitter=10,
                    bbox_normalization=True,
                    bbox_format="cxcywh",
                    n_classes=3,
                    fold=0,
                    n_workers=4,
        )

        dm.prepare_data()
        dm.setup("test")

        vp = VertebraParameters()

        # Collect data into a matrix
        X_train = {
            "ha": [],
            "hp": [],
            "hm": [],
            "apr": [],
            "mpr": [],
            "mar": [],
        }
        grades_train = []
        types_train = []

        for batch in dm.train_dataloader():
            y = batch.y
            x = batch.x

            with torch.no_grad():
                x = x.to(vertebra_model.device)
                y_pred = vertebra_model(x)

            keypoints   = y_pred.keypoints.mu
            types       = torch.stack([target.labels for target in y]).squeeze(1)
            grades      = torch.stack([target.visual_grades for target in y]).squeeze(1)

            params = vp(keypoints) # Dict[str, Tensor]

            for key in X_train.keys():
                X_train[key].append(params[key])

            grades_train.append(grades)
            types_train.append(types)

        for key in X_train.keys():
            X_train[key] = torch.cat(X_train[key], dim=0)

        X_train = torch.stack([X_train["ha"], X_train["hp"], X_train["hm"], X_train["apr"], X_train["mpr"], X_train["mar"]], dim=1).cpu()
        grades_train = torch.cat(grades_train, dim=0).cpu()
        types_train = torch.cat(types_train, dim=0).cpu()

        types_pred = rf_types.predict_proba(X_train)
        grades_pred = rf_grades.predict_proba(X_train)

        types_pred = torch.from_numpy(types_pred)
        grades_pred = torch.from_numpy(grades_pred)

        all_groups_grades =[
                ("normal", ([0, ], [1, 2, 3])),
                ("mild",([1, ], [0, 2, 3])),
                ("moderate",([2, ], [0, 1, 3])),
                ("severe",([3, ], [0, 1, 2])),
                ("normal+mild",([0, 1], [2, 3])),
            ]
        
        all_groups_types = [
                ("normal", ([0, ], [1, 2])),
                ("wedge", ([1, ], [0, 2])),
                ("concave", ([2, ], [0, 1])),
            ]
        
        results_types = [{"target": "types", **r} for r in classification_metrics(types_train, types_pred, all_groups_types)]
        results_grades = [{"target": "grades", **r} for r in classification_metrics(grades_train, grades_pred, all_groups_grades)]

        records.extend(results_types)
        records.extend(results_grades)

    df = pd.DataFrame.from_records(records)

    return df


# %%
train()

df = test()

grouped = df.groupby(["target", "group_name"]).agg("mean", "std")
grouped.to_excel("random_forest_results.xlsx")
# %%
