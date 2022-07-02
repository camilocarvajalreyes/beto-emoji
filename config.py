import pickle

file_names = {
    "df_es_mapping": "../Mineria/Proyecto/data-mining-2022/Data/mapping/df_es_mapping.pickle",
    "df_es_test": "../Mineria/Proyecto/data-mining-2022/Data/test/df_es_test.pickle",
    "df_es_train": "../Mineria/Proyecto/data-mining-2022/Data/train/df_es_train.pickle",
    "df_es_trial": "../Mineria/Proyecto/data-mining-2022/Data/trial/df_es_trial.pickle",
}

df_es_mapping = pickle.load(open(file_names["df_es_mapping"], "rb")).sort_values("label")
