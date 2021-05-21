import pandas as pd


class DataIO:
    def __init__(self):
        self.io = "data_io object"

    def format_submission(self, df, scores, file_name_addition=None):
        if "srch_id" not in df.columns:
            raise ValueError("Dataframe containing column 'srch_id' needed.")
        df["relevance_score"] = scores
        df.sort_values(["srch_id", "relevance_score"])
        df_output = df[["srch_id", "prop_id"]]
        df_output.to_csv("out/submission{0}".format(file_name_addition), index=False)
