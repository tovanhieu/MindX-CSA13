import math

import pandas as pd
import numpy as np

import plotly.graph_objects as go
import plotly.subplots as sp
from sklearn.model_selection import train_test_split

from config import Config


class LettuceDataset:
    def __init__(self, datapath, split):
        self.datapath = datapath
        self.df = load_csv(self.datapath)
        self.split = split
    
    def preprocess(self, seed=42):
        if self.split == "train":
            X = self.df.drop(columns=['Plant_ID', 'Date', 'Growth Days'], axis=1)
            y = self.df["Growth Days"]
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)

            std_features = standardize(X)
            log_TDS = np.log1p(X["TDS Value (ppm)"])
            std_log_TDS = standardize(log_TDS)
            
            X["Date"] = pd.to_datetime(self.df["Date"])
            X["Month"] = X["Date"].dt.month
            X["DayofYear"] = X["Date"].dt.dayofyear

            X = pd.concat([
                std_features,
                std_log_TDS.rename('Standardized Log TDS'),
                X[["Month", "DayofYear"]]
            ], axis=1)

            X_train_transformed = X.loc[X_train.index, :]
            X_val_transformed = X.loc[X_val.index, :]

            return X_train_transformed, X_val_transformed, y_train, y_val
        
        elif self.split == "test":
            X = self.df.drop(columns=['Plant_ID', 'Date'], axis=1)

            std_features = standardize(X)
            log_TDS = np.log1p(X["TDS Value (ppm)"])
            std_log_TDS = standardize(log_TDS)

            X["Date"] = pd.to_datetime(self.df["Date"])
            X["Month"] = X["Date"].dt.month
            X["DayofYear"] = X["Date"].dt.dayofyear

            X = pd.concat([
                std_features,
                std_log_TDS.rename('Standardized Log TDS'),
                X[["Month", "DayofYear"]]
            ], axis=1)

            return X.loc[X.index, :]
    
    def append_data(self, data):
        self.df = pd.concat([self.df, pd.DataFrame([data])], ignore_index=True)
    
    def write_data(self):
        if self.split == "train":
            self.df.to_csv(Config.TRAIN_PATH, index=False)
        elif self.split == "test":
            self.df.to_csv(Config.TEST_PATH, index=False)
    
    def read_data(self):
        if self.split == "train":
            self.df = pd.read_csv(Config.TRAIN_PATH)
        elif self.split == "test":
            self.df = pd.read_csv(Config.TEST_PATH)

    def visualize_preprocess(self):
        self.df['Date'] = pd.to_datetime(self.df['Date']).dt.strftime("%m/%d/%Y")

    def visualize(self, chart_type, features=None, title=None):
        figures = list()
        if chart_type == "corr":
            df = self.df.drop("Date", axis=1)
            corr = df.corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            
            figures.append(
                go.Figure(go.Heatmap(z=corr.mask(mask), x=corr.columns, y=corr.columns, colorscale = 'Viridis'))
            )
            return plot_from_trace(figures, rows=len(figures), cols=1, vertical_spacing=0.5, title=title)
        
        elif chart_type == "dist":
            for feature in features:
                figures.append(go.Figure(go.Histogram(x=self.df[feature], name=feature,
                                       marker=dict(line=dict(width=1, colorscale='Viridis')),
                )))
            return plot_from_trace(figures, rows=math.ceil(len(figures)), title=title)
        
        elif chart_type == "bar":
            df = self.df.drop(["Date"], axis=1)
            df = df.corr()["Growth Days"].drop("Growth Days")
            largest_value = abs(df).max()
            colors = ['lightslategray' if abs(df[i]) != largest_value else 'crimson' for i in df.index]
            figures.append(
                go.Figure(go.Bar(x=df, y=df.index, orientation="h", marker_color=colors))
            )
            return plot_from_trace(figures, title=title)
        
        elif chart_type == "scatter":
            figures.append(
                go.Figure(go.Scatter(x=self.df[features[0]], y=self.df[features[1]],
                                     mode="markers",
                                     marker=dict(
                                         colorscale='Inferno',
                                         color=self.df["Plant_ID"],
                                        )
                                     ))
            )
            return plot_from_trace(figures, title=title, xaxis_title=features[0], yaxis_title=features[1])
        
    def visualize_predictions(self, y_true, y_pred, features:list, title:str):
        figures = list()
        X = self.df.loc[:, features]

        figures.append(
            go.Figure(
                data=[
                    go.Scatter3d(
                        x=X.iloc[:, 0], y=X.iloc[:, 1], z=y_true,
                        mode="markers",
                        marker=dict(
                            color='blue',
                            size=5,
                            line=dict(
                                color='black',
                                width=0.5
                            )
                        ),
                        name='True Growth Days',
                        showlegend=True
                    ),
                    go.Scatter3d(
                        x=X.iloc[:, 0],
                        y=X.iloc[:, 1],
                        z=y_pred,
                        mode="markers",
                        marker=dict(
                            color='red',
                            size=5,
                            line=dict(
                                color='black',
                                width=0.5
                            )
                        ),
                        name='Prediction Growth Days',
                        showlegend=True
                    )
                ]
            )
        )

        return plot_from_trace(
            figures, 
            title=title,
            scene=dict(
                xaxis_title=features[0],
                yaxis_title=features[1],
                zaxis_title='Growth Days'
            ),
            specs=[[{"type": "scatter3d"}]]
        )

    def get_latest_data(self, feature, plant_id):
        """
        Retrieves the latest growth day for a given ID.
        
        Parameters:
            plant_id (int): The ID of the plant.
            
        Returns:
            int: The index of the row of the latest growth day of the plant.
        """
        # Filter the data for the specified plant ID
        filtered_data = self.df.loc[self.df["Plant_ID"] == plant_id]
        
        # Check if there are entries for the given Plant_ID
        if not filtered_data.empty:
            # Find the maximum value of 'Growth Days' for the specified plant ID
            idx_latest_growth_day = filtered_data[feature].idxmax()
            return idx_latest_growth_day
        else:
            return None
    
def load_csv(file_path, encoding="latin-1"):
    df = pd.read_csv(file_path, encoding=encoding)
    return df

@staticmethod
def standardize(features:pd.DataFrame):
    return (features - features.mean()) / features.std()

def plot_from_trace(figures, rows=1, cols=1, title=None, xaxis_title=None, yaxis_title=None, zaxis_title=None, scene=None, **kwargs):
    fig = sp.make_subplots(rows=rows, cols=cols, **kwargs)
    for i, figure in enumerate(figures):
        row = (i//cols) + 1
        col = (i % cols) + 1
        for trace in figure["data"]:
            fig.append_trace(trace, row=row, col=col)
    
    fig.update_layout(
        title={
            'text': title,
        },
        autosize=False,
        height=600,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='black',
                  size=15),
        scene = scene,
        )
    
    if xaxis_title:
        fig.update_xaxes(title=xaxis_title)

    if yaxis_title:
        fig.update_yaxes(title=yaxis_title)
        
    return fig