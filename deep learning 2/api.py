from typing import List, Dict, Any

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn

MODEL_PATH = "model/best_model.pth"
PREPROCESSOR_PATH = "model/preprocessor.pkl"
MAXMIN_SCALER_PATH = "model/minmaxscaler.pkl"
