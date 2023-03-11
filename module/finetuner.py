import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TuningModel(nn.Module):
    def __init__(self, model, model_topology, n_emotion, dropout):
        super.__init__()
        self.model = model
        self.emotion_linear = nn.Linear(model_topology, n_emotion)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input):
        output = self.model(input)
        output = self.emotion_linear(output)
        return output

