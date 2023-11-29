import torch 
import torch.nn as nn
import torch.nn.functional as F
from GENSynthetic_MultiClassData import generate_multiclass_synthetic_data

X_data, y_data = generate_multiclass_synthetic_data(100,5,3)

model = nn.Sequential(
    nn.Linear(5,20),
    nn.Linear(20,40),
    nn.Linear(40,3),
    nn.Softmax(dim=-1)
)

results = model(X_data)
print(results[0:10])

predicted_labels = torch.argmax(results,dim=1)
print(predicted_labels[0:10])

hot_encoded_data = F.one_hot(predicted_labels,num_classes=3)
print(hot_encoded_data[0:10])

