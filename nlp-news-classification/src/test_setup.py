import torch
import datasets
import transformers
import pandas as pd
import sklearn

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("datasets:", datasets.__version__)
print("transformers:", transformers.__version__)
print("pandas:", pd.__version__)
print("sklearn:", sklearn.__version__)