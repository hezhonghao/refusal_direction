import torch
import os 

# Set the current work directory 
os.chdir('/home/ubuntu/model-diffing/refusal_direction/pipeline/runs/meta-llama-3-8b-instruct/generate_directions/')
# Load the tensor from the file
mean_diffs = torch.load('mean_diffs.pt')
# Print the tensor to inspect its contents
print(mean_diffs)

# Print the shape of the tensor
print(mean_diffs.shape)

# Print the data type of the tensor
print(mean_diffs.dtype)
