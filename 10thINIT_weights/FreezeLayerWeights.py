import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(64,128),
    nn.Linear(128,256)
)

for name , param in model.named_parameters():
    if name == '0.weight' or name == '0.bias':
        param.requires_grad = False



# Save the initial state of the parameters
initial_params = {name: param.clone() for name, param in model.named_parameters()}

# Dummy input and target (for demonstration purposes)
input = torch.randn(1, 64)
target = torch.randn(1, 256)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)

# Perform a training step
optimizer.zero_grad()
output = model(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()

# Check if the parameters have changed
for name, param in model.named_parameters():
    if param.requires_grad:
        changed = not torch.equal(initial_params[name], param)
        print(f'Parameter {name} changed: {changed}')
    else:
        print(f'Parameter {name} is frozen.')
