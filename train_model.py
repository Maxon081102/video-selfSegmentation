import os
import torch

def create_model_and_optimizer(model_class, model_params, device, lr=1e-3, beta1=0.9, beta2=0.999):
    model = model_class(**model_params)
    model = model.to(device)
    
    optimized_params = []
    for param in model.parameters():
        if param.requires_grad:
            optimized_params.append(param)
    optimizer = torch.optim.Adam(optimized_params, lr, [beta1, beta2])
    return model, optimizer

def load_model_and_create_optimizer(
    model_class, 
    model_params, 
    model_name,
    device,
    epoch,
    chkp_folder = "chkp",
    lr=1e-3, 
    beta1=0.9, 
    beta2=0.999, 
):
    model = model_class(**model_params)
    model = model.to(device)
    
    checkpoint = torch.load(os.path.join(chkp_folder, model_name, f'{model_name}#{epoch}.pt'))

    model, optimizer = create_model_and_optimizer(
        model_class,
        model_params,
        device,
        lr,
        beta1, 
        beta2,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, optimizer