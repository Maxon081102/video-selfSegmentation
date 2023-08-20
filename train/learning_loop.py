import os
import torch
import warnings
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from collections import defaultdict
from IPython.display import clear_output

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def learning_loop(
    train,
    val,
    model,
    optimizer,
    train_loader,
    val_loader,
    criterion,
    scheduler=None,
    min_lr=None,
    epochs=10,
    val_every=1,
    draw_every=1,
    separate_show=False,
    model_name=None,
    chkp_folder="./chkps",
    metric_names=None,
    save_only_best=True,
    test_dataset=None,
    loss_info=None,
    out_size=1,
):
    if model_name is None:
        if os.path.exists(chkp_folder):
            num_starts = len(os.listdir(chkp_folder)) + 1
        else:
            num_starts = 1
        model_name = f'model#{num_starts}'
    
    if os.path.exists(os.path.join(chkp_folder, model_name)):
        model_name = model_name + "_v2"
        warnings.warn(f"Selected model_name was used already! To avoid possible overwrite - model_name changed to {model_name}")
    os.makedirs(os.path.join(chkp_folder, model_name))
    
    losses = {'train': [], 'val': []}
    lrs = []
    best_val_loss = np.Inf
    if metric_names is not None:
        metrics = defaultdict(list)

    for epoch in range(1, epochs+1):
        print(f'#{epoch}/{epochs}:')

        lrs.append(get_lr(optimizer))
        
        model, optimizer, loss = train(model, optimizer, train_loader, criterion)
        losses['train'].append(loss)

        if not (epoch % val_every):
            loss, metrics_ = val(model, val_loader, criterion, metric_names=metric_names)
            losses['val'].append(loss)
            if metrics_ is not None:
                for name, value in metrics_.items():
                    metrics[name].append(value)
            
            if ((not save_only_best) or (loss < best_val_loss)):
                if not os.path.exists(chkp_folder):
                    os.makedirs(chkp_folder)
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'losses': losses,
                    },
                    os.path.join(chkp_folder, model_name, f'{model_name}#{epoch}.pt'),
                )
                best_val_loss = loss
            
            if scheduler:
                try:
                    scheduler.step()
                except:
                    scheduler.step(loss)

        if not (epoch % draw_every):
            clear_output(True)
            ww = 3 if separate_show else 2
            ww_metrics = 0
            if metric_names is not None:
                plot_ids_ = [
                    [key, metric_meta.get("plot id", 1)]
                    for key, metric_meta
                    in metric_names.items()
                ]
                ww_metrics = len(set(el[1] for el in plot_ids_))
                assert all(el[1] <= ww_metrics for el in plot_ids_)
                
                plot_ids = defaultdict(list)
                for el in plot_ids_:
                    plot_ids[el[1]].append(el[0])
                
            fig, ax = plt.subplots(1, ww + ww_metrics, figsize=(20, 10))
            fig.suptitle(f'#{epoch}/{epochs}:')

            plt.subplot(1, ww + ww_metrics, 1)
            plt.plot(losses['train'], 'r.-', label='train')
            if separate_show:
                plt.title('loss on train')
                plt.legend()
            plt.grid()

            if separate_show:
                plt.subplot(1, ww + ww_metrics, 2)
                plt.title('loss on validation')
                plt.grid()
            else:
                plt.title('losses')
            plt.plot(losses['val'], 'g.-', label='val')
            plt.legend()
            
            plt.subplot(1, ww + ww_metrics, ww)
            plt.title('learning rate')
            plt.plot(lrs, 'g.-', label='lr')
            plt.legend()
            plt.grid()
            
            if metric_names is not None:
                for plot_id, keys in plot_ids.items():
                    for key in keys:
                        plt.subplot(1, ww + ww_metrics, ww + plot_id)
                        plt.title(f'additional metrics #{plot_id}')
                        for name in metrics:
                            if key in name:
                                plt.plot(metrics[name], '.-', label=name)
                        plt.legend()
                        plt.grid()
            
            plt.show()
            
            if test_dataset is not None:
                show_examples(5, test_dataset, model, out_size=out_size)

        if min_lr and get_lr(optimizer) <= min_lr:
            print(f'Learning process ended with early stop after epoch {epoch}')
            break
    
    return model, optimizer, losses



def show_examples(count, dataset, model, out_size=1):
    for i in range(count):
        number = np.random.randint(0, len(dataset))
        if dataset.annotate:
            img, mask, annotate = dataset[number]
            outputs = model(img[None].float().cuda(), annotate[None].float().cuda())
        else:
            img, mask = dataset[number]
            outputs = model(img[None].float().cuda())
            if type(mask) == bool:
                res = (outputs >= 0.5).item()
                print("pred", res, "real", mask)
                fig, axs = plt.subplots(nrows= 1 , ncols= 1, figsize=(5, 5))
                axs.imshow(img.numpy().transpose(1, 2, 0))
                plt.show()
                continue 
        pred = outputs.cpu() 
        if out_size == 1:
            pred = nn.functional.sigmoid(pred.permute(0, 2, 3, 1))
            pred = pred.view(pred.shape[0], pred.shape[1], -1)
            res = (pred >= 0.5).float()
        else:
            pred = pred.permute(0, 2, 3, 1)
            res = torch.max(pred, dim=-1).indices
        res_for_plt = res.repeat_interleave(4, dim=2).repeat_interleave(4, dim=1).repeat(3, 1, 1).numpy().transpose((1, 2, 0)).astype('float32')
        fig, axs = plt.subplots(nrows= 1 , ncols= 3, figsize=(15, 15))
        axs[0].imshow(res_for_plt)
        axs[1].imshow(mask.numpy())
        axs[2].imshow(img.numpy().transpose(1, 2, 0))
        plt.show()