def tversky_index_loss(pred, real, a, b):
    axis = (1, 2)
    real = real.float()
    pred = pred.float().repeat_interleave(4, dim=2).repeat_interleave(4, dim=1)
    intersection = (pred * real).sum(dim=axis)
    difference_real = (real * (1 - pred)).sum(dim=axis)
    difference_pred = (pred * (1 - real)).sum(dim=axis)
    loss = intersection / (intersection + a * difference_real + b * difference_pred)
    return - loss.mean()

class TverskyIndexLoss(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self, pred, real):
        axis = (1, 2)
        real = real.float()
        pred = pred.float().repeat_interleave(4, dim=2).repeat_interleave(4, dim=1)
        intersection = (pred * real).sum(dim=axis)
        difference_real = (real * (1 - pred)).sum(dim=axis)
        difference_pred = (pred * (1 - real)).sum(dim=axis)
        loss = intersection / (intersection + self.a * difference_real + self.b * difference_pred)
        return 1 - loss.mean()

def dice_coef_loss(pred, real, reverse=False, val=False, increase=True, axis=(1, 2)):
    # axis = (1, 2)
    if increase:
        pred = pred.cpu().repeat_interleave(4, dim=2).repeat_interleave(4, dim=1).numpy()
    real = real.cpu().numpy()
    if reverse:
        pred = 1 - pred
        real = 1 - real
    intersection = (2 * pred * real).sum(axis=axis)
    unification = pred.sum(axis=axis) + real.sum(axis=axis)
    return (intersection/ (unification)).mean()

def prepare_for_entropy_loss(pred, mask, val=False):
    log_pred = nn.functional \
            .log_softmax(pred.permute(0, 2, 3, 1), dim=3) \
            .repeat_interleave(4, dim=2) \
            .repeat_interleave(4, dim=1) \

    log_pred = log_pred.flatten(start_dim=0, end_dim=2)
    return log_pred, mask_batch.flatten(start_dim=0, end_dim=2).long()

def prepare_for_dice_loss(pred, mask, val=False):
    pred = nn.functional.sigmoid(pred.permute(0, 2, 3, 1))
    pred = pred.view(pred.shape[0], pred.shape[1], -1)
    return pred, mask

