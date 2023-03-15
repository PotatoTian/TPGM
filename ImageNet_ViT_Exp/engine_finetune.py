import torch
import utils.misc as misc
from utils.class_sampling import class_subsampling
from timm.utils import accuracy

@torch.no_grad()
def evaluate(data_loader, dataset_name, model, device):
    
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"

    if dataset_name == "imagenet-r" or dataset_name == "imagenet-a":
        CLASS_SUBLIST = class_subsampling(dataset_name)

    # switch to evaluation mode
    model.eval()
    

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        output = model(images)

        if dataset_name == "imagenet-r" or dataset_name == "imagenet-a":
            new_target = []
            for t in target:
                new_target.append(CLASS_SUBLIST.index(t))
            target = torch.tensor(new_target).to(device, non_blocking=True)
            output = output[:,CLASS_SUBLIST]
        else:
            target = target.to(device, non_blocking=True)
        
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
    print(
        "* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}".format(
            top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss
        )
    )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
