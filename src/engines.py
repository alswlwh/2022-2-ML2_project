import torch

from torchmetrics.aggregation import MeanMetric


def train(loader, model, optimizer, scheduler, loss_fn, metric_fn, device):
    model.train()
    loss_mean = MeanMetric()
    metric_mean = MeanMetric()
    
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        
        loss = loss_fn(outputs, targets)
        metric = metric_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_mean.update(loss.to('cpu'))
        metric_mean.update(metric.to('cpu'))

        scheduler.step()

    summary = {'loss': loss_mean.compute(), 'metric': metric_mean.compute()}

    return summary


def evaluate(loader, model, loss_fn, metric_fn, device):
    model.eval()
    loss_mean = MeanMetric()
    metric_mean = MeanMetric()
    
    y_pred = []
    y_true = []
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            
        loss = loss_fn(outputs, targets)
        metric = metric_fn(outputs, targets)

        loss_mean.update(loss.to('cpu'))
        metric_mean.update(metric.to('cpu'))
        
        # Save Prediction
        outputs = (torch.max(outputs, 1)[1]).data.cpu().numpy()
        y_pred.extend(outputs) 

        # Save Truth
        targets = targets.data.cpu().numpy()
        y_true.extend(targets) 
    
    summary = {'loss': loss_mean.compute(), 'metric': metric_mean.compute()}

    return summary, y_pred, y_true        
        
def pl_train(train_loader, unlabeled_loader, model, optimizer, scheduler, loss_fn, metric_fn, device, epoch, step, alpha_weight):
    model.train()
    loss_mean = MeanMetric()
    metric_mean = MeanMetric()

    for batch_idx, x_unlabeled in enumerate(unlabeled_loader):
        x_unlabeled = x_unlabeled.to(device)

        model.eval()
        output_unlabeled = model(x_unlabeled)
        _, pseudo_labeled = torch.max(output_unlabeled, 1)
        model.train()
        
        outputs = model(x_unlabeled)
        unlabeled_loss = alpha_weight(step) * loss_fn(outputs, pseudo_labeled)
        unlabeled_metric = metric_fn(outputs, pseudo_labeled)

        optimizer.zero_grad()
        unlabeled_loss.backward()
        optimizer.step()
        
        loss_mean.update(unlabeled_loss.to('cpu'))
        metric_mean.update(unlabeled_metric.to('cpu'))

        scheduler.step()
        
        # For every 128 batches train one epoch on labeled data 
        if batch_idx % 128 == 0:
                
            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)

                loss = loss_fn(outputs, targets)
                metric = metric_fn(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_mean.update(loss.to('cpu'))
                metric_mean.update(metric.to('cpu'))

                scheduler.step()
                    
            step += 1

    summary = {'loss': loss_mean.compute(), 'metric': metric_mean.compute()}

    return summary, step

def pl_evaluate(loader, model, loss_fn, metric_fn, device):
    model.eval()
    loss_mean = MeanMetric()
    metric_mean = MeanMetric()
    
    y_pred = []
    y_true = []
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            
        loss = loss_fn(outputs, targets)
        metric = metric_fn(outputs, targets)

        loss_mean.update(loss.to('cpu'))
        metric_mean.update(metric.to('cpu'))
        
        # Save Prediction
        outputs = (torch.max(outputs, 1)[1]).data.cpu().numpy()
        y_pred.extend(outputs) 
        
        # Save Truth
        targets = targets.data.cpu().numpy()
        y_true.extend(targets) 
    
    summary = {'loss': loss_mean.compute(), 'metric': metric_mean.compute()}

    return summary, y_pred, y_true