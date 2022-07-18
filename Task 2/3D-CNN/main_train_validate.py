import torch

def train(
    dataloader, 
    data_transform,
    model,
    loss_fn, 
    optimizer, 
    device, 
    scheduler=None
):

    """
    Primary training function.

    Args:
        dataloader:
            An instance of Dataloader using the protease train dataset
        data_transform:
            A data transformation layers that returns voxels from the dataloader
        model:
            An instance of Model_3DCNN from model.py
        loss_fn:
            A torch.nn loss function object
        optimizer:
            A torch.optim optimizer object
        device:
            expects "cpu" or "cuda" as device specification.
        scheduler:
            One may optionally scpecify a scheduler. If not, the it will be set as "None" and will not be updated 
            in the optimization loop.

    Returns:
    --------
    losses: 
        A list of losses from each batch computation. 

    """

    # initialize batch data
    size = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    losses = [] 
    
    # model setup
    model.to(device)
    model.train()
    vol_dim = data_transform.vol_dim

    for batch_idx, batch_data in enumerate(dataloader):

        # pass inputs and labels to gpu
        inputs_cpu, labels_cpu = batch_data
        inputs, labels = inputs_cpu.to(device), labels_cpu.to(device)

        vol_batch = data_transform(inputs)
        pred, _ = model(vol_batch)
        loss = loss_fn(pred, labels)
        loss_record = loss.cpu().data.item()
        losses.append(loss_record)

        # backward step 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()


        if batch_idx % 50 == 0:
            current = batch_idx*len(inputs)
            print(f"loss: {loss_record:>7f} [{current:>5d}/{size:>5d}]")
        
    return losses



def validate(
    dataloader, 
    data_transform,
    model,
    loss_fn,  
    device
):

    """
    Primary Validation function.

    Args:
        dataloader:
            An instance of Dataloader using the protease train dataset,
        data_transform:
            A data transformation layers that returns voxels from the dataloader
        model:
            An instance of Model_3DCNN from model.py
        loss_fn:
            A torch.nn loss function object.
        device:
            expects "cpu" or "cuda" as device specification.

    Returns:
    --------
    avg_losses: 
        The average of losses computed across each batch.
    accuracy:
        The percent of correct classifications made on the valiation set.
    """
    # initialize loop
    size = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    avg_loss, accuracy = 0.0, 0.0

    num_batches = len(dataloader)

    # setup model
    model.eval()
    model.to(device)

    for batch_idx, batch_data in enumerate(dataloader):

        # pass inputs and labels to gpu
        inputs_cpu, labels_cpu = batch_data
        inputs, labels = inputs_cpu.to(device), labels_cpu.to(device)

        # loop over individual batch elements
        with torch.no_grad():
            
            vol_batch = data_transform(inputs)
            
            # forward step     
            pred, _ = model(vol_batch)
            loss = loss_fn(pred, labels)
            avg_loss += loss.cpu().data.item()

            accuracy += (pred.argmax(1) == labels).type(torch.float).sum().item()

    avg_loss /= num_batches   
    accuracy /= size
    print(f"Validation Error:\n Accuracy: {(100*accuracy):>0.1f} %, Avg loss:{avg_loss:>8f} \n")
    return avg_loss, accuracy