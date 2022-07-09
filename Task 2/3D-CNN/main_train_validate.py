import torch

def train(
    dataloader, 
    voxelizer,
    gaussian_filter,
    model,
    loss_fn, 
    optimizer, 
    device
):

    """
    Primary training function.

    Args:
        dataloader:
            An instance of Dataloader using the protease train dataset
        voxelizer: 
            An instance of Voxelizer3D from voxelizer.py
        gaussian_filter:
            An instance of GaussianFilter from gaussian_filter.py
        model:
            An instance of Model_3DCNN from model.py
        loss_fn:
            A torch.nn loss function object
        optimizer:
            A torch.optim optimizer object
        device:
            expects "cpu" or "cuda" as device specification.

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

    for batch_idx, batch_data in enumerate(dataloader):

        # pass inputs and labels to gpu
        inputs_cpu, labels_cpu = batch_data
        inputs, labels = inputs_cpu.to(device), labels_cpu.to(device)

        # loop over individual batch elements
        vol_batch = torch.zeros( (batch_size,19,48,48,48), dtype=torch.float, device=device)
        for i in range(inputs.shape[0]):
            xyz, feat = inputs[i,:,:3], inputs [i,:,3:]
            vol_batch[i,:,:,:,:] = voxelizer(xyz,feat)

        # forward step     
        vol_batch = gaussian_filter(vol_batch)
        pred, _ = model(vol_batch)
        loss = loss_fn(pred, labels)
        loss_record = loss.cpu().data.item()
        losses.append(loss_record)

        # backward step 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if batch_idx % 100 == 0:
            current = batch_idx*len(inputs)
            print(f"loss: {loss_record:>7f} [{current:>5d}/{size:>5d}]")
        
    return losses



def validate(
    dataloader, 
    voxelizer,
    gaussian_filter,
    model,
    loss_fn,  
    device
):

    """
    Primary Validation function.

    Args:
        dataloader:
            An instance of Dataloader using the protease train dataset,
        voxelizer: 
            An instance of Voxelizer3D from voxelizer.py
        gaussian_filter:
            An instance of GaussianFilter from gaussian_filter.py
        model:
            An instance of Model_3DCNN from model.py
        loss_fn:
            A torch.nn loss function object.
        optimizer:
            A torch.optim optimizer object.
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

            vol_batch = torch.zeros( (batch_size,19,48,48,48), dtype=torch.float, device=device)
            for i in range(inputs.shape[0]):
                xyz, feat = inputs[i,:,:3], inputs [i,:,3:]
                vol_batch[i,:,:,:,:] = voxelizer(xyz,feat)

            # forward step     
            vol_batch = gaussian_filter(vol_batch)
            pred, _ = model(vol_batch)
            loss = loss_fn(pred, labels)
            avg_loss += loss.cpu().data.item()

            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()

    avg_loss /= num_batches   
    accuracy /= size
    print(f"Validation Error:\n Accuracy: {(100*accuracy):>0.1f} %, Avg loss:{avg_loss:>8f} \n")
    return avg_loss, accuracy