import torch.optim as optim
import torch
from tqdm import tqdm


def train(epochs, trainloader, valloader, optimizer, criterion, NET, PATH):

    train_set_loss = []
    val_set_loss = []
    number_of_epochs = []
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        NET.train()
        running_loss = 0.0
        for idx, data in enumerate(trainloader):
            gray, color = data
            optimizer.zero_grad()
            outputs = NET(gray)
            loss = criterion(outputs, color)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # if idx % 5000 == 4999:
            #     print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(epoch, idx * len(data),
            #             len(trainloader), 100.0 * idx / len(trainloader), loss.item()))

        train_set_loss.append(running_loss / len(trainloader))
        number_of_epochs.append(epoch)
        val_loss = val(valloader, criterion, NET)
        desciption = "Val Loss: {:.6f}".format(val_loss)
        pbar.set_description(desciption)
        val_set_loss.append(val_loss)
        running_loss = 0.0

    torch.save(NET.state_dict(), PATH)
    print("Finished Training")
    return train_set_loss, val_set_loss, number_of_epochs


def val(valloader, criterion, NET):
    NET.eval()
    val_loss = 0
    with torch.no_grad():
        for idx, data in enumerate(valloader):
            gray, color = data

            outputs = NET(gray)
            loss = criterion(outputs, color)
            val_loss += loss.item()

        val_loss /= len(valloader)
    # print("\nvalidation set: Average loss: {:.4f}".format(val_loss))
    return val_loss
