import torch.optim as optim
import torch


def train(epochs, trainloader, valloader, optimizer, criterion, NET):

    train_set_loss = []
    val_set_loss = []
    number_of_epochs = []

    for epoch in range(epochs):  # loop over the dataset multiple times
        NET.train()
        running_loss = 0.0
        for idx, data in enumerate(trainloader):
            # print(
            # "epoch: {} ,running_loss_start: {}".format(epoch, running_loss)
            # )
            # get the inputs; data is a list of [inputs, labels]
            gray, color = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = NET(gray)
            loss = criterion(outputs, color)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            # print((f"loss_item: {loss.item()} "))

            # print(("running_loss_end: {} ").format(running_loss))

            # # if idx % 2000 == 1999:  # print every 2000 mini-batches
            #     print(
            #         "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
            #             epoch,
            #             idx * len(data),
            #             len(trainloader),
            #             100.0 * idx / len(trainloader),
            #             loss.item(),
            #         )
            #     )

            if idx % 2000 == 1999:  # print every 2000 mini-batches
                print(
                    "Train Epoch: {} , len(data): {} , [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        len(data),
                        idx * len(data),
                        len(trainloader),
                        100.0 * idx / len(trainloader),
                        loss.item(),
                    )
                )

        train_set_loss.append(running_loss / len(trainloader))

        number_of_epochs.append(epoch)
        val_loss = val(valloader, criterion, NET)

        val_set_loss.append(val_loss)
        running_loss = 0.0
    print("Finished Training")
    return train_set_loss, val_set_loss, number_of_epochs


# new val function:
def val(valloader, criterion, NET):
    NET.eval()
    val_loss = 0
    with torch.no_grad():
        for idx, data in enumerate(valloader):
            gray, color = data
            # calculate outputs by running images through the network
            outputs = NET(gray)
            loss = criterion(outputs, color)
            val_loss += loss.item()
            # print(val_loss)
            # imshow(gray[3])
            # plt.figure()
            # imshow(outputs[3])
            # break
        val_loss /= len(valloader)
    print("\nvalidation set: Average loss: {:.4f}".format(val_loss))
    return val_loss
