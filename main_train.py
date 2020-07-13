import os
import numpy as np
import time

import torch
import torch.nn
import torch.nn.functional

import data_loader


def train_one_epoch(device, model, optimizer, criterion, train_loader):
    model.train()
    loss_epoch = 0.
    for b, (batch_input, batch_label) in enumerate(train_loader):
        for i in range(len(batch_input)):
            # reset gradient history
            optimizer.zero_grad()
            # read data
            data_input, data_label = batch_input[i], batch_label[i] - 1
            # feed
            output = model(data_input).view(1, -1)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            loss = criterion(output, data_label)
            loss.backward()
            optimizer.step()
            loss_epoch = loss.item()
        print("\r\ttrain batch:{}/{}".format(b, len(train_loader)), end="")
    return round(loss_epoch, 4)


def validate_one_epoch(device, model, criterion, valid_loader):
    model.eval()
    num_valid = len(valid_loader.sampler.indices)
    if num_valid == 0:
        raise FileNotFoundError("number of data is 0")
    val_loss = 0.
    num_correct = 0
    for b, (batch_input, batch_label) in enumerate(valid_loader):
        for i in range(len(batch_input)):
            # read data
            data_input, data_label = batch_input[i], batch_label[i] - 1
            # feed
            output = model(data_input).view(1, -1)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            # record fitness
            val_loss += criterion(output, data_label).item()
            if torch.max(output, 1)[1] == data_label:
                num_correct += 1
        print("\r\tvalid batch:{}/{}".format(b, len(valid_loader)), end="")

    val_loss /= num_valid
    num_correct /= num_valid
    return round(val_loss, 4), round(num_correct*100, 4)


def validate_train_loop(device, model, optimizer, scheduler, criterion, valid_loader, train_loader,
                        num_epoch, num_epoch_per_valid, save_path_format, start_epoch=0):
    result = validate_one_epoch(device, model, criterion, valid_loader)
    print("\rvalid loss:{} accuracy:{}%".format(*result))

    for epoch in range(start_epoch, start_epoch+num_epoch):
        result = train_one_epoch(device, model, optimizer, criterion, train_loader)
        print("\rtrain epoch:{} loss:{}".format(epoch, result))
        if (epoch + 1) % num_epoch_per_valid == 0:
            result = validate_one_epoch(device, model, criterion, valid_loader)
            print("\rvalid loss:{} accuracy:{}%".format(*result))
        scheduler.step()
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict()
        }, save_path_format.format(epoch))


def main():
    timer = time.time()
    # generate guess data
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # prepare neural network
    model = torch.nn.Sequential(
        torch.nn.LeakyReLU(),
        torch.nn.Linear(500, 250),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(250, 100),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(100, 17),
        torch.nn.LeakyReLU(),
    )
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.75)
    criterion = torch.nn.CrossEntropyLoss()

    # state_dict = torch.load("state_dicts/state_dict_9")
    # model.load_state_dict(state_dict["model"])
    # optimizer.load_state_dict(state_dict["optimizer"])
    # scheduler.load_state_dict(state_dict["scheduler"])

    # prepare data
    dataset = data_loader.DatasetBandStructureToPlaneGroup(device, ["data/cluster_1/"])
    valid_loader, train_loader = data_loader.get_valid_train_loader(dataset, batch_size=50, valid_size=.1)

    torch.backends.cudnn.benchmark = True

    # train
    validate_train_loop(
        device, model, optimizer, scheduler, criterion, valid_loader, train_loader,
        num_epoch=10, num_epoch_per_valid=5, save_path_format="state_dicts/state_dict_{}", start_epoch=0
    )

    print(str(time.time() - timer) + "s")

    import winsound
    winsound.Beep(200, 500)


if __name__ == "__main__":
    main()
