from AlexNet import *
from train import *
from test import *
from data_split import *

if __name__ == '__main__':
    AlexNet_Model = AlexNet().to(device="cuda")

    # Initialize the loss function
    learning_rate = 1e-2
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(AlexNet_Model.parameters(), lr=learning_rate)

    epochs = 10
    Train_epoch_loss_list=[]
    Val_epoch_loss_list=[]
    Val_epoch_acc_list=[]

    #Get data
    train_dataloader, val_dataloader,test_dataloader = data_split()

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------") 
        tr_lss,vl_lss,vl_acc=train_loop(train_dataloader, val_dataloader, AlexNet_Model, loss_fn, optimizer)
        Train_epoch_loss_list.append(tr_lss)
        Val_epoch_loss_list.append(vl_lss)
        Val_epoch_acc_list.append(vl_acc)

    test_loop(test_dataloader, AlexNet_Model, loss_fn)
    print("Done!")