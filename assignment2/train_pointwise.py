import dataset
import utils
import numpy as np
import torch
import argparse
from pointwise import PointwiseLTRModel
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scheduler import CycleScheduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--epochs', type=int, default=50, help='amount of epochs')
    PARSER.add_argument('--batch_size', type=int, default=128, help='training batch size')
    PARSER.add_argument('--lr', type=float, default=0.001,help='Adam optimizer learning rate')
    ARGS = PARSER.parse_args()

    datafold = dataset.get_dataset().get_data_folds()[0]
    datafold.read_data()
    print("Read data")
    traindata = OurDataSet(datafold.train)
    valdata = OurDataSet(datafold.validation)

    validation_dataloader = DataLoader(valdata, batch_size=32, shuffle=False)
    train_dataloader = DataLoader(traindata, batch_size=ARGS.batch_size, shuffle=True)
    print("Create dataloaders")


    model = PointwiseLTRModel(datafold.num_features,the_layer_amount).to(device)
    train(model, train_dataloader, validation_dataloader, ARGS)

    torch.save(model.state_dict(), "pickles/pointwisemodel.pt")

    """
    with open("pickles/results" + paramstr + ".csv", 'w') as f:
        f.write("metric,value\n")
        for loss in losses:
            f.write("loss," + str(loss) + "\n")
        for t in trainndcg:
            f.write("trainn," + str(t) + "\n")
        for t in valndcg:
            f.write("valn," + str(t) + "\n")
    """

def train(model, train_dataloader,
        validation_dataloader, ARGS):
    optimizer = torch.optim.Adam(model.parameters(), lr=ARGS.learning_rate)
    criterion = nn.CrossEntropyLoss()
    #scheduler = CycleScheduler(optimizer, ARGS.lr, n_iter=len(train_dataloader), momentum=None)
    total_losses = []
    #total_train_ndcg = []
    total_validation_ndcg = []
    #prev_validation = 0
    #reasons_to_kill = 0

    for epoch in range(ARGS.epochs):
        losses = []
        model.train()
        for i, (x, y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)
            # Pass through the model
            scores = model(x)

            # Calculate squared two norm loss for regression
            loss = criterion(scores, y)
            loss.backward()
            optimizer.step()
            #scheduler.step()

        model.eval()
        mean_loss = torch.mean(torch.tensor(losses))
        total_losses.append(mean_loss.item())

        val_ndcg = calculate_only_ndcg(model, validation_dataloader)
        total_validation_ndcg.append(val_ndcg)
        print(epoch, "loss", mean_loss.item() , "val ndcg", val_ndcg)

    return total_validation_ndcg



if __name__ == '__main__':
    main()
