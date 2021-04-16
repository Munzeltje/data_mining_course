import argparse

def LSTM_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--sequence_length', type=int, default=2)

    parser.add_argument('--input_dim', type=int, default=19)
    parser.add_argument('--hidden_dim', type=int, default=16)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=8e-4)
    parser.add_argument('--model_checkpoint', type=str, default=None)

    parser.add_argument('--print_every', type=int, default=1)

    parser.add_argument('--data_path', type=str, default="../data/dataset_train.csv")

    parser.add_argument('--val_data_path', type=str, default="../data/dataset_test.csv")

    parser.add_argument('--model_name', type=str, default="")


    args = parser.parse_args()

    print(args)
    with open("config.txt", 'w') as file:
        file.write(str(args))
    return args
