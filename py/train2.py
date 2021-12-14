import torch
import numpy as np
import gradinit
from collections import OrderedDict

cuda = torch.device('cuda:0')
cpu = torch.device('cpu')

def load_data(file_name = 'games.torch', shuffle=True):
    data = torch.load(file_name, map_location=cpu)
    fields = data['fields']
    mask = 2 ** torch.arange(0, 27, dtype = torch.int32)
    fields = fields.reshape(-1, 1).bitwise_and(mask).ne(0).reshape(-1, 9, 9).float()
    scores = data['scores']
    scores = torch.log(scores)
    std_scores, mean_scores = torch.std_mean(scores)
    scores = (scores - mean_scores) / std_scores
    print(f'Loaded: {fields.shape[0]} positions. Mean log scores value: {mean_scores.item()}. Std: {std_scores.item()}')
    test_size = fields.shape[0] // 2500 * 125
    train_size = fields.shape[0] // 125 * 125
    test_fields = fields[-test_size:train_size,:,:]
    test_scores = scores[-test_size:train_size,:]
    fields = fields[:-test_size,:,:]
    scores = scores[:-test_size,:]
    train_size -= test_size
    print(f'Train size: {train_size}. Test size: {test_size}')
    fields = torch.cat([fields, fields.rot90(1, [1, 2]), fields.rot90(2, [1, 2]), fields.rot90(3, [1, 2])])
    fields = torch.cat([fields, fields.flip(1)])
    test_fields = torch.cat([test_fields, test_fields.rot90(1, [1, 2]), test_fields.rot90(2, [1, 2]), test_fields.rot90(3, [1, 2])])
    test_fields = torch.cat([test_fields, test_fields.flip(1)])
    if 'broadcast_to' in dir(torch):
        scores = torch.broadcast_to(scores, (8, scores.shape[0])).reshape(-1, 1)
        test_scores = torch.broadcast_to(test_scores, (8, test_scores.shape[0])).reshape(-1, 1)
    else:
        scores = torch.cat([scores]*8).reshape(-1, 1)
        test_scores = torch.cat([test_scores]*8).reshape(-1, 1)
    test_size *= 8
    train_size *= 8
    print(f'Augmented to: {train_size + test_size} positions')
    if shuffle:
        perm = torch.randperm(fields.shape[0])
        fields = fields[perm, :, :]
        scores = scores[perm, :]
        del perm
        print('Random permute completed')
    return fields, scores, test_fields, test_scores

model = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(81, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 8),
    torch.nn.ReLU(),
    torch.nn.Linear(8, 1),
    )

EPOCH_NUMBER = 50
BATCH_SIZE = 100

def train_model(model, fields, scores, test_fields, test_scores, shuffle=True, epoch_number=EPOCH_NUMBER):
    loss_fn = torch.nn.MSELoss()
    model.eval()
    test_pred = model(test_fields)
    test_loss = loss_fn(test_pred, test_scores)
    print(f'Test loss: {test_loss.item()}')
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=3e-3, weight_decay=1e-4)
    train_size = fields.shape[0]
    batch_count = train_size // BATCH_SIZE
    for epoch in range(epoch_number):
        if shuffle:
            perm = torch.randperm(train_size)
        TOTAL_loss = 0
        total_loss = 0
        for batch in range(batch_count):
            if shuffle:
                batch_i = perm[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE]
                x = fields[batch_i, :, :]
                y = scores[batch_i, :]
            else:
                x = fields[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE, :, :]
                y = scores[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE, :]
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            if (batch+1)%1000 == 0:
                print(f'Epoch: {epoch+1}/{epoch_number}. Batch: {batch+1}/{batch_count}.\tLoss: {total_loss / 1000}')
                TOTAL_loss += total_loss
                total_loss = 0
        model.eval()
        test_pred = model(test_fields)
        test_loss = loss_fn(test_pred, test_scores)
        print(f'Test loss: {test_loss.item()}')
        print(f'Train loss: {(TOTAL_loss+total_loss) / batch_count}')
        torch.save(model.state_dict(), f'epoch{epoch}.torch')
        model.train()

def save_model(model, file_name):
    for param in model.parameters():
        param.requires_grad = False
    mask = 2 ** torch.arange(0, 27, dtype = torch.int32)
    def solve(batch):
        batch = batch.reshape(-1, 1).bitwise_and(mask).ne(0).reshape(-1, 9, 9).float()
        batch = torch.cat([batch, batch.rot90(1, [1, 2]), batch.rot90(2, [1, 2]), batch.rot90(3, [1, 2])])
        batch = torch.cat([batch, batch.flip(1)])
        predictions = model(batch)
        predictions = predictions.reshape(8, -1)
        predictions = predictions.sum(axis=0)
        return predictions
    example = torch.randint(0, 1<<27, (5, 3), dtype=torch.int32)
    script = torch.jit.trace(solve, example)
    script.save(file_name)


if __name__ == "__main__":
    #model.load_state_dict(torch.load('1700.torch'))
    data = load_data('games20k.torch')
    train_model(model, *data)

