import torch
import numpy as np
import gradinit
from collections import OrderedDict

cuda = torch.device('cuda:0')
cpu = torch.device('cpu')

def load_data(file_name = 'games.torch', shuffle=True):
    data = torch.load(file_name, map_location=cpu)
    fields = data['fields']
    scores = data['scores']
    scores = torch.log(scores)
    std_scores, mean_scores = torch.std_mean(scores)
    scores = (scores - mean_scores) / std_scores
    print(f'Loaded: {fields.shape[0]} positions. Mean log scores value: {mean_scores.item()}. Std: {std_scores.item()}')
    fields = torch.cat([fields, fields.rot90(1, [1, 2]), fields.rot90(2, [1, 2]), fields.rot90(3, [1, 2])])
    fields = torch.cat([fields, fields.flip(1)])
    if 'broadcast_to' in dir(torch):
        scores = torch.broadcast_to(scores, (8, scores.shape[0])).reshape(-1, 1)
    else:
        scores = torch.cat([scores]*8).reshape(-1, 1)
    print(f'Augmented to: {fields.shape[0]} positions')
    if shuffle:
        perm = torch.randperm(fields.shape[0])
        fields = fields[perm, :, :]
        scores = scores[perm, :]
        del perm
        print('Random permute completed')
    test_size = fields.shape[0] // 20000 * 1000
    train_size = fields.shape[0] // 1000 * 1000
    test_fields = fields[-test_size:train_size,:,:]
    test_scores = scores[-test_size:train_size,:]
    fields = fields[:-test_size,:,:]
    scores = scores[:-test_size,:]
    train_size -= test_size
    print(f'Train size: {train_size}. Test size: {test_size}')
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
#    torch.nn.Softplus()
    )

#model = torch.nn.Sequential(OrderedDict([
#    ('flatten', torch.nn.Flatten()),
#    ('linear1', torch.nn.Linear(81, 512)),
#    ('dropout1', torch.nn.Dropout(p=0.5)),
#    ('relu1', torch.nn.ReLU()),
#    ('linear2', torch.nn.Linear(512, 512)),
#    ('dropout2', torch.nn.Dropout(p=0.5)),
#    ('relu2', torch.nn.ReLU()),
#    ('linear3', torch.nn.Linear(512, 64)),
#    ('dropout3', torch.nn.Dropout(p=0.3)),
#    ('relu3', torch.nn.ReLU()),
#    ('linear4', torch.nn.Linear(64, 8)),
#    ('dropout4', torch.nn.Dropout(p=0.2)),
#    ('relu4', torch.nn.ReLU()),
#    ('linear5', torch.nn.Linear(8, 1))
#    ]))
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
            #loss = torch.mean(torch.square(pred-y)/(pred + y + 0.1)) # loss_fn(pred, y)
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

if __name__ == "__main__":
    #model.load_state_dict(torch.load('1700.torch'))
    data = load_data('games20k.torch')
    train_model(model, *data)

