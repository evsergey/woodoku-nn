import copy
import pywood
import torch
import numpy as np
from collections import OrderedDict

TEST_GAMES_COUNT = 200
TEST_GAMES_LIMIT = 12000
TRAIN_GAMES_COUNT = 10000
TRAIN_BATCH_SIZE = 100
TRAIN_GAMES_LIMIT = 1200
RANDOM_MOVES = 0
np.random.seed(3)

with open('figures.txt', "rt") as file:
    figures = pywood.Figure.read(file.read())
print(f'{len(figures)} figures loaded')
TEST_GAMES = np.random.randint(len(figures), size=(TEST_GAMES_COUNT, TEST_GAMES_LIMIT))
TEST_SEEDS = np.random.randint(1, 10000, size=TEST_GAMES_COUNT)
TRAIN_GAMES = np.random.randint(len(figures), size=(TRAIN_GAMES_COUNT, TRAIN_GAMES_LIMIT))

class RandomPolicy:
    def play(self, field, figs):
        field = field.copy()
        figs = [figures[f] for f in figs]
        total_weight = sum([f.weight for f in figs])
        score = field.add_random(figs, np.random.randint(1, 10000))
        return None if score < total_weight else field, score, 1

class TrivialPolicy:
    def play(self, field, figs, need_choice = False):
        field = field.copy()
        figs = [figures[f] for f in figs]
        total_weight = sum([f.weight for f in figs])
        if need_choice:
            next_fields, choices, score = field.get_all_next(figs, True)
        else:
            next_fields, score = field.get_all_next(figs)
        if score < total_weight:
            return (None, None, score, 0) if need_choice else (None, score, 0)
        if len(next_fields) == 1:
            best = 0
        else:
            t = next_fields.to_numpy('int8')
            t = t.reshape(-1,81)
            t = np.sum(t, axis=1)
            best = int(np.argmin(t))
        if need_choice:
            return next_fields[best], choices[best], score, len(next_fields)
        else:
            return next_fields[best], score, len(next_fields)


def evaluate_policy(policy):
    total_score = 0
    for game in range(0, TEST_GAMES_COUNT):
        field = pywood.Field()
        field.add_random([figures[f] for f in TEST_GAMES[game, :RANDOM_MOVES]], TEST_SEEDS[game])
        triples = TEST_GAMES[game, RANDOM_MOVES:].reshape(-1, 3)
        game_score = 0
        for i in range(triples.shape[0]):
            field, score = policy.play(field, triples[i, :])[:2]
            game_score += score
            if (i+1)%250 == 0:
                print(game_score)
            if field is None:
                break
        total_score += game_score
        print(f'game={game+1}, game_score={game_score}')
    return total_score / TEST_GAMES_COUNT

class TorchPolicy:
    def __init__(self, file=None, central=False, device=torch.device('cpu')):
#        self.nn = torch.nn.Sequential(OrderedDict([
#            ('flatten', torch.nn.Flatten()),
#            ('linear1', torch.nn.Linear(81, 512)),
#            ('dropout1', torch.nn.Dropout(p=0.5)),
#            ('relu1', torch.nn.ReLU()),
#            ('linear2', torch.nn.Linear(512, 512)),
#            ('dropout2', torch.nn.Dropout(p=0.5)),
#            ('relu2', torch.nn.ReLU()),
#            ('linear3', torch.nn.Linear(512, 64)),
#            ('dropout3', torch.nn.Dropout(p=0.3)),
#            ('relu3', torch.nn.ReLU()),
#            ('linear4', torch.nn.Linear(64, 8)),
#            ('dropout4', torch.nn.Dropout(p=0.2)),
#            ('relu4', torch.nn.ReLU()),
#            ('linear5', torch.nn.Linear(8, 1))
#            ]))
        self.nn = torch.nn.Sequential(
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
            torch.nn.Sequential() if central else torch.nn.Softplus()
            )
        if file:
            self.nn.load_state_dict(torch.load(file, map_location=device))
        self.nn = self.nn.to(device)
        self.device = device
        self.loss_fn = torch.nn.MSELoss()
        #self.optimizer = torch.optim.SGD(self.nn.parameters(), lr=3e-3, weight_decay=1e-4)
        self.epoches = 5
        self.split_size = 100
        self.best_score = 0
        self.best_model = None

    def set_score(self, score):
        if self.best_score < score:
            self.best_score = score
            self.best_state = copy.deepcopy(self.nn)
            print('Model is the best, saving')
            torch.save(self.nn.state_dict(), 'best_model.torch')

    def play(self, field, figs, need_choice = False):
        figs = [figures[f] for f in figs]
        total_weight = sum([f.weight for f in figs])
        if need_choice:
            next_fields, choices, score = field.get_all_next(figs, True)
        else:
            next_fields, score = field.get_all_next(figs)
        if score < total_weight:
            return (None, None, score, 0) if need_choice else (None, score, 0)
        if len(next_fields) == 1:
            best = 0
        else:
            next_fields.shrink(10000)
            t = next_fields.to_numpy('float32')
            t = torch.from_numpy(t).to(self.device)
            t = torch.cat([t, t.rot90(1, [1, 2]), t.rot90(2, [1, 2]), t.rot90(3, [1, 2])])
            t = torch.cat([t, t.flip(1)])
            predictions = self.nn(t)
            predictions = predictions.reshape(8, -1)
            predictions = predictions.sum(axis=0).T
            best = torch.argmax(predictions).item()
        if need_choice:
            return next_fields[best], choices[best], score, len(next_fields)
        else:
            return next_fields[best], score, len(next_fields)

    def train(self, fields, scores):
        if type(fields) == torch.Tensor:
            X = fields.numpy()
        else:
            X = fields.to_numpy('float32')
        X = np.concatenate([X, np.rot90(X, axes=(1, 2), k=1), np.rot90(X, axes=(1, 2), k=2), np.rot90(X, axes=(1, 2), k=3)], axis=0)
        X = np.concatenate([X, np.flip(X, axis=1)], axis=0)
        X = torch.from_numpy(X)
        if type(scores) == torch.Tensor:
            Y = scores.numpy()
        else:
            Y = np.array(scores, dtype='float32')
        Y = Y / np.mean(Y)
        Y = np.broadcast_to(Y, (8, Y.shape[0])).reshape(-1, 1)
        Y = torch.from_numpy(Y)
#        Y = self.nn(X).detach()*(1-self.blend) + Y * self.blend
        M = 0
        total_loss = 0
        for i in range(self.epoches):
            for (x, y) in zip(torch.split(X, self.split_size), torch.split(Y, self.split_size)):
                self.optimizer.zero_grad()
                pred = self.nn(x)
                loss = self.loss_fn(pred, y)
                print(loss.item())
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                M += 1
        #self.swa.update_parameters(self.nn)
        print(f'  loss: {total_loss / M}')

def play_game(policy, game, fields, scores, alts:list = None, moves:list = None, start_field = None):
    field = start_field or pywood.Field()
    old_len = len(scores)

    if game is None:
        def get_triple(i):
            return np.random.randint(len(figures), size=3).tolist()
    else:
        triples = game.reshape(-1, 3)
        def get_triple(i):
            return triples[i,:] if i < triples.shape[0] else None

    total_alternatives = 0
    i = 0
    while True:
        triple = get_triple(i)
        i += 1
        if triple is None:
            break
        if not any([field.can_place(figures[f]) for f in triple]):
            continue
        fields.append(field)
        field, score, alternatives = policy.play(field, triple)
        total_alternatives += alternatives
        scores.append(score)
        if alts is not None:
            alts.append(alternatives)
        if moves is not None:
            moves.append(0)
        if field is None:
            break
    s = 0
    for i in range(1, len(scores) - old_len + 1):
        s += scores[-i]
        scores[-i] = s
    if moves is not None:
        for i in range(1, len(scores) - old_len + 1):
            moves[-i] = i
    return s, total_alternatives

def play_games(policy, games, with_replay=True, start_fields = None):
    fields, scores, alts, moves = pywood.Fields(), [], [], []
    if type(games) == int:
        np.random.seed()
    if start_fields is None:
        total_s = 0
        for i in range(0, games if type(games) == int else games.shape[0]):
            print(i)
            s, _ = play_game(policy, None if type(games) == int else games[i, :], fields, scores, alts, moves, start_field=None)
            total_s += s
            if with_replay:
                replay_fields = list(fields[-7:-2])
                for j, replay_field in enumerate(replay_fields):
                    _ = play_game(policy, None if type(games) == int else games[(i+j+1) % games.shape[0], :], fields, scores, alts, moves, start_field = replay_field)
        print(f"average score = {total_s / games}")
    else:
        if type(start_fields) == torch.Tensor:
            start_fields = [
                pywood.Field("".join(
                    ['x' if x>0 else '.' for x in start_fields[i,:,:].reshape(-1)]))
                for i in range(start_fields.shape[0])]
            for k, start_field in enumerate(start_fields):
                print(k)
                for i in range(games):
                    score, _ = play_game(policy, None, [], [], None, None, start_field=start_field)
                    scores.append(score)
                sv = np.array(scores[-games:])
                print(f"Mean scores={np.mean(sv)}, median={np.median(sv)}, std={np.std(sv)}")
            fields = start_fields
    return fields, scores, alts, moves

def train_policy(policy, with_replay=True):
    fields = pywood.Fields()
    scores = []
    total_score = 0
    total_alternatives = 0
    for i in range(0, TRAIN_GAMES_COUNT):
        game = TRAIN_GAMES[i, :]
        score, alternatives = play_game(policy, game, fields, scores)
        total_score += score
        total_alternatives += alternatives
        if with_replay and len(fields) > 10:
            for j, replay in enumerate(list(fields[-7:-2])):
                _ = play_game(policy, TRAIN_GAMES[(i+j+1)%TRAIN_GAMES_COUNT, :], fields, scores, start_field = replay)
        if (i+1) % TRAIN_BATCH_SIZE == 0:
            print(f'i={i}, avg_fields={len(fields)/TRAIN_BATCH_SIZE}, avg_score={total_score/TRAIN_BATCH_SIZE}, avg_alt={total_alternatives/len(fields)}')
            policy.set_score(total_score)
            policy.train(fields, scores)
            fields.clear()
            scores.clear()
            total_score = 0
            total_alternatives = 0

if __name__ == '__main__':
    policy = TorchPolicy('epoch1.torch')
    #val = evaluate_policy(policy)
    #print(f'Pre training val: {val}')
    train_policy(policy, False)
    #val = evaluate_policy(policy)
    #print(f'Post training val: {val}')
