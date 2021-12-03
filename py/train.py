import copy
import pywood
import torch
import numpy as np

TEST_GAMES_COUNT = 200
TEST_GAMES_LIMIT = 900
TRAIN_GAMES_COUNT = 10000
TRAIN_BATCH_SIZE = 100
TRAIN_GAMES_LIMIT = 300
RANDOM_MOVES = 6
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

def evaluate_policy(policy):
    total_score = 0
    for game in range(0, TEST_GAMES_COUNT):
#        print(f'game={game}')
        field = pywood.Field()
        field.add_random([figures[f] for f in TEST_GAMES[game, :RANDOM_MOVES]], TEST_SEEDS[game])
        triples = TEST_GAMES[game, RANDOM_MOVES:].reshape(-1, 3)
        for i in range(triples.shape[0]):
            field, score = policy.play(field, triples[i, :])[:2]
            total_score += score
            if field is None:
                break
    return total_score / TEST_GAMES_COUNT

class TorchPolicy:
    def __init__(self, file=None):
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
            torch.nn.Softplus()
            )
        if file:
            self.nn.load_state_dict(torch.load(file))
        self.swa = torch.optim.swa_utils.AveragedModel(self.nn)
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.nn.parameters(), lr=3e-3, weight_decay=1e-4)
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
            t = next_fields.to_numpy('float32')
            t = torch.from_numpy(t)
            predictions = self.nn(t)
            best = torch.argmax(predictions).item()
        if need_choice:
            return next_fields[best], choices[best], score, len(next_fields)
        else:
            return next_fields[best], score, len(next_fields)

    def train(self, fields, scores):
        X = fields.to_numpy('float32')
        X = np.concatenate([X, np.rot90(X, axes=(1, 2), k=1), np.rot90(X, axes=(1, 2), k=2), np.rot90(X, axes=(1, 2), k=3)], axis=0)
        X = np.concatenate([X, np.flip(X, axis=1)], axis=0)
        X = torch.from_numpy(X)
        Y = np.array(scores, dtype='float32')
        Y = Y / np.mean(Y)
        Y = np.broadcast_to(Y, (8, Y.shape[0])).reshape(-1, 1)
        Y = torch.from_numpy(Y)
#        Y = self.nn(X).detach()*(1-self.blend) + Y * self.blend
        for i in range(self.epoches):
            total_loss = 0
            for (x, y) in zip(torch.split(X, self.split_size), torch.split(Y, self.split_size)):
                self.optimizer.zero_grad()
                pred = self.nn(x)
                loss = self.loss_fn(pred, y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
        #self.swa.update_parameters(self.nn)
        print(f'  loss: {total_loss / Y.shape[0]}')

def play_game(policy, game, fields, scores):
    field = pywood.Field()
    field.add_random([figures[f] for f in game[:RANDOM_MOVES]])
    old_len = len(scores)
    triples = game[RANDOM_MOVES:].reshape(-1, 3)
    total_alternatives = 0
    for i in range(0, triples.shape[0]):
        if not any([field.can_place(figures[f]) for f in triples[i, :]]):
            continue
        fields.append(field)
        field, score, alternatives = policy.play(field, triples[i, :])
        total_alternatives += alternatives
        scores.append(score)
        if field is None:
            break
    s = 0
    for i in range(1, len(scores) - old_len + 1):
        s += scores[-i]
        scores[-i] = s
    return s, total_alternatives

def train_policy(policy):
    fields = pywood.Fields()
    scores = []
    total_score = 0
    total_alternatives = 0
    for i in range(0, TRAIN_GAMES_COUNT):
        game = TRAIN_GAMES[i, :]
        score, alternatives = play_game(policy, game, fields, scores)
        total_score += score
        total_alternatives += alternatives
        if (i+1) % TRAIN_BATCH_SIZE == 0:
            print(f'i={i}, avg_fields={len(fields)/TRAIN_BATCH_SIZE}, avg_score={total_score/TRAIN_BATCH_SIZE}, avg_alt={total_alternatives/len(fields)}')
            policy.set_score(total_score)
            policy.train(fields, scores)
            fields.clear()
            scores.clear()
            total_score = 0
            total_alternatives = 0

if __name__ == '__main__':
    policy = TorchPolicy()#('57.6k.torch')
    val = evaluate_policy(policy)
    print(f'Pre training val: {val}')
    train_policy(policy)
    val = evaluate_policy(policy)
    print(f'Post training val: {val}')
