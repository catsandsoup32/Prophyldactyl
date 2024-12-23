import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

from data.dataloader import LichessDataset
from models.models import HalfKP_NNUE
from util import fen_to_halfKP

MODE = 'HalfKP'

device = 'cuda' if torch.cuda.is_available else 'cpu'

train_dataset = LichessDataset(mode='train')
val_dataset = LichessDataset(mode='val')
test_dataset = LichessDataset(mode='test')

def HalfKP_collate(batch):
    fens, evals = zip(*batch)
    w, b, stm = [], [], []
    for fen_string in fens:
        w_t, b_t, s_t = fen_to_halfKP(fen_string)
        w.append(w_t)
        b.append(b_t)
        stm.append(s_t)

    return torch.stack(w), torch.stack(b), torch.stack(stm), torch.tensor(evals)
        

def HalfKP_train(num_epochs, model, lr, experiment_num, balance, save_name, collate_fn):
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=10, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=8, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=8, collate_fn=collate_fn)

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-6, weight_decay=1e-4)
    criterion = nn.MSELoss()
    train_losses, val_losses = [], []  

    for epoch in range(num_epochs):

        # Train loop
        model.train()
        running_loss = 0.0
        counter = 0
        for white_features, black_features, stms, evals in tqdm(train_loader, desc=f" Epoch {epoch+1}: training loop"):
            white_features = white_features.to(device)
            white_features = white_features.view(white_features.size(0), -1)
            black_features = black_features.to(device)
            black_features = black_features.view(white_features.size(0), -1)
            stms = stms.to(device)
            evals = evals.to(device)
            scaled_evals = torch.sigmoid(evals / 400) # Transforms from centipawn scale to win-draw-loss, helps with gradients and groups large evals 
            outputs = model(white_features, black_features, stms).squeeze(1)
            scaled_outputs = torch.sigmoid(outputs / 400) 

            # Stockfish docs has a good implementation interpolating final game result here 
            # Don't have access to game results, but still don't want the model to just instinctively output near high/low saturations
            # Use a small weighting on raw centipawn eval as an alternative
            loss_eval = criterion(scaled_outputs, scaled_evals)
            loss_raw = criterion(outputs, evals)
            loss_total = balance * loss_eval + (1-balance) * loss_raw

            running_loss += loss_total 
            counter += 1

            loss_total.backward()
            optimizer.step()
            optimizer.zero_grad()

        train_loss = running_loss / counter
        train_losses.append(train_loss)

        # Val loop
        model.eval()
        running_loss, running_acc = 0.0, 0.0
        counter = 0
        with torch.no_grad():
            for white_features, black_features, stms, evals in tqdm(val_loader, desc=f" Epoch {epoch+1}: validation loop"):
                white_features = white_features.to(device)
                white_features = white_features.view(white_features.size(0), -1)
                black_features = black_features.to(device)
                black_features = black_features.view(white_features.size(0), -1)
                stms = stms.to(device)
                evals = evals.to(device)
                scaled_evals = torch.sigmoid(evals / 400) 
                outputs = model(white_features, black_features, stms).squeeze(1)
                scaled_outputs = torch.sigmoid(outputs / 400) 
                loss_eval = criterion(scaled_outputs, scaled_evals) 
                loss_raw = criterion(outputs, evals) 
                loss_total = balance * loss_eval + (1-balance) * loss_raw
                acc = 1 - (balance * abs(scaled_outputs[0] - scaled_evals[0]) + (1-balance) * abs(outputs[0] - evals[0]))
                running_loss += loss_total
                running_acc += acc
                counter += 1
        val_loss = running_loss / counter
        total_acc = running_acc / counter
        val_losses.append(val_loss)

        print(f" -> Epoch: {epoch+1} / Train Loss: {train_loss} / Val Loss: {val_loss} / Acc: {total_acc}")
        torch.save(model.state_dict(), f"models/save_states/Exp{experiment_num}{save_name}E{epoch+1}")


if __name__ == '__main__':
    if (MODE == 'HalfKP'):
        HalfKP_train(
            num_epochs = 100,
            model = HalfKP_NNUE(), 
            lr = 1e-4,
            experiment_num = 1,
            balance = 0.8, 
            save_name = 'HalfKP',
            collate_fn = HalfKP_collate
        )


            
            


    