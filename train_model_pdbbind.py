import argparse
import datetime
import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import time
import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lba.datasets import LMDBDataset
from scipy.stats import spearmanr

from model import CNN3D_LBA
from data import CNN3D_TransformLBA
import resnet
import wandb

# Construct model
def conv_model(in_channels, spatial_size, args_dict):
    num_conv = args_dict['num_conv']
    conv_filters = [32 * (2**n) for n in range(num_conv)]
    conv_kernel_size = 3
    max_pool_positions = [0, 1]*int((num_conv+1)/2)
    max_pool_sizes = [2]*num_conv
    max_pool_strides = [2]*num_conv
    fc_units = [512]

    model = CNN3D_LBA(
        in_channels, spatial_size,
        args_dict['conv_drop_rate'],
        args_dict['fc_drop_rate'],
        conv_filters, conv_kernel_size,
        max_pool_positions,
        max_pool_sizes, max_pool_strides,
        fc_units,
        batch_norm=args_dict['batch_norm'],
        dropout=not args_dict['no_dropout'])
    return model


def train_loop(pre_model, model, loader, optimizer, device):
    model.train()

    losses = []
    epoch_loss = 0
    progress_format = 'train loss: {:6.6f}'
    with tqdm.tqdm(total=len(loader), desc=progress_format.format(0)) as t:
        for i, data in enumerate(loader):
            #print(data['id'])
            feature = data['feature'].to(device).to(torch.float32)
            new_feature = pre_model(feature)
            #print(new_feature.shape)
            label = data['label'].to(device).to(torch.float32)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            output = model(new_feature)
            batch_losses = F.mse_loss(output, label, reduction='none')
            batch_losses_mean = batch_losses.mean()
            batch_losses_mean.backward()
            optimizer.step()
            # stats
            epoch_loss += (batch_losses_mean.item() - epoch_loss) / float(i + 1)
            losses.extend(batch_losses.tolist())
            t.set_description(progress_format.format(np.sqrt(epoch_loss)))
            t.update(1)

    return np.sqrt(np.mean(losses))


@torch.no_grad()
def test(pre_model, model, loader, device):
    model.eval()

    losses = []

    ids = []
    y_true = []
    y_pred = []

    for data in loader:
        #print(data['id'])
        feature = data['feature'].to(device).to(torch.float32)
        new_feature = pre_model(feature)
        label = data['label'].to(device).to(torch.float32)
        output = model(new_feature)
        batch_losses = F.mse_loss(output, label, reduction='none')
        losses.extend(batch_losses.tolist())
        ids.extend(data['id'])
        y_true.extend(label.tolist())
        y_pred.extend(output.tolist())

    results_df = pd.DataFrame(
        np.array([ids, y_true, y_pred]).T,
        columns=['structure', 'true', 'pred'],
        )
    r_p = np.corrcoef(y_true, y_pred)[0,1]
    r_s = spearmanr(y_true, y_pred)[0]

    return np.sqrt(np.mean(losses)), r_p, r_s, results_df


def save_weights(model, weight_dir):
    torch.save(model.state_dict(), weight_dir)

def train(args_dict, device, rep_name, test_mode=False):
    print(f"Training model for {rep_name} with config:")
    print(str(json.dumps(args_dict, indent=4)) + "\n")

    # Initialize a WandB run with a unique name for each repetition
    wandb.init(project="3D_CNN_LBA_", name=rep_name, config=args_dict, reinit=True)

    # Save config
    with open(os.path.join(args_dict['output_dir'], 'config.json'), 'w') as f:
        json.dump(args_dict, f, indent=4)

    np.random.seed(args_dict['random_seed'])
    torch.manual_seed(args_dict['random_seed'])

    train_dataset = LMDBDataset(os.path.join(args_dict['data_dir'], 'train'),
                                transform=CNN3D_TransformLBA(random_seed=args_dict['random_seed']))
    val_dataset = LMDBDataset(os.path.join(args_dict['data_dir'], 'val'),
                              transform=CNN3D_TransformLBA(random_seed=args_dict['random_seed']))
    test_dataset = LMDBDataset(os.path.join(args_dict['data_dir'], 'test'),
                               transform=CNN3D_TransformLBA(random_seed=args_dict['random_seed']))

    train_loader = DataLoader(train_dataset, args_dict['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, args_dict['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, args_dict['batch_size'], shuffle=False)

    pre_model = resnet.generate_model(18).to(device)
    pre_model.load_state_dict(torch.load('pth/model_stage1_epoch20.pth'), strict=False)

    in_channels = 32
    spatial_size = 23

    model = conv_model(in_channels, spatial_size, args_dict)
    print(model)
    model.to(device)

    best_val_loss = np.Inf
    best_rp = 0
    best_rs = 0
    best_epoch = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=args_dict['learning_rate'])

    # CSV file for recording each epoch
    csv_path = os.path.join(args_dict['output_dir'], "training_results.csv")
    with open(csv_path, 'w') as f:
        f.write("Epoch,Train RMSE,Train Pearson R,Train Spearman R,Val RMSE,Val Pearson R,Val Spearman R\n")

    # CSV file for recording best results
    best_csv_path = os.path.join(args_dict['output_dir'], "best_results.csv")
    with open(best_csv_path, 'w') as f:
        f.write("Best Epoch,Best Val RMSE,Best Pearson R,Best Spearman R\n")

    for epoch in range(1, args_dict['num_epochs'] + 1):
        start = time.time()
        train_loss = train_loop(pre_model, model, train_loader, optimizer, device)
        train_loss, train_r_p, train_r_s, _ = test(pre_model, model, train_loader, device)
        val_loss, val_r_p, val_r_s, val_df = test(pre_model, model, val_loader, device)
        
        # Save the best model and results
        if val_loss < best_val_loss:
            print(f"\nSave model at epoch {epoch:03d}, val_loss: {val_loss:.4f}")
            save_weights(model, os.path.join(args_dict['output_dir'], 'best_weights.pt'))
            best_val_loss = val_loss
            best_rp = val_r_p
            best_rs = val_r_s
            best_epoch = epoch
            
            # Write best results to CSV
            with open(best_csv_path, 'w') as f:
                f.write(f"{best_epoch},{best_val_loss:.7f},{best_rp:.7f},{best_rs:.7f}\n")

        elapsed = (time.time() - start)
        print('Epoch {:03d} finished in : {:.3f} s'.format(epoch, elapsed))
        print('\tTrain RMSE: {:.7f}, Val RMSE: {:.7f}, Pearson R: {:.7f}, Spearman R: {:.7f}'.format(
            train_loss, val_loss, val_r_p, val_r_s))

        # Append results to CSV for each epoch
        with open(csv_path, 'a') as f:
            f.write(f"{epoch},{train_loss:.7f},{train_r_p:.7f},{train_r_s:.7f},{val_loss:.7f},{val_r_p:.7f},{val_r_s:.7f}\n")

        # Log results to WandB
        wandb.log({
            "Epoch": epoch,
            "Train RMSE": train_loss,
            "Train Pearson R": train_r_p,
            "Train Spearman R": train_r_s,
            "Val RMSE": val_loss,
            "Val Pearson R": val_r_p,
            "Val Spearman R": val_r_s,
            "Time": elapsed
        })

    if test_mode:
        model.load_state_dict(torch.load(os.path.join(args_dict['output_dir'], 'best_weights.pt')))
        rmse, pearson, spearman, test_df = test(pre_model, model, test_loader, device)
        test_df.to_pickle(os.path.join(args_dict['output_dir'], 'test_results.pkl'))
        print('Test RMSE: {:.7f}, Pearson R: {:.7f}, Spearman R: {:.7f}'.format(
            rmse, pearson, spearman))
        test_file = os.path.join(args_dict['output_dir'], 'test_results.txt')
        with open(test_file, 'a+') as out:
            out.write('{}\t{:.7f}\t{:.7f}\t{:.7f}\n'.format(
                args_dict['random_seed'], rmse, pearson, spearman))

    return best_val_loss, best_rp, best_rs 

# Function to run multiple repetitions
def run_multiple_reps(args_dict, device, num_reps=3):
    best_results = []

    for rep in range(1, num_reps + 1):
        print(f"Running repetition {rep}/{num_reps}")
        
        # Set unique output directory for each repetition
        args_dict['output_dir'] = os.path.join(args_dict['output_dir'], f'rep_{rep}')
        os.makedirs(args_dict['output_dir'], exist_ok=True)

        # Update random seed for each repetition
        args_dict['random_seed'] = int(np.random.randint(1, 10e6))

        # Generate a name for the repetition in WandB
        rep_name = f"rep{rep}"

        # Call the train function and store best results
        best_val_loss, best_rp, best_rs = train(args_dict, device, rep_name, test_mode=False)
        best_results.append({
            "Repetition": rep,
            "Best Val Loss": best_val_loss,
            "Best Pearson R": best_rp,
            "Best Spearman R": best_rs
        })

    # Save all best results for each repetition to a CSV
    results_df = pd.DataFrame(best_results)
    results_df.to_csv('best_results_summary.csv', index=False)
    print("Completed all repetitions. Best results saved to 'best_results_summary.csv'.")

def main():
    # Example parameter set
    args_dict = {
        'data_dir': 'pdbbind_output_mdb/split/data',
        'mode': 'test',
        'output_dir': 'output_RLaffinity',
        'unobserved': False,
        'learning_rate': 0.0001,
        'conv_drop_rate': 0.1,
        'fc_drop_rate': 0.25,
        'num_epochs': 300,  # Example shorter number of epochs for testing
        'num_conv': 4,
        'batch_norm': False,
        'no_dropout': False,
        'batch_size': 16,
        'random_seed': int(np.random.randint(1, 10e6))  # Will update per repetition
    }

    # Set device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Run the training function with three repetitions
    run_multiple_reps(args_dict, device, num_reps=3)

# Run main
if __name__ == "__main__":
    main()