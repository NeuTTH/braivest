import os
import wandb
from braivest.train.Trainer import Trainer
from braivest.utils import load_data

ARTIFACT_DIR = "/scratch/gpfs/tt1131/projects/braivest/dataset/zf-qrem/"
MODEL_DIR = "/scratch/gpfs/tt1131/projects/braivest/models/zf-qrem/"

base_model_config = {
    'num_layers': 2,
    'layer_dims': 250,
    'batch_size': 256,
    'lr': 1e-4,
    'nw': 0,
    'kl': 1e-5,
    'latent': 2,
    'time': True,
    'emg': False,
    'w_label': True,
    'save_best': False,
    'epochs': 500,
    'seed': 42
}

def train_config(config=None, model_name=None):
    ### Save locally first
    run = wandb.init(config=config, mode='offline')
    # wandb.config.epochs = config['epochs']['value']
    config = wandb.config

    train_X = load_data(ARTIFACT_DIR, 'train.npy')
    print("Data loaded")
    input_dim = train_X.shape[1]
    if config.w_label:
        input_dim = input_dim - 1

    trainer = Trainer(config, input_dim)

    trainer.load_dataset(ARTIFACT_DIR)
    print("Trainer initialized")
    
    history = trainer.train(wandb=True)
    trainer.model.save_weights(os.path.join(MODEL_DIR, model_name))
    run.finish()

def main():
    for seed in [3, 42, 68, 57]:
        base_model_config['seed'] = seed
        train_config(base_model_config, f"model_last_{seed}.h5")

if __name__ == '__main__':
	main()