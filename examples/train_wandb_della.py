import os
import wandb
from braivest.train.Trainer import Trainer
from braivest.utils import load_data

ARTIFACT_DIR = "/scratch/gpfs/tt1131/projects/braivest/artifacts/training_set:v0"

base_model_config = {
    'num_layers': 2,
    'layer_dims': 250,
    'batch_size': 10000,
    'lr': 1e-3,
    'nw': 0,
    'kl': 1e-5,
    'latent': 2,
    'time': False,
    'emg': True,
    'save_best': False,
    'epochs': 500
}

# sweep_config = {
#     'method': 'random',
#     'metric': {
#         'goal': 'minimize',
#         'name': 'val_loss'
#     },
#     'parameters':{
#         'num_layers':{'value': 2},
#         'layer_dims':{'value': 250},
#         'batch_size': {'values':  [1000, 4000, 10000]},
#         'lr': {'max': 0.1, 'min': 0.0001},
#         'nw': {'value': 0},
#         'kl': {'values': [0, 1e-4, 1e-2, 1]},
#         'latent': {'value': 2},
#         'time':{'value': False},
#         'emg': {'value': True},
#         'save_best':{'value': False}
#     },
#     'epochs': {'value': 1000},
# }

def train_config(config=None):
    ### Save locally first
    run = wandb.init(config=config, mode='offline')
    # wandb.config.epochs = config['epochs']['value']
    config = wandb.config

    train_X = load_data(ARTIFACT_DIR, 'train.npy')
    print("Data loaded")
    input_dim = train_X.shape[1]
    trainer= Trainer(config, input_dim)

    trainer.load_dataset(ARTIFACT_DIR)
    print("Trainer initialized")
    
    history = trainer.train(wandb=True)
    trainer.model.save_weights(os.path.join(wandb.run.dir, "model.h5"))
    run.finish()

def main():
    # Sweep is not supported in offline mode
    # sweep_id = wandb.sweep(sweep_config) 
    # wandb.agent(sweep_id, train_config, count=10, project='braivest_test')

    for seed_val in [1, 10, 100]:
        base_model_config['seed'] = seed_val
        for i in range(2):
            train_config(base_model_config)

if __name__ == '__main__':
	main()