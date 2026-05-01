from typing import Dict, Any, List
import os
import yaml
import random
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.loaders.dataset import TilePermutationDataset
from src.models.model import get_model
from src.utils.permutations import generate_permutations, identity_permutation
from src.training.train import train_one_epoch, validate, save_checkpoint


class ExperimentRunner:
    """Runs experiments based on a simple YAML config.

    The runner creates datasets for requested grid sizes and permutations,
    trains models using provided hyperparameters, and writes CSV summaries.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _load_samples(self, data_dir: str):
        # Expect data_dir contains class-labeled files named cat.1234.jpg or dog.5678.jpg
        samples = []
        for fname in os.listdir(data_dir):
            if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
            label = 0 if 'cat' in fname.lower() else 1
            samples.append((os.path.join(data_dir, fname), label))
        return samples

    def run(self):
        out_dir = self.config.get('output_dir', 'results')
        os.makedirs(out_dir, exist_ok=True)
        data_dir = self.config['data_dir']
        samples = self._load_samples(data_dir)

        rows = []
        models = self.config['models']
        grids = self.config['grids']
        n_perms = self.config.get('n_permutations', 5)
        seeds = self.config.get('seeds', [0])

        for model_name in models:
            for G in grids:
                perms = [identity_permutation(G)] + generate_permutations(G, n_perms - 1, seed=42)
                for perm_idx, perm in enumerate(perms):
                    for seed in seeds:
                        run_id = f"{model_name}_G{G}_p{perm_idx}_s{seed}"
                        print('Running', run_id)
                        # Build dataset and loader with optional stronger augmentations
                        use_improved = bool(self.config.get('improved', False))
                        # Default transforms
                        from torchvision import transforms
                        if use_improved:
                            base_transform = transforms.Compose([
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                                transforms.ToTensor(),
                            ])
                        else:
                            base_transform = transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                            ])

                        ds = TilePermutationDataset(samples, grid_size=G, permutation=perm, seed=seed, base_transform=base_transform)
                        # Simple split: small val set
                        n = len(ds)
                        nval = min(500, int(0.1 * n))
                        train_ds, val_ds = torch.utils.data.random_split(ds, [n - nval, nval], generator=torch.Generator().manual_seed(seed))
                        train_loader = DataLoader(train_ds, batch_size=self.config.get('batch_size', 32), shuffle=True, num_workers=2)
                        val_loader = DataLoader(val_ds, batch_size=self.config.get('batch_size', 32), shuffle=False, num_workers=2)

                        model = get_model(model_name, num_classes=2, pretrained=self.config.get('pretrained', True))
                        model = model.to(self.device)
                        optimizer = torch.optim.SGD(model.parameters(), lr=self.config.get('lr', 1e-3), momentum=0.9)
                        criterion = torch.nn.CrossEntropyLoss()

                        mixup_alpha = float(self.config.get('mixup_alpha', 0.0))
                        epochs = self.config.get('epochs', 3)
                        for epoch in range(epochs):
                            tr = train_one_epoch(model, train_loader, optimizer, self.device, criterion, epoch, mixup_alpha=mixup_alpha)
                            va = validate(model, val_loader, self.device, criterion)
                            print(f"{run_id} epoch {epoch}: tr_acc={tr['acc']:.3f} val_acc={va['acc']:.3f}")

                        # Save checkpoint and row summary
                        ckpt_path = os.path.join(out_dir, f"{run_id}.pt")
                        save_checkpoint({'model_state': model.state_dict(), 'config': self.config}, ckpt_path)
                        rows.append({
                            'run_id': run_id,
                            'model': model_name,
                            'grid': G,
                            'perm_idx': perm_idx,
                            'seed': seed,
                            'val_acc': va['acc'],
                        })

        df = pd.DataFrame(rows)
        summary_path = os.path.join(out_dir, 'summary.csv')
        df.to_csv(summary_path, index=False)
        print('Wrote', summary_path)


def load_config(path: str):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main(config_path: str):
    cfg = load_config(config_path)
    runner = ExperimentRunner(cfg)
    runner.run()


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, required=True)
    args = p.parse_args()
    main(args.config)
