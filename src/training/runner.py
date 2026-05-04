from typing import Dict, Any, List
import os
import yaml
import random
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.models.model import get_model
from src.preprocessing.legacy_dataset import TilePermutationDataset
from src.preprocessing.permutations import generate_permutations, identity_permutation
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
        """Load legacy sample tuples from a labeled image directory.

        Args:
            data_dir: Directory containing files named like ``cat.1234.jpg`` or ``dog.5678.jpg``.

        Returns:
            List of ``(path, label)`` tuples.
        """

        samples = []
        for filename in os.listdir(data_dir):
            if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
            label = 0 if 'cat' in filename.lower() else 1
            samples.append((os.path.join(data_dir, filename), label))
        return samples

    def run(self):
        """Run the legacy experiment loop and write a summary CSV."""

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
            for grid_size in grids:
                permutations = [identity_permutation(grid_size)] + generate_permutations(grid_size, n_perms - 1, seed=42)
                for permutation_index, permutation in enumerate(permutations):
                    for seed in seeds:
                        run_id = f"{model_name}_G{grid_size}_p{permutation_index}_s{seed}"
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

                        dataset = TilePermutationDataset(
                            samples,
                            grid_size=grid_size,
                            permutation=permutation,
                            seed=seed,
                            base_transform=base_transform,
                        )
                        # Simple split: small val set
                        sample_count = len(dataset)
                        val_count = min(500, int(0.1 * sample_count))
                        train_dataset, val_dataset = torch.utils.data.random_split(
                            dataset,
                            [sample_count - val_count, val_count],
                            generator=torch.Generator().manual_seed(seed),
                        )
                        train_loader = DataLoader(train_dataset, batch_size=self.config.get('batch_size', 32), shuffle=True, num_workers=2)
                        val_loader = DataLoader(val_dataset, batch_size=self.config.get('batch_size', 32), shuffle=False, num_workers=2)

                        model = get_model(model_name, num_classes=2, pretrained=self.config.get('pretrained', True))
                        model = model.to(self.device)
                        optimizer = torch.optim.SGD(model.parameters(), lr=self.config.get('lr', 1e-3), momentum=0.9)
                        criterion = torch.nn.CrossEntropyLoss()

                        mixup_alpha = float(self.config.get('mixup_alpha', 0.0))
                        epochs = self.config.get('epochs', 3)
                        for epoch in range(epochs):
                            train_metrics = train_one_epoch(model, train_loader, optimizer, self.device, criterion, epoch, mixup_alpha=mixup_alpha)
                            val_metrics = validate(model, val_loader, self.device, criterion)
                            print(f"{run_id} epoch {epoch}: tr_acc={train_metrics['acc']:.3f} val_acc={val_metrics['acc']:.3f}")

                        # Save checkpoint and row summary
                        ckpt_path = os.path.join(out_dir, f"{run_id}.pt")
                        save_checkpoint({'model_state': model.state_dict(), 'config': self.config}, ckpt_path)
                        rows.append({
                            'run_id': run_id,
                            'model': model_name,
                            'grid': grid_size,
                            'perm_idx': permutation_index,
                            'seed': seed,
                            'val_acc': val_metrics['acc'],
                        })

        df = pd.DataFrame(rows)
        summary_path = os.path.join(out_dir, 'summary.csv')
        df.to_csv(summary_path, index=False)
        print('Wrote', summary_path)


def load_config(path: str):
    """Load a YAML config for the legacy runner.

    Args:
        path: YAML config path.

    Returns:
        Parsed configuration dictionary.
    """

    with open(path, 'r') as handle:
        config = yaml.safe_load(handle)
    return config


def main(config_path: str):
    """Run the legacy experiment runner from a config path.

    Args:
        config_path: YAML config path.
    """

    cfg = load_config(config_path)
    runner = ExperimentRunner(cfg)
    runner.run()


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, required=True)
    args = p.parse_args()
    main(args.config)
