"""Simple script to start an experiment using configs/experiments.yaml."""
import sys
from src.training.runner import main


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python scripts/train_experiment.py <config.yaml>')
        sys.exit(1)
    main(sys.argv[1])
