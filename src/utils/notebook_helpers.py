from typing import List, Optional, Sequence, Tuple
import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch._C import device as TorchDevice
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Import repo modules through the canonical preprocessing path.
try:
    from src.preprocessing.image_transforms import PILToFloatTensor
    from src.preprocessing.datasets import DogsCatsDataset
except Exception:
    PILToFloatTensor = None
    DogsCatsDataset = None

try:
    from src.training.engine import evaluate, train_one_epoch
except Exception:
    evaluate = None
    train_one_epoch = None


def list_sample_paths(data_dir: str, max_items: Optional[int] = 20) -> List[Tuple[str, int]]:
    """List up to ``max_items`` (path, label) samples from ``data_dir``.

    Args:
        data_dir: Directory containing image files named like `cat.123.jpg` or `dog.45.jpg`.
        max_items: Maximum number of samples to return. Use ``None`` to return all samples.

    Returns:
        List of tuples ``(path, label)`` where ``label`` is 0 for cat and 1 for dog.
    """
    samples = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        label = 0 if 'cat' in fname.lower() else 1
        samples.append((os.path.join(data_dir, fname), label))
        if max_items is not None and len(samples) >= max_items:
            break
    return samples


def count_sample_labels(data_dir: str, max_items: Optional[int] = None) -> dict:
    """Return label counts for the sample files in ``data_dir``.

    Args:
        data_dir: Dataset directory containing labeled filenames.
        max_items: Optional limit on how many files to inspect.

    Returns:
        Dictionary with counts for 'cats' and 'dogs'.
    """
    samples = list_sample_paths(data_dir, max_items=max_items)
    counts = {'cats': 0, 'dogs': 0}
    for _, label in samples:
        if label == 0:
            counts['cats'] += 1
        else:
            counts['dogs'] += 1
    return counts


def stratified_train_val_split(sample_paths: List[Tuple[str, int]], train_ratio: float = 0.8, seed: int = 0):
    """Split paths into stratified train and validation sets by class.

    Args:
        sample_paths: List of (path, label) tuples.
        train_ratio: Fraction of samples to keep in the training set.
        seed: Random seed for reproducibility.

    Returns:
        ``(train_paths, val_paths)`` where both lists preserve class balance.
    """
    cat_paths = [p for p in sample_paths if p[1] == 0]
    dog_paths = [p for p in sample_paths if p[1] == 1]

    rng = random.Random(seed)
    rng.shuffle(cat_paths)
    rng.shuffle(dog_paths)

    n_cat_train = int(round(len(cat_paths) * train_ratio))
    n_dog_train = int(round(len(dog_paths) * train_ratio))

    train_paths = cat_paths[:n_cat_train] + dog_paths[:n_dog_train]
    val_paths = cat_paths[n_cat_train:] + dog_paths[n_dog_train:]

    rng.shuffle(train_paths)
    rng.shuffle(val_paths)
    return train_paths, val_paths


def build_train_val_dataloaders(
    sample_paths: List[Tuple[str, int]],
    grid: int = 2,
    batch_size: int = 8,
    train_ratio: float = 0.8,
    seed: int = 0,
    base_transform: Optional[transforms.Compose] = None,
):
    """Build stratified train and validation loaders from labeled sample paths."""
    train_paths, val_paths = stratified_train_val_split(sample_paths, train_ratio=train_ratio, seed=seed)
    train_dl, train_ds = build_tiny_dataloader(
        sample_paths=train_paths,
        grid=grid,
        batch_size=batch_size,
        base_transform=base_transform,
        use_synthetic=False,
        seed=seed,
    )
    val_dl, val_ds = build_tiny_dataloader(
        sample_paths=val_paths,
        grid=grid,
        batch_size=batch_size,
        base_transform=base_transform,
        use_synthetic=False,
        seed=seed,
    )
    return train_dl, val_dl, train_ds, val_ds


def load_pil_image(path: str) -> Image.Image:
    """Load an image from ``path`` and return an RGB PIL Image."""
    return Image.open(path).convert('RGB')


def create_synthetic_rgb_images(n: int, size: Tuple[int, int] = (224, 224), seed: int = 0) -> List[Image.Image]:
    """Create ``n`` reproducible synthetic RGB PIL images.

    Images are simple patterned arrays useful for visualization and quick tests.

    Args:
        n: Number of images to create.
        size: (width, height) of each image.
        seed: RNG seed for reproducibility.

    Returns:
        List of PIL Image objects.
    """
    rng = np.random.RandomState(seed)
    imgs: List[Image.Image] = []
    for i in range(n):
        arr = rng.randint(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
        # Add a simple gradient + circle so tiles look distinct
        yy, xx = np.mgrid[:size[1], :size[0]]
        arr[..., 0] = (arr[..., 0] + (xx + i * 13) % 256) % 256
        arr[..., 1] = (arr[..., 1] + (yy + i * 7) % 256) % 256
        imgs.append(Image.fromarray(arr))
    return imgs


def split_into_tiles(img: Image.Image, tiles_per_side: int) -> List[Image.Image]:
    """Split PIL ``img`` into a square tile grid and return tiles in row-major order.

    Args:
        img: PIL Image.
        tiles_per_side: Number of tiles along each axis.

    Returns:
        List of PIL Image tiles.
    """
    w, h = img.size
    tile_w = w // tiles_per_side
    tile_h = h // tiles_per_side
    tiles: List[Image.Image] = []
    for r in range(tiles_per_side):
        for c in range(tiles_per_side):
            left = c * tile_w
            upper = r * tile_h
            right = left + tile_w
            lower = upper + tile_h
            tiles.append(img.crop((left, upper, right, lower)))
    return tiles


def visualize_tiles(tiles: Sequence[Image.Image], tile_permutation: Optional[Sequence[int]] = None, cols: int = 0):
    """Return a matplotlib Figure visualizing tiles in original or reordered output order.

    Args:
        tiles: Sequence of PIL Image tiles.
        tile_permutation: Optional sequence mapping output positions to source tile indices.
        cols: Number of columns for display. If 0, set to grid width.

    Returns:
        matplotlib.figure.Figure
    """
    n = len(tiles)
    if tile_permutation is None:
        order = list(range(n))
    else:
        order = list(tile_permutation)
    G = int(np.sqrt(n))
    if cols <= 0:
        cols = G
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axs = np.array(axs).reshape(-1)
    for ax in axs:
        ax.axis('off')
    for i, idx in enumerate(order):
        ax = axs[i]
        ax.imshow(tiles[idx])
        ax.set_title(str(idx))
    return fig


class _SyntheticTileDataset(Dataset):
    """Small Dataset that returns image tensors from a list of PIL images."""

    def __init__(self, images: List[Image.Image], labels: Optional[List[int]] = None, transform: Optional[transforms.Compose] = None):
        self.images = images
        self.labels = labels if labels is not None else [0] * len(images)
        self.transform = transform or transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        return self.transform(img), int(self.labels[idx])


def build_tiny_dataloader(
    sample_paths: Optional[List[Tuple[str, int]]] = None,
    grid: int = 2,
    batch_size: int = 8,
    base_transform: Optional[transforms.Compose] = None,
    use_synthetic: bool = True,
    synthetic_count: int = 16,
    seed: int = 0,
) -> Tuple[DataLoader, Dataset]:
    """Build a small DataLoader for quick experiments.

    If ``sample_paths`` is provided and the repo's ``DogsCatsDataset`` is available,
    it will be used. Otherwise a small synthetic dataset is returned.

    Returns:
        (dataloader, dataset)
    """
    if base_transform is None:
        base_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    if sample_paths is not None and DogsCatsDataset is not None:
        dataset = DogsCatsDataset(sample_paths, transform=base_transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader, dataset

    # Fallback: synthetic dataset
    imgs = create_synthetic_rgb_images(synthetic_count, size=(224, 224), seed=seed)
    labels = [i % 2 for i in range(len(imgs))]
    dataset = _SyntheticTileDataset(imgs, labels, transform=base_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, dataset


def run_quick_train_step(model: torch.nn.Module, dataloader: DataLoader, device: Optional[TorchDevice] = None, optimizer: Optional[torch.optim.Optimizer] = None, epoch: int = 0, mixup_alpha: float = 0.0):
    """Run a single train + validate step using repo training helpers when available.

    This function will attempt to use ``train_one_epoch`` and ``evaluate`` from
    ``src.training.engine``. If they are not importable, a minimal local train
    loop will be used.

    Args:
        model: torch model.
        dataloader: training DataLoader (will be reused as validation here for quick checks).
        device: device to run on; defaults to CUDA if available else CPU.
        optimizer: optional optimizer. If None, SGD is created.
        epoch: Currently unused argument.
        mixup_alpha: Currently unused argument.

    Returns:
        Dict with keys 'train' and 'val' mapping to summary dicts.
    """
    if device is None:
        device = TorchDevice('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    if optimizer is None:
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    del epoch, mixup_alpha
    if train_one_epoch is not None and evaluate is not None:
        train_metrics = train_one_epoch(model, dataloader, optimizer, criterion, device)
        val_metrics = evaluate(model, dataloader, criterion, device)
        result = {'train': train_metrics, 'val': val_metrics}
        return result

    # Minimal fallback train loop
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for xb, yb in dataloader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += xb.size(0)
    train_metrics = {'train_loss': running_loss / max(1, total), 'train_accuracy': correct / max(1, total)}

    # Quick validation: reuse same dataloader
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            running_loss += loss.item() * xb.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
    val_metrics = {'val_loss': running_loss / max(1, total), 'val_accuracy': correct / max(1, total)}
    result = {'train': train_metrics, 'val': val_metrics}
    return result
