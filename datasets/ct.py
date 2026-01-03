import os
from PIL import Image
from torch.utils.data import Dataset


class CTDataset(Dataset):
    """Custom CT Image Dataset.

    Loads all JPG images from a directory for training/testing DDIM model.

    Args:
        root (str): Root directory where CT images are stored.
            The directory should contain JPG format images directly.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        extensions (tuple): Valid image file extensions. Default is ('.jpg', '.jpeg').
    """

    def __init__(self, root, transform=None, extensions=('.jpg', '.jpeg')):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.extensions = extensions

        # Get all image files from the root directory
        self.image_files = self._find_images()

        if len(self.image_files) == 0:
            raise RuntimeError(
                f"Found 0 images in '{self.root}'. "
                f"Supported extensions are: {self.extensions}"
            )

    def _find_images(self):
        """Find all image files with valid extensions in the root directory."""
        if not os.path.isdir(self.root):
            raise RuntimeError(f"Directory '{self.root}' does not exist.")

        images = []
        for filename in sorted(os.listdir(self.root)):
            if filename.lower().endswith(self.extensions):
                images.append(os.path.join(self.root, filename))
        return images

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dummy label (0).
                   CT images are unconditional, so we return 0 as placeholder.
        """
        img_path = self.image_files[index]
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        # Return dummy target (0) for unconditional generation
        return img, 0

    def __len__(self):
        return len(self.image_files)

    def __repr__(self):
        return (
            f"CTDataset(root={self.root}, "
            f"num_images={len(self)}, "
            f"extensions={self.extensions})"
        )
