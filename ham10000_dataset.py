"""Windows-importable Dataset used by the VS Code verification notebook."""

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

CLASS_TO_IDX = {
    "akiec": 0,
    "bcc": 1,
    "bkl": 2,
    "df": 3,
    "mel": 4,
    "nv": 5,
    "vasc": 6,
}


class HAM10000Dataset(Dataset):
    """Same image-path and label logic as cell 40 of the Colab notebook."""

    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image = Image.open(
            self.image_dir / f"{row['image_id']}.jpg"
        ).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, CLASS_TO_IDX[row["dx"]]
