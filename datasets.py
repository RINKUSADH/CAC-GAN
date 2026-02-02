import os
from torch.utils.data import Dataset
from PIL import Image

class MRIImageDataset(Dataset):
    def __init__(self, t1_images_dir, t2_images_dir, input_modality, target_modality, transform=None):
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

        # Extract identifiers without _T1 or _T2 suffix
        t1_files = {
            f.replace("_" + input_modality.upper(), "").rsplit(".", 1)[0]: os.path.join(t1_images_dir, f)
            for f in os.listdir(t1_images_dir) if f.lower().endswith(valid_extensions)
        }
        t2_files = {
            f.replace("_" + target_modality.upper(), "").rsplit(".", 1)[0]: os.path.join(t2_images_dir, f)
            for f in os.listdir(t2_images_dir) if f.lower().endswith(valid_extensions)
        }

        # Debugging: Print some identifiers to verify matching
        print(f"{input_modality.upper()} Identifiers Sample:", sorted(t1_files.keys())[:10])
        print(f"{target_modality.upper()} Identifiers Sample:", sorted(t2_files.keys())[:10])

        # Find common identifiers
        common_ids = sorted(set(t1_files.keys()) & set(t2_files.keys()))
        print("Matching Identifiers Sample:", common_ids[:10])

        if not common_ids:
            raise ValueError(f"No matching {input_modality.upper()}-{target_modality.upper()} image pairs found! Check filenames.")

        # Store only matched T1-T2 pairs
        self.t1_image_file_names = [t1_files[id] for id in common_ids]
        self.t2_image_file_names = [t2_files[id] for id in common_ids]

        self.transform = transform

    def __len__(self):
        return len(self.t1_image_file_names)

    def __getitem__(self, idx):
        try:
            t1_path = self.t1_image_file_names[idx]
            t2_path = self.t2_image_file_names[idx]

            t1_image = Image.open(t1_path).convert('L')
            t2_image = Image.open(t2_path).convert('L')

            if self.transform:
                t1_image = self.transform(t1_image)
                t2_image = self.transform(t2_image)

            return {self.input_modality: t1_image, self.target_modality: t2_image}

        except (IOError, OSError) as e:
            print(f"Skipping corrupted image: {t1_path} or {t2_path} - {e}")
            return None  # Instead of forcing another sample, we skip it
