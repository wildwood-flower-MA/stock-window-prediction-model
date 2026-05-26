import os
import io
import zipfile
from typing import List
import numpy as np
import h5py
import torch

current_dir = os.path.dirname(__file__)
path = os.path.abspath(os.path.join(current_dir, "..", "BenchmarkDatasets.zip"))

def get_txt_paths() -> List[str]:
    with zipfile.ZipFile(path, 'r') as archive:
        all_files = archive.namelist()

        txt_files = [
            file for file in all_files if file.endswith('.txt')
        ]
                
    return txt_files

def read_zip_file(file_path: str) -> str:
    with zipfile.ZipFile(path, 'r') as archive:
        with archive.open(file_path) as f:
            return f.read().decode('utf-8', errors='ignore')
        
def build_hdf5_database(zip_paths: List[str], output_h5_path: str):
    
    num_features = 149 # specyfika datasetu

    with h5py.File(output_h5_path, 'w') as h5f:
        
        dataset = h5f.create_dataset(
            'limit_order_book',
            shape=(0, num_features),
            maxshape=(None, num_features),
            chunks=True,
            compression='gzip'
        )

        current_rows = 0

        for i, internal_path in enumerate(zip_paths):
            
            total_files = len(zip_paths)
            current_file_idx = i + 1
            
            print(f"Teraz: {current_file_idx}/{total_files}: {internal_path}")

            raw_text = read_zip_file(internal_path)
            data_stream = io.StringIO(raw_text)
            data = np.genfromtxt(data_stream)
            
            # wiersz: krok czasowy; kolumna: cechy rynku
            data = data.T 

            num_new_rows = data.shape[0]
            new_total_rows = current_rows + num_new_rows

            dataset.resize((new_total_rows, num_features))
            dataset[current_rows:new_total_rows, :] = data
            current_rows = new_total_rows

class LOBDataset(torch.utils.data.Dataset):
    def __init__(self, file_path: str, sequence_length: int, horizon_idx: int = 0):
        self.file_path = file_path
        self.sequence_length = sequence_length
        self.horizon_idx = horizon_idx
        self.dataset = None
        
        with h5py.File(self.file_path, 'r') as f:
            self.length = len(f['limit_order_book']) - self.sequence_length  # type: ignore
            
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if self.dataset is None:
            self.file = h5py.File(self.file_path, 'r')
            self.dataset = self.file['limit_order_book']
            
        window = self.dataset[idx : idx + self.sequence_length] # type: ignore
        
        x = window[:, :144] # type: ignore
        y_raw = window[-1, 144 + self.horizon_idx] # type: ignore
        y = int(y_raw) - 1  # type: ignore
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)