python -m venv virtual_env
.\virtual_env\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision numpy matplotlib jupyter ipykernel tensorboard h5py
python -m ipykernel install --user --name ml_course --display-name "ML Course"