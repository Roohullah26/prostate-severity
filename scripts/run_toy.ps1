# windows powershell helper: runs a tiny toy training + eval + infer session
python -m src.train --toy --epochs 1 --bs 4
python -m src.eval --toy --model-path models/prototype_toy.pth
python -m src.infer --toy --model-path models/prototype_toy.pth
