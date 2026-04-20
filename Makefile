PYTHON ?= python
PYTHONPATH ?= src

.PHONY: install train-fashion train-cifar eval-fashion eval-cifar matrix-fashion matrix-cifar

install:
	$(PYTHON) -m pip install -r requirements.txt

train-fashion:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/train_baseline.py --dataset fashion_mnist --epochs 1 --out-checkpoint results/checkpoints/fashion_baseline.pt

train-cifar:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/train_baseline.py --dataset cifar10 --epochs 1 --out-checkpoint results/checkpoints/cifar10_baseline.pt

eval-fashion:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/run_decode_eval.py --checkpoint results/checkpoints/fashion_baseline.pt --out-json results/eval/fashion_random_b16.json --schedule random --block-size 16

eval-cifar:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/run_decode_eval.py --checkpoint results/checkpoints/cifar10_baseline.pt --out-json results/eval/cifar_random_b16.json --schedule random --block-size 16

matrix-fashion:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/run_minimum_matrix.py --dataset fashion_mnist --checkpoint results/checkpoints/fashion_baseline.pt

matrix-cifar:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/run_minimum_matrix.py --dataset cifar10 --checkpoint results/checkpoints/cifar10_baseline.pt
