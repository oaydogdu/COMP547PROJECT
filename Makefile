PYTHON ?= python
PYTHONPATH ?= src

.PHONY: install pcnnpp-train-fashion pcnnpp-train-cifar pcnnpp-eval-fashion pcnnpp-eval-cifar

install:
	$(PYTHON) -m pip install -r requirements.txt

pcnnpp-train-fashion:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/train_pixelcnnpp.py --dataset fashion_mnist --epochs 30 --save-dir results/pixelcnnpp

pcnnpp-train-cifar:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/train_pixelcnnpp.py --dataset cifar10 --epochs 50 --save-dir results/pixelcnnpp

pcnnpp-eval-fashion:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/eval_pixelcnnpp.py --checkpoint results/pixelcnnpp/checkpoints/pixelcnnpp_fashion_mnist_lr0.00020_res5_f160.pt --out-json results/pixelcnnpp/eval/fashion_eval.json --out-grid results/pixelcnnpp/eval/fashion_grid.png

pcnnpp-eval-cifar:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/eval_pixelcnnpp.py --checkpoint results/pixelcnnpp/checkpoints/pixelcnnpp_cifar10_lr0.00020_res5_f160.pt --out-json results/pixelcnnpp/eval/cifar_eval.json --out-grid results/pixelcnnpp/eval/cifar_grid.png
