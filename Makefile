PYTHON ?= python
PYTHONPATH ?= src

.PHONY: install pcnnpp-train-fashion pcnnpp-train-cifar pcnnpp-eval-fashion pcnnpp-eval-cifar arpg-checkout arpg-train-fashion arpg-eval-fashion

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

# ARPG lives on origin/cifar — checkout without merging
arpg-checkout:
	git fetch origin cifar
	git checkout origin/cifar -- src/ARPG/ scripts/train_arpg.py scripts/eval_arpg.py

arpg-train-fashion:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/train_arpg.py --dataset fashion_mnist --save-dir results/arpg_fashion --epochs 20 --batch-size 16 --d-model 192 --n-heads 6 --n-layers 6

arpg-eval-fashion:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/eval_arpg.py --checkpoint results/arpg_fashion/checkpoints/arpg_fashion_mnist_d192_l6.pt --out-dir results/arpg_fashion/eval --ks 1,2,4,7,14,28,56,112,196,392,784 --schedules random,raster,row --n-samples 25
