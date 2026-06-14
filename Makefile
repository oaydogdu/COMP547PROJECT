PYTHON ?= python
PYTHONPATH ?= src

.PHONY: install pcnnpp-train-fashion pcnnpp-eval-fashion arpg-train-fashion arpg-eval-fashion fashion-report fid-baseline fid-arpg

BASELINE_CKPT ?= results/pixelcnnpp_fashion_e20/checkpoints/best.pt
ARPG_CKPT ?= results/arpg_fashion/checkpoints/best.pt

install:
	$(PYTHON) -m pip install -r requirements.txt

pcnnpp-train-fashion:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/train_pixelcnnpp.py --dataset fashion_mnist --epochs 20 --batch-size 16 --num-workers 0 --save-dir results/pixelcnnpp_fashion_e20

pcnnpp-eval-fashion:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/eval_pixelcnnpp.py --checkpoint results/pixelcnnpp_fashion_e20/checkpoints/best.pt --out-json results/pixelcnnpp_fashion_e20/eval/fashion_eval.json --out-grid results/pixelcnnpp_fashion_e20/eval/fashion_grid.png

arpg-train-fashion:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/train_arpg.py --dataset fashion_mnist --save-dir results/arpg_fashion --epochs 20 --batch-size 16 --d-model 192 --n-heads 6 --n-layers 6 --num-workers 0

arpg-eval-fashion:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/eval_arpg.py --checkpoint results/arpg_fashion/checkpoints/best.pt --out-dir results/arpg_fashion/eval --ks 1,2,4,7,14,28,56,112,196,392,784 --schedules random,raster,row --n-samples 25

fashion-report:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/build_fashion_report.py --baseline-eval-json results/pixelcnnpp_fashion_e20/eval/fashion_eval.json --arpg-sweep-json results/arpg_fashion/eval/sweep.json --baseline-metrics-json results/pixelcnnpp_fashion_e20/metrics/pixelcnnpp_fashion_mnist_lr0.00020_res5_f160.json --out-dir results/fashion_presentation

fid-baseline:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/compute_fid_fashion.py --model baseline --checkpoint $(BASELINE_CKPT) --out-dir results/fid/baseline --n-samples 2048 --compute-fid --out-json results/fid/baseline_fid.json

fid-arpg:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/compute_fid_fashion.py --model arpg --checkpoint $(ARPG_CKPT) --out-dir results/fid/arpg_random_K28 --k 28 --schedule random --n-samples 2048 --compute-fid --out-json results/fid/arpg_random_K28_fid.json
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/compute_fid_fashion.py --model arpg --checkpoint $(ARPG_CKPT) --out-dir results/fid/arpg_random_K784 --k 784 --schedule random --n-samples 2048 --compute-fid --out-json results/fid/arpg_random_K784_fid.json
