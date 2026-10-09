.PHONY: test smoke figures lazy

test:
	python -m pytest

smoke:
	python -m resnet_limits.run --config configs/smoke.yaml

figures: smoke
	python -m resnet_limits.plot --results results/smoke --output figures/smoke.png

lazy:
	python -m resnet_limits.lazy_run --width 512 --depth 64 --steps 5 --output results/lazy
