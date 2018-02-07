dist:
	make test && make build && make upload
build:
	python3 setup.py bdist_wheel --universal

test:
	python3 -m pytest -s --ignore tests/test_benchmark.py tests/

benchmark:
	python3 -m pytest -s tests/test_benchmark.py

upload:
	twine upload dist/*
