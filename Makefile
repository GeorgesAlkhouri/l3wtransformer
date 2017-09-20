dist:
	make test && make build && make upload
build:
	python setup.py bdist_wheel --universal

test:
	py.test -s --ignore tests/test_benchmark.py tests/

benchmark:
	py.test -s tests/test_benchmark.py

upload:
	twine upload dist/*
