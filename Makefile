dist:
	make test && make build && make upload
build:
	python setup.py bdist_wheel --universal

test:
	py.test -s tests/

upload:
	twine upload dist/*
