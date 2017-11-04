
test: install-dev
	pytest --doctest-modules mailscanner
.PHONY: test

install:
	conda install --file conda-requirements.txt
	pip install --requirement requirements.txt
.PHONY: install

upload:
	python setup.py sdist upload
.PHONY: upload

