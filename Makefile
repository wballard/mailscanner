
all: install test


test:
	pytest --doctest-modules mailscanner
.PHONY: test


install:
	pip install -r requirements.txt
	python setup.py install
.PHONY: install

upload: README.rst
	python setup.py sdist upload

upload-to-pypitest: README.rst
	python setup.py sdist upload -r pypitest
.PHONY: upload-to-pypitest

install-from-pypitest::
	pip install -U --no-cache-dir -i https://testpypi.python.org/pypi mailscanner
.PHONY: install-from-pypitest

install-dev:
	pip install -r requirements.txt
	python setup.py develop
.PHONY: install-dev
