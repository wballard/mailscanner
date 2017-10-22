
test: install-dev
	pytest --doctest-modules mailscanner
.PHONY: test

install:
	python setup.py install
.PHONY: install

upload:
	python setup.py sdist upload
.PHONY: upload

install-dev:
	python setup.py develop
.PHONY: install-dev
