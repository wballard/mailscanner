

WORKERS?=1
PORT?=5000

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


var/data/replies.weights:
	./bin/prepare-replies-dataset var/data/replies.txt var/data/replies.weights var/data/replies.pickle

server:
	gunicorn -w $(WORKERS) -b 0.0.0.0:$(PORT) --timeout 400 mailscanner.server.server:application
.PHONY: server