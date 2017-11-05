

WORKERS?=1
PORT?=5000

test:
	pytest --doctest-modules mailscanner
.PHONY: test

install:
	conda install --file conda-requirements.txt
	pip install --requirement requirements.txt
.PHONY: install


var/data/gmail.db:
	python download-gmail.py var/data/gmail.db
.PHONY: var/data/gmail.db

var/data/replies.txt: var/data/gmail.db
	python prepare-replies-dataset.py var/data/gmail.db var/data/replies.txt

var/data/replies.weights: var/data/replies.txt
	python prepare-replies-model.py var/data/replies.txt var/data/replies.weights var/data/replies.pickle

server:
	gunicorn -w $(WORKERS) -b 0.0.0.0:$(PORT) --timeout 400 mailscanner.server.server:application
.PHONY: server