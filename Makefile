

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
	@echo "run ./bin/download-gmail var/data/gmail.db <your_email_address>"

var/data/replies.weights: var/data/gmail.db
	./bin/prepare-replies-dataset var/data/replies.txt var/data/replies.weights var/data/replies.pickle

server:
	gunicorn -w $(WORKERS) -b 0.0.0.0:$(PORT) --timeout 400 mailscanner.server.server:application
.PHONY: server