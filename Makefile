install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

lint:
	pylint --disable=R,C,pointless-statement,undefined-variable,f-string-without-interpolation,unused-variable --extension-pkg-whitelist='pydantic' ./10_code/*.ipynb ./10_code/*.py

format:
	black ./10_code/*.ipynb ./10_code/*.py

all: install lint format
