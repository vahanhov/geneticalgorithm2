

pypipush:
	python setup.py develop
	python setup.py sdist
	python setup.py bdist_wheel
	twine upload dist/* --skip-existing
