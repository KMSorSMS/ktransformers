flake_find:
	cd ktransformers && flake8 | grep -Eo '[A-Z][0-9]{3}' | sort | uniq| paste -sd ',' - 
format:
	@cd ktransformers && black .
	@black setup.py