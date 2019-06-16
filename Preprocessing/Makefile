protobowl.log:
	wget https://www.google.com/url?q=https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fpinafore-us-west-2%2Fqanta-jmlr-datasets%2Fprotobowl-042818.log&sa=D&sntz=1&usg=AFQjCNG5M0eFv-1auQf4Oop-HC-BM7S4bQ
	mv protobowl-042818.log protobowl.log

	python convertQanta.py
	python guessToQanta.py
	python consolidate.py
	python normalize.py