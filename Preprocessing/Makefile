wget https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets/protobowl-042818.log	mv protobowl-042818.log protobowl.log

 	wget https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets/qanta.train.2018.04.18.json
	mv qanta.train.2018.04.18.json qanta.train.json
	wget https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets/qanta.dev.2018.04.18.json
	mv qanta.dev.2018.04.18.json qanta.dev.json
	wget https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets/qanta.test.2018.04.18.json
	mv qanta.test.2018.04.18.json qanta.test.json	
python convertQanta.py
	python guessToQanta.py
	python consolidate.py
	python normalize.py