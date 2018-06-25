
**1. Download dataset:**

1. Go to https://nlp.stanford.edu/sentiment/code.html
2. Go to "Dataset Downloads" on the right size of the page
3. Click "Train, Dev, Test, Splits in PTB Tree Format" to download the data
	1. unzip the downloaded zip file, you will see a "trees" folder
	2. put the "trees" folder under the data/large_files/stanford_sentiment folder  in this project

Note: the data have already been included in this project. 

**2. Project folder structure**
* sentiment_analysis
	* data
		* large_files
			* stanford_sentiment
				* parsed_data
				* trees

Project report,  proposal, and all .ipybn files are under the "sentiment_analysis" folder. The data downloaded from Stanford web site should be put under "stanford_sentiment" folder.

After you run stanford_recursive_network_sentiment_analysis_raw_data_parsing.ipynb, the parsed data should be put in the "parsed_data" folder.

**3. Libraries used:**

* theano
* numpy
* matplotlib
* nltk
* sklearn
* tensorflow
* keras
* json

**4. Recommened Steps**

In this project, we have 5 ipython notebooks:

1. stanford_recursive_network_sentiment_analysis_raw_data_parsing.ipynb
2. benchmark_model.ipynb
3. stanford_sentiment_analysis_for_RNN_keras.ipynb
4. stanford_sentiment_analysis_for_RNN_tensorflow.ipynb
5. recursive_neural_network_for_sentiment_analysis_coding .ipynb

You may review these notebooks following the order described above. 
* Step 1: Parse the raw data using stanford_recursive_network_sentiment_analysis_raw_data_parsing.ipynb
* Step 2: train the benchmark model using benchmark_model.ipynb
* Step 3: train the RNN mode using stanford_sentiment_analysis_for_RNN_keras.ipynb
* Step 4: (Optional) train the RNN mode using stanford_sentiment_analysis_for_RNN_tensorflow.ipynb. 
	* This step is optional since we implemented exactly the same model by using Keras and Tensorflow respectively (some hyperparameters are different). Only runing one of them is suffcient. 
* Step 5: train Recursive Neural Network using recursive_neural_network_for_sentiment_analysis_coding .ipynb
