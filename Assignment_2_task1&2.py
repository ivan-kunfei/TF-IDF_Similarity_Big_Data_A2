import sys
import re
import numpy as np
from numpy.linalg import norm
from pyspark import SparkContext
from pyspark.sql import SQLContext

# Set the file paths on your local machine
# Change this line later on your python script when you want to run this on the CLOUD (GC or AWS)

if __name__ == "__main__":
	wikiPagesFile = sys.argv[1]
	wikiCategoryFile = sys.argv[2]
	output_dir_1 = sys.argv[3]
	output_dir_2 = sys.argv[4]
	output_dir_3 = sys.argv[5]

	# wikiPagesFile = "WikipediaPages_oneDocPerLine_1000Lines_small.txt"
	# wikiCategoryFile = "wiki-categorylinks-small.csv.bz2"
	# output_dir_1 = 're_1'
	# output_dir_2 = 're_2'
	# output_dir_3 = 're_3'

	sc = SparkContext()

	wikiCategoryLinks = sc.textFile(wikiCategoryFile)

	wikiCats = wikiCategoryLinks.map(lambda x: x.split(",")).map(
		lambda x: (x[0].replace('"', ''), x[1].replace('"', '')))

	wikiCategoryLinks.take(2)

	wikiCats.take(2)

	# Now the wikipages
	wikiPages = sc.textFile(wikiPagesFile)
	# wikiPages.take(1)

	sql_context = SQLContext(sc)

	# Assumption: Each document is stored in one line of the text file
	# We need this count later ...
	numberOfDocs = wikiPages.count()

	print("Number of docs: {}".format(numberOfDocs))
	# Each entry in validLines will be a line from the text file
	validLines = wikiPages.filter(lambda x: 'id' in x and 'url=' in x)

	# Now, we transform it into a set of (docID, text) pairs
	keyAndText = validLines.map(lambda x: (x[x.index('id="') + 4: x.index('" url=')], x[x.index('">') + 2:][:-6]))

	# keyAndText.take(1)

	# Now, we split the text in each (docID, text) pair into a list of words
	# After this step, we have a data set with
	# (docID, ["word1", "word2", "word3", ...])
	# We use a regular expression here to make
	# sure that the program does not break down on some of the documents

	punctuations = [',', '.', ':', ';', '?', '``', "''", '(', ')', '`', '`', '[', ']', '&', '!', '*', '@', '#', '$',
					'%']


	def get_tokens(text):
		text = text.lower()
		tokens = re.findall('[A-Za-z]+', text)
		return tokens


	keyAndListOfWords = keyAndText.map(lambda x: (str(x[0]), get_tokens(x[1])))
	keyAndListOfWords.take(1)

	# to ("word1", 1) ("word2", 1)...
	allWords = keyAndListOfWords.flatMap(lambda x: [(word, 1) for word in x[1]])
	# allWords.collect()

	# Now, count all of the words, giving us ("word1", 1433), ("word2", 3423423),etc.
	allCounts = allWords.reduceByKey(lambda x, y: x + y)

	# Get the top 20,000 words in a local array in a sorted format based on frequency
	# If you want to run it on your laptio, it may a longer time for top 20k words.
	topWords = allCounts.top(20000, key=lambda x: x[1])
	print("Top Words in Corpus:", allCounts.top(10, key=lambda x: x[1]))

	# We'll create a RDD that has a set of (word, dictNum) pairs
	# start by creating an RDD that has the number 0 through 20000
	# 20000 is the number of words that will be in our dictionary
	topWordsK = sc.parallelize(range(20000))

	# Now, we transform (0), (1), (2), ... to ("MostCommonWord", 1)
	# ("NextMostCommon", 2), ...
	# the number will be the spot in the dictionary used to tell us
	# where the word is located
	dictionary = topWordsK.map(lambda x: (topWords[x][0], x))
	print("Word Postions in our Feature Matrix. Last 20 words in 20k positions: ",
		  dictionary.top(20, key=lambda x: x[1]))


	def build_array(list_of_indices):
		return_val = np.zeros(20000)

		for index in list_of_indices:
			return_val[index] = return_val[index] + 1

		my_sum = np.sum(return_val)
		return_val = np.divide(return_val, my_sum)

		return return_val


	# Next, we get a RDD that has, for each (docID, ["word1", "word2", "word3", ...]),
	# ("word1", docID), ("word2", docId), ...

	allWordsWithDocID = keyAndListOfWords.flatMap(lambda x: ((j, x[0]) for j in x[1]))

	# print(allWordsWithDocID.take(3))

	# Now join and link them, to get a set of ("word1", (dictionaryPos, docID)) pairs
	allDictionaryWords = dictionary.join(allWordsWithDocID)

	# Now, we drop the actual word itself to get a set of (docID, dictionaryPos) pairs
	justDocAndPos = allDictionaryWords.map(lambda x: (x[1][1], x[1][0]))

	# Now get a set of (docID, [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
	allDictionaryWordsInEachDoc = justDocAndPos.groupByKey()

	# The following line this gets us a set of
	# (docID,  [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
	# and converts the dictionary positions to a bag-of-words numpy array...

	allDocsAsNumpyArrays = allDictionaryWordsInEachDoc.map(lambda x: (x[0], build_array(x[1])))
	print(allDocsAsNumpyArrays.take(3))

	# Now, create a version of allDocsAsNumpyArrays where, in the array,
	# every entry is either zero or one.
	# A zero means that the word does not occur,
	# and a one means that it does.

	def build_zero_one_array(list_of_indices):
		return_val = np.zeros(20000)
		# listOfIndices = set ([listOfIndices])
		for index in list_of_indices:
			return_val[index] = 1
		return return_val


	zeroOrOne = allDictionaryWordsInEachDoc.mapValues(build_zero_one_array)

	# zeroOrOne.take(10)

	# Now, add up all of those arrays into a single array, where the
	# i^th entry tells us how many
	# individual documents the i^th word in the dictionary appeared in
	dfArray = zeroOrOne.reduce(lambda x1, x2: ("", np.add(x1[1], x2[1])))[1]

	# Create an array of 20,000 entries, each entry with the value numberOfDocs (number of docs)
	numberOfDocs = wikiPages.count()
	multiplier = np.full(20000, numberOfDocs)
	# Get the version of dfArray where the i^th entry is the inverse-document frequency for the
	# i^th word in the corpus
	idfArray = np.log(np.divide(multiplier, dfArray))

	# Finally, convert all of the tf vectors in allDocsAsNumpyArrays to tf * idf vectors
	allDocsAsNumpyArraysTFidf = allDocsAsNumpyArrays.map(lambda x: (x[0], np.multiply(x[1], idfArray)))
	print(allDocsAsNumpyArraysTFidf.take(2))
	allDocsAsNumpyArraysTFidf.cache()
	# Now, we join it with categories, and map it after join so that we have only the wikipageID
	# This joun can take time on your laptop.
	# You can do the join once and generate a new wikiCats data and store it. Our WikiCategories includes all categories
	# of wikipedia.

	# featuresRDD = wikiCats.join(allDocsAsNumpyArraysTFidf).map(lambda x: (x[1][0], x[1][1]))
	# featuresRDD.take(10)

	# Let us count and see how large is this data set.

	# featuresRDD.cache()

	'''################### TASK 2  ##################'''
	# Finally, we have a function that returns the prediction for the label of a string, using a kNN algorithm

	# Create an RDD out of the text input

	def cousin_sim(x, y):
		norm_a = np.linalg.norm(x)
		norm_b = np.linalg.norm(y)
		return np.dot(x, y) / (norm_a * norm_b)

	# Finally, we have a function that returns the prediction for the label of a string, using a kNN algorithm
	def get_prediction(text_input, k):
		my_doc = sc.parallelize([(text_input)])

		# Flat map the text to (word, 1) pair for each word in the doc
		words_in_doc = my_doc.flatMap(get_tokens)

		words_in_doc = words_in_doc.map(lambda x: (x, 1))

		# This will give us a set of (word, (dictionaryPos, 1)) pairs
		# 这里的1看作是 doc_id
		all_dictionary_words_in_doc = dictionary.join(words_in_doc).map(lambda x: (x[1][1], x[1][0])).groupByKey()

		# # Get tf array for the input string
		my_array = build_array(all_dictionary_words_in_doc.top(1)[0][1])
		# # Get the tf * idf array for the input string
		my_array = np.multiply(my_array, idfArray)
		# Get the distance from the input text string to all database documents, using cosine similarity (np.dot() )

		distances = allDocsAsNumpyArraysTFidf.map(lambda x: (x[0], cousin_sim(x[1], my_array)))
		# distances = allDocsAsNumpyArraysTFidf.map (lambda x : (x[0], cousinSim (x[1],myArray)))
		# get the top k distances
		distances = wikiCats.join(distances).map(lambda x: (x[1][0], x[1][1]))
		top_k = distances.top(k, key=lambda x: x[1])

		# and transform the top k distances into a set of (docID, 1) pairs
		doc_id_represented = sc.parallelize(top_k).map(lambda x: (x[0], 1))

		# now, for each docID, get the count of the number of times this document ID appeared in the top k
		num_times = doc_id_represented.reduceByKey(lambda x, y: x + y)

		# Return the top 1 of them.
		# Ask yourself: Why we are using twice top() operation here?
		return num_times.top(k, key=lambda x: x[1])


	re_1 = get_prediction('Sport Basketball Volleyball Soccer', 10)
	print(re_1)
	data = sc.parallelize(re_1)
	data = data.coalesce(1)
	data.saveAsTextFile(output_dir_1)

	re_2 = get_prediction('What is the capital city of Australia?', 10)
	print(re_2)
	data = sc.parallelize(re_2)
	data = data.coalesce(1)
	data.saveAsTextFile(output_dir_2)

	re_3 = get_prediction('How many goals Vancouver score last year?', 10)
	print(re_3)
	data = sc.parallelize(re_3)
	data = data.coalesce(1)
	data.saveAsTextFile(output_dir_3)
