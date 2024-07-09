The preprocessing of the datasets involved several essential steps to prepare the data for analysis. We cleaned the text for all datasets by removing noise, special characters (e.g., @, #, $, %), numbers, and common English stopwords (e.g., the, is, in, at). After cleaning, we tokenized the text into individual words or tokens and then applied lemmatization or stemming to reduce words to their base forms. The data was split into training (70%), validation (10%), and test (20%) sets using stratified sampling on the time attribute to ensure a uniform distribution of time periods across the splits. We also removed duplicates, texts with less than 10 words, lowercase all words, and truncated longer texts to the first and last 256 tokens to fit BERT’s input limit of 512 tokens. For the ArXiv dataset, we extracted the year, month, and day from the 'versions' field and included only records from 2001 to 2020, cleaning author names and categories. In the Ciao dataset, we removed reviews rated as not helpful, included only those with a minimum length of 10 words and a helpfulness rating, and converted the text to lowercase. We removed URLs, special characters, non-English characters, and stopwords for the Reddit dataset, ensuring a clean dataset, and filtered authors who posted in at least four distinct subreddits. Finally, for the Yelp dataset, reviews were filtered based on length (minimum of 10 words) and user activity between 2010 and 2022, considering users with at least 100 reviews during this period. 

In summary, the following are the steps followed to preprocess each dataset:

Common Steps:

1. Text Cleaning:

- Removed special characters (e.g., @, #, $, %, etc.) to avoid noise in the data.
- Removed numbers to prevent numerical bias in text analysis.
- Removed common English stopwords (e.g., the, is, in, at, etc.) to focus on meaningful words.

2. Tokenization:

- Split text into individual words or tokens suitable for model training. For example, "This is an example" becomes ["This," "is," "an," "example"].

3. Lemmatization and Stemming:

- Lemmatization was used to convert words to their base or dictionary form (e.g., "running" becomes "run").
- Stemming was applied to reduce words to their root forms (e.g., "connections" becomes "connect").

4. Filtering and Splitting:

- Removed duplicates and text (reviews) with less than 10 words.

5. Lowercase all words.
- Split data into training (70%), validation (10%), and test (20%) set using stratified sampling on the time attribute to ensure a uniform distribution of time periods across the splits.


Specific Steps:

ArXiv Dataset:

- Extracted the year, month, and day from the 'versions' field and filtered records to include only those from 2001 to 2020.
- Cleaned the author names and categories, removing special characters and ensuring uniformity.
- Created a graph based on the Jaccard similarity of author sets between different subject classes, with an edge existing if the similarity exceeds 0.01.

Ciao Dataset:

- Further removed reviews rated as not helpful.
- Filtered reviews to include only those with a minimum length of 10 words and a helpfulness rating.
- Converted text to lowercase and assigned a binary label based on the rating (1-2 stars as 0, 3-5 stars as 1).
- Created a directed graph based on explicit trust relations between users.

Reddit Dataset:

- Removed URLs, special characters, Chinese and Japanese characters, and stopwords to ensure a clean dataset.
- Filtered authors who posted in at least four different subreddits to ensure balance.
- Created a graph based on user overlap within subreddits similar to the ArXiv dataset.

Yelp Dataset:

- Filtered reviews based on length (minimum of 10 words) and user activity between 2010 and 2022 (users with at least 100 reviews during this period).
- Converted text to lowercase and assigned a binary label based on the rating (1-2 stars as 0, 3-5 stars as 1).
- Created a directed graph based on friendship relations between users.
