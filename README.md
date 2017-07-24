# l3wtransformer

> A word hashing method to reduce the dimensionality of the bag-of-words term vectors. It is based on letter n-gram. Given a word (e.g. good), it first adds word starting and ending marks to the word (e.g. #good#). Then, breaks the word into letter n-grams (e.g. letter trigrams: #go, goo, ood, od#). Finally, the word is represented using a vector of letter n-grams. 

[Huang et al.2013, Learning Deep Structured Semantic Models for Web Search using Clickthrough Data]

---

This implementation supports the transformation from **text into sequenzes of numbers**, with the numbers indicating the descending word frequency.

For example:

*Lorem ipsum dolor sit amet, consectetuer adipiscing elit...* is transformed into *23, 1, 80, 86, 47, 50001, 21, 59, 83, 93, 14, 50003, 4, 7*

Also, after each word flags indicating lower case, upper case, mixed case or initial capitalization are added. 

### To do

There will be an implementation supporting the transformation from **text into bag-of-word vectors**.
