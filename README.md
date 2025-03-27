# oam-nltk


**co-occurence.sliding.window.py**
Feature:
* map documents with oam lexicons
* use sliding window to see the co-occurence frenquency and words' relationship
* provide original matrix and also a graph built based on normalized weight
* Significance Testing upon the weight, complementing the node graph

**tfidf.radar.matrix.py**
Feature:
* map documents with oam lexicons
* gather the result into tfidf matrix to filter tops
* and show results in radar graph.

**trans_pdf.py**
transist the pdf file to txt file. Currently only txt could be converted, but the texts in the graph cannot.


### Validation Matrix
We perform a fisher exact test on the co-occurence matrix we calculated.

p-value ≤ 0.05: This indicates that the observed association between the terms is statistically significant, and you can reject the null hypothesis. In other words, there's strong evidence that the focus term and association term occur together more often than expected by chance.

p-value > 0.05: This indicates that the observed association is not statistically significant, and you fail to reject the null hypothesis. This means that there's not enough evidence to suggest an association between the terms.


### Pre install packages
```bash
pip install -r requirements.txt
```
