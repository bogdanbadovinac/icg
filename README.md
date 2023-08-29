## Image caption generator

Cilj projekta je generisati opis zadate slike. Skup podataka sa kojim se radi se sastoji od osam hiljada slika i za svaku od tih slika po pet opisa. CNN je korišćena za slike i LSTM za tekst. BLEU score je korišćen kao metrika za računanje performansi treniranog modela.

Link ka skupu podataka: https://www.kaggle.com/datasets/adityajn105/flickr8k

### Korišćene mreže
* VGG16 Network
* CNN-LSTM Network

### Potrebne biblioteke
* numpy
* matplotlib
* keras
* tensorflow
* nltk

### Rezultati
* BLEU-1 Score: 0.515462
* BLEU-2 Score: 0.288135

### Korišćeno okruženje
* Google colab