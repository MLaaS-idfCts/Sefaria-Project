import string

s = """
    OVERALL                                             ---  125260        0.99   0.817009  0.819607  0.793470
0                                                  None   95158        0.76   0.851984  0.975386  0.909519
5                                               abraham    2237        0.02   0.819780  0.559220  0.664884
1                                           dinei-yibum     944        0.01   0.828125  0.646341  0.726027
7                   financial-ramifications-of-marriage    1264        0.01   0.730612  0.473545  0.574639
3                                              haggadah    1357        0.01   0.863281  0.556675  0.676876
14                                             idolatry    1035        0.01   0.598039  0.192429  0.291169
6                                                 jacob    1715        0.01   0.830128  0.492395  0.618138
4                                                joseph    1496        0.01   0.801802  0.582969  0.675095
10                                           king-david    1108        0.01   0.720930  0.262712  0.385093
18                            laws-of-judges-and-courts    1755        0.01   0.577778  0.101961  0.173333
2                  laws-of-transferring-between-domains    1203        0.01   0.859779  0.621333  0.721362
12                                           leadership    1409        0.01   0.789916  0.212670  0.335116
17                                             learning    1251        0.01   0.578313  0.120907  0.200000
11                                                moses    2280        0.02   0.660305  0.254786  0.367694
8                                              passover    2336        0.02   0.809117  0.420741  0.553606
15                                               prayer    1703        0.01   0.651163  0.161850  0.259259
19       procedures-for-judges-and-conduct-towards-them    1690        0.01   0.758621  0.090722  0.162063
9          rabbinically-forbidden-activities-on-shabbat    1116        0.01   0.820652  0.410326  0.547101
13                                             teshuvah    1099        0.01   0.698925  0.199387  0.310263
20                                                torah    1600        0.01   0.560606  0.084282  0.146535
16                                                women 
"""
s = ''.join([char for char in s if char not in '1234567890.']).split()


s2 = """
sklearn
seaborn
gensim
nltk
bs4
# python-matplotlib
matplotlib
tqdm
pandas
numpy
lxml
ipynb-py-convert
scikit-multilearn
itertools-s
virtualenv
ipykernel
Unidecode
mplcursors
# fastxml
nano
# vim
scipy
mglearn
"""
s2 = '\n'.join(sorted(s2.split()))
print(s2)