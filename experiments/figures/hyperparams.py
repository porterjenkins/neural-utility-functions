import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


## CHOICE DATA

movie_logit = pd.read_csv("tuning-movie-logit.csv")

fig,ax=plt.subplots(figsize=(10,6))
ax.set_xlabel("lambda",fontsize=20)
ax.set_ylabel("HR@5", fontsize=30)
ax.set_ylim([.5,1])
ax.plot(movie_logit['lambda'], movie_logit['Hit Ratio'], marker='o', color='b', label='HR@5')
ax.tick_params(axis='both', which='major', labelsize=16)
ax.legend(loc='upper left', fontsize=25)

ax2=ax.twinx()
# make a plot with different y-axis using second axis object
ax2.plot(movie_logit['lambda'],  movie_logit['NDCG'], '-.' ,color="red",marker="o", label='NDCG@5')
ax2.set_ylabel("NCDG@5", fontsize=30)
ax2.tick_params(axis='both', which='major', labelsize=16)
ax2.set_ylim([.5,1])

ax2.legend(loc='upper right', fontsize=25)
plt.savefig("tuning-movie-logit.pdf")


### RANKINGS DATA


movie_ranking = pd.read_csv("tuning-movie-rating.csv")


fig,ax=plt.subplots(figsize=(10,6))
ax.set_xlabel("lambda",fontsize=20)
ax.set_ylabel("MSE",fontsize=30)
ax.set_ylim([0,6])
ax.tick_params(axis='both', which='major', labelsize=16)

ax.plot(movie_ranking['lambda'], movie_ranking['MSE'], marker='o', color='b', label='MSE')
ax.legend(loc='upper left', fontsize=25)

ax2=ax.twinx()
# make a plot with different y-axis using second axis object

ax2.set_ylabel("DCG@5", fontsize=30)
ax2.set_ylim([5,14])
ax2.tick_params(axis='both', which='major', labelsize=16)

ax2.plot(movie_ranking['lambda'],  movie_ranking['DCG'], '-.', color="red",marker="o", label="DCG@5")
ax2.legend(loc='upper right', fontsize=25)

plt.savefig("tuning-movie-rating.pdf")