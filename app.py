from flask import Flask, render_template, request
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree

app = Flask(__name__)

import os
dirname = os.path.dirname(__file__)


@app.route('/')
def index():  # put application's code here
    return render_template('index.html')

@app.route('/contacts/')
def contacts():
    developer_name = "Nick"
    context = {'name': developer_name}
    return render_template('contacts.html', props=context)

@app.route('/notebook/')
def notebook_get():
    return render_template('notebook.html')

@app.route('/notebook/', methods = ['POST'])
def notebook_post():
    df_path = os.path.join(dirname, 'data/mkis32.csv')
    df = pd.read_csv(df_path,skiprows=1)

    df['Дата'] = pd.to_datetime(df.Дата, errors='coerce')
    df["others"] = np.where(df["others"] == "0", 0, 1)
    plt.figure(figsize=(16, 10))
    ax = sns.heatmap(df.corr(), cmap='seismic', annot=True)
    plt.savefig("static/heatmap.png")
    plt.clf()

    df['Месяц'] = df['Дата'].dt.strftime('%m')
    column = ['партнера', 'сотрудника', 'Ирина Викт', 'Конкурсы', 'Занятия', 'Экскурсии', 'Праздники',
              'Театры, аквапарки,…', 'Лагеря', 'Гости']

    aov1 = df.groupby(df['Месяц'])[column[0]].nunique()
    aov1.plot(kind='line', color='green', title="День рождения партнера")
    plt.savefig("static/partner_birthday.png")
    plt.clf()

    df['Месяц'] = df['Дата'].dt.strftime('%m')
    column = ['партнера', 'сотрудника', 'Ирина Викт', 'Конкурсы', 'Занятия', 'Экскурсии', 'Праздники',
              'Театры, аквапарки,…', 'Лагеря', 'Гости']

    aov1 = df.groupby(df['Месяц'])[column[1]].nunique()
    aov1.plot(kind='line', color='green', title="День рождения сотрудника")
    plt.savefig("static/employee_birthday.png")
    plt.clf()

    df['Месяц'] = df['Дата'].dt.strftime('%m')
    column = ['партнера', 'сотрудника', 'Ирина Викт', 'Конкурсы', 'Занятия', 'Экскурсии', 'Праздники',
              'Театры, аквапарки,…', 'Лагеря', 'Гости']

    aov1 = df.groupby(df['Месяц'])[column[2]].nunique()
    aov1.plot(kind='line', color='green', title="event_IrinaVikt")
    plt.savefig("static/event_IrinaVikt.png")
    plt.clf()

    df['Месяц'] = df['Дата'].dt.strftime('%m')
    column = ['партнера', 'сотрудника', 'Ирина Викт', 'Конкурсы', 'Занятия', 'Экскурсии', 'Праздники',
              'Театры, аквапарки,…', 'Лагеря', 'Гости']

    aov1 = df.groupby(df['Месяц'])[column[3]].nunique()
    aov1.plot(kind='line', color='green', title="event_states")
    plt.savefig("static/event_states.png")
    plt.clf()

    df['Месяц'] = df['Дата'].dt.strftime('%m')
    column = ['партнера', 'сотрудника', 'Ирина Викт', 'Конкурсы', 'Занятия', 'Экскурсии', 'Праздники',
              'Театры, аквапарки,…', 'Лагеря', 'Гости']

    aov1 = df.groupby(df['Месяц'])[column[4]].nunique()
    aov1.plot(kind='line', color='green', title="labour")
    plt.savefig("static/labour.png")
    plt.clf()

    df['Месяц'] = df['Дата'].dt.strftime('%m')
    column = ['партнера', 'сотрудника', 'Ирина Викт', 'Конкурсы', 'Занятия', 'Экскурсии', 'Праздники',
              'Театры, аквапарки,…', 'Лагеря', 'Гости']

    aov1 = df.groupby(df['Месяц'])[column[5]].nunique()
    aov1.plot(kind='line', color='green', title="excursions")
    plt.savefig("static/excursions.png")
    plt.clf()

    df['Месяц'] = df['Дата'].dt.strftime('%m')
    column = ['партнера', 'сотрудника', 'Ирина Викт', 'Конкурсы', 'Занятия', 'Экскурсии', 'Праздники',
              'Театры, аквапарки,…', 'Лагеря', 'Гости']

    aov1 = df.groupby(df['Месяц'])[column[6]].nunique()
    aov1.plot(kind='line', color='green', title="Праздники")
    plt.savefig("static/holidays.png")
    plt.clf()

    df['Месяц'] = df['Дата'].dt.strftime('%m')
    column = ['партнера', 'сотрудника', 'Ирина Викт', 'Конкурсы', 'Занятия', 'Экскурсии', 'Праздники',
              'Театры, аквапарки,…', 'Лагеря', 'Гости']

    aov1 = df.groupby(df['Месяц'])[column[7]].nunique()
    aov1.plot(kind='line', color='green', title="culture")
    plt.savefig("static/culture.png")
    plt.clf()

    df['Месяц'] = df['Дата'].dt.strftime('%m')
    column = ['партнера', 'сотрудника', 'Ирина Викт', 'Конкурсы', 'Занятия', 'Экскурсии', 'Праздники',
              'Театры, аквапарки,…', 'Лагеря', 'Гости']

    aov1 = df.groupby(df['Месяц'])[column[8]].nunique()
    aov1.plot(kind='line', color='green', title="camp")
    plt.savefig("static/camp.png")
    plt.clf()

    df['Месяц'] = df['Дата'].dt.strftime('%m')
    column = ['партнера', 'сотрудника', 'Ирина Викт', 'Конкурсы', 'Занятия', 'Экскурсии', 'Праздники',
              'Театры, аквапарки,…', 'Лагеря', 'Гости']
    aov1 = df.groupby(df['Месяц'])[column[9]].nunique()
    aov1.plot(kind='line', color='green', title="hosts")
    plt.savefig("static/hosts.png")
    plt.clf()

    df['Дата'] = pd.DatetimeIndex(df['Дата']).month
    df = df.drop(['Месяц'], axis=1)
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    kmeans = KMeans(n_clusters=4, max_iter=50)
    kmeans.fit(df)
    kmeans.labels_
    silhouette_avg = silhouette_score(df, kmeans.labels_)

    df2 = pd.DataFrame(np.random.randint(0, 2, size=(1000, 11)),
                       columns=['партнера', 'сотрудника', 'Ирина Викт', 'Конкурсы', 'Занятия', 'Экскурсии', 'Праздники',
                                'Театры, аквапарки,…', 'Лагеря', 'Гости', 'others'])
    df2['Дата'] = np.random.randint(1, 10, size=(1000, 1))
    df = df.append(df2, ignore_index=False)

    ssd = []
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 10, 16, 36, 64, 86]
    for num_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
        kmeans.fit(df)

        ssd.append(kmeans.inertia_)

    # plot the SSDs for each n_clusters
    plt.plot(ssd)
    plt.savefig("static/ssd.png")
    plt.clf()

    # Silhouette analysis
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 10, 16, 36, 64, 86, 144, 244]

    silhouettes = dict.fromkeys(range_n_clusters, 0)

    for num_clusters in range_n_clusters:
        # intialise kmeans
        kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
        kmeans.fit(df)

        cluster_labels = kmeans.labels_

        # silhouette score
        silhouette_avg = silhouette_score(df, cluster_labels)
        print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))
        silhouettes[num_clusters] = silhouette_avg

    kmeans = KMeans(n_clusters=86, max_iter=50)
    kmeans.fit(df)
    kmeans.labels_
    df['Cluster_Id'] = kmeans.labels_

    sns.set(rc={'figure.figsize': (30, 10)})
    sns.boxplot(x='Cluster_Id', y='Дата', data=df)
    plt.savefig("static/cluster_id.png")
    plt.clf()

    mergings = linkage(df, method="single", metric='euclidean')
    fig, axes = plt.subplots(figsize=(30, 10))
    dendrogram(mergings)
    plt.savefig("static/mergings_single.png")
    plt.clf()

    mergings = linkage(df, method="complete", metric='euclidean')
    fig, axes = plt.subplots(figsize=(30, 10))
    dendrogram(mergings)
    plt.savefig("static/mergings_complete.png")
    plt.clf()

    mergings = linkage(df, method="average", metric='euclidean')
    fig, axes = plt.subplots(figsize=(30, 10))
    dendrogram(mergings)
    plt.savefig("static/mergings_average.png")
    plt.clf()

    cluster_labels = cut_tree(mergings, n_clusters=86).reshape(-1, )
    df['Cluster_Labels'] = cluster_labels

    sns.boxplot(x='Cluster_Labels', y='Дата', data=df)
    plt.savefig("static/cluster_labels.png")
    plt.clf()

    context = {'silhouette': silhouette_avg, 'silhouettes': silhouettes.items()}
    return render_template('result.html', props=context)



if __name__ == '__main__':
    app.run(debug=True)
