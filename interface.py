from FlexibleSkylineOperator import *
import pymongo
import certifi
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
import matplotlib
import operator as op
import time
ca = certifi.where()

def generateNonDominated(input,dim,A,ops):
    fso = FlexibleSkylineOperator(dim)
    nd = fso.alg_SVE1F_ND(input, A, ops)
    return nd

def extract_tuples(limits):
    myclient = pymongo.MongoClient(
        "PrivateClientKey!", tlsCAFile=ca)
    db = myclient["DB"]
    topics = db["Topics_Authored"]
    return list(topics.find({},{'distribution': 1, 'posted_at': 1, '_id': 0}, limit=limits))

def initialize_operator(extracted, fields, time_dist):
    dates = list()
    insertion_tuples = list()
    A = np.array([[-1,1,0]])
    ops = [op.ge]

    for element in extracted:
        dates.append(element['posted_at'])
        local_tuple = []
        for index in fields:
            local_tuple.append((element['distribution'][index][1]))
        insertion_tuples.append(tuple(local_tuple))
    # Conseguido!!!
    dates = time_series_normalize(dates)
    #Tuples generation routine
    tuples = []
    for i in range(len(insertion_tuples)):
        tuples.append((insertion_tuples[i], dates[i][0]))

    ## Tuple format: [((param1:float,param2:float),date: float)]
    operator = FlexibleSkylineOperator(len(fields),time_dist, tuples)
    start_time = time.time()
    nd = operator.alg_SVE1F_ND(A,ops)
    general_set = [(el[0][0],el[0][1],el[1]) for el in tuples]
    return nd, general_set

# Normalizes a time series
def time_series_normalize(time_series):
    time_series = pd.Series(time_series)
    time_series = pd.to_datetime(time_series)
    time_series = time_series.values.reshape((len(time_series), 1))
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(time_series)
    return scaler.transform(time_series)

### Data visualization functions

## Grid plotting for different parameters
def plot_3d_generator():
    results = []
    extracted = extract_tuples(200)
    time_dists= [[id_dist,id_dist],[x_med,id_dist],[id_dist,x_med]]
    for time_dist in time_dists:
        nd, insertion = initialize_operator(extracted, [3,6], time_dist)
        ndtuples = [(el[0][0][0],el[0][0][1],el[0][1]) for el in nd]
        results.append(ndtuples)
    return results

## Base 3d plotting function
def plot_3d(nd1, nd2, nd3):
    nd1_x, nd1_y, nd1_z = zip(*nd1)
    nd2_x, nd2_y, nd2_z = zip(*nd2)
    nd3_x, nd3_y, nd3_z = zip(*nd3)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    plt.title("ND sets for different time distributions")
    ax.scatter(nd1_x, nd1_y, nd1_z, zorder=0, label="ND1 set", s=70, alpha=1)
    ax.scatter(nd2_x, nd2_y, nd2_z, zorder=5, label="ND2 set", s=120, alpha=1)
    ax.scatter(nd3_x, nd3_y, nd3_z, zorder=10, label="ND3 set", s=180, alpha=1)
    plt.legend()
    ax.set_xlabel('Variable 1')
    ax.set_ylabel('Variable 2')
    ax.set_zlabel('Time')
    #plt.savefig("example.png", dpi=100)
    plt.show()

# Basic plotting function
def plotND(input, nd):
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    })
    input_x, input_y = zip(*input)
    nd_x, nd_y = zip(*nd)
    plt.figure(figsize=(6, 2.5))
    plt.title("r set and ND set")
    plt.scatter(input_x, input_y, zorder=10, label="r set", s=50, alpha=0.2)
    plt.scatter(
        nd_x,
        nd_y,
        zorder=5,
        label="ND set",
        s=150,
        alpha=1,
    )
    plt.legend()
    plt.xlim([0, 0.4])
    plt.ylim([0, 0.4])
    plt.xlabel("Variable 1")
    plt.ylabel("Variable 2")
    plt.grid(True, alpha=0.5, ls="--", zorder=0)
    plt.tight_layout()
    #plt.savefig("example.png", dpi=100)
    plt.savefig('CSQ_2.pgf')
    plt.show()
    