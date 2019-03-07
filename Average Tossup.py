import pandas as pd
from pandas import ExcelWriter
import numpy as np
import os
import csv
import matplotlib.pyplot as plt

os.chdir("C:\Users\dhoth\Documents\Quizbowl\ACF Regionals 2019")

df = pd.read_csv("regionals19-tossups.tsv", delimiter = '\t')

#remove bad data
df = df[df.buzz_location > 0]
df = df[df.buzz_location_pct <= 1]
#print(df.head(10))

TUH_Count = 0

def find_most_negged():
    df1 = df.groupby(["answer"]).filter(lambda x: len(x) >= 10)
    df1 = df1.groupby(['answer','buzz_value']).size().rename('count')
    d_pcts = (df1/df1.groupby(level=0).sum()).rename('Percentage')
    d_pcts = d_pcts.reset_index()
    d_pcts = d_pcts[d_pcts.buzz_value == -5]
    d_pcts = d_pcts.sort_values(by = ['Percentage'], ascending = False)
    print(d_pcts.head(10))

def late_buzzes():
    df1 = df[df.buzz_value == 10]
    df1 = df1[df1.buzz_location != 0]
    df1 = df1.groupby(['answer']).filter(lambda x: len(x) >= 10)
    df1 = df1.groupby(['answer'])['buzz_location_pct'].aggregate(np.mean).rename('Avg. Buzz Pct')
    print(df1.reset_index().sort_values(by='Avg. Buzz Pct',ascending = False).head(10))

def find_most_skewed():
    df1 = df[df.buzz_value == 10]
    df1 = df1[df1.buzz_location != 0]
    df1 = df1.groupby(['answer']).filter(lambda x: len(x) >= 10)
    df1 = df1.groupby(['answer'])['buzz_location_pct'].aggregate('skew').rename('Skew')
    print(df1.reset_index().sort_values(by='Skew',ascending = False).head(10))

def sort_pdf():
    #toggle on or off depending upon if you include bouncebacks
    df1 = df[df.bounceback != 'bounceback']
    df1 = df1[df1.buzz_value == 10]
    #df1 = df[df.buzz_value == 10]
    max_gets = len(df1)
    print(max_gets)
    df1 = df1.groupby(['buzz_location_pct']).size().rename('count')
    pdf_array = np.zeros(1000)
    for x in df1.index:
        pdf_array[int(1000*x)-1] += df1.loc[x]
    pdf_array = pdf_array / np.linalg.norm(pdf_array,ord = 1)
    return pdf_array



def sort_cdf():
    pdf_array = sort_pdf()
    cum_array = np.cumsum(pdf_array)
    return cum_array

def plot_cdf(cum_array = None, best_poly = None):
    if cum_array is None:
        cum_array = sort_cdf()
    x_val = np.linspace(.001,1,1000)
    plt.title("CDF Comparison Without Bouncebacks")
    plt.plot(x_val, cum_array, color = 'red', label = "ACF Regs 2019")
    plt.plot(x_val, np.square(x_val), color = 'green', label = 'y = x^2')
    plt.plot(x_val, np.power(x_val, 3), color = 'yellow', label = 'y = x^3')
    plt.xlabel('Buzz Location Pct')
    plt.ylabel("Cumulative Pct")
    if best_poly is not None:
        p = np.poly1d(best_poly)
        plt.plot(x_val, p(x_val), color = 'blue', label = 'Best Fit Deg '+str(len(best_poly)-1))
    plt.legend(shadow = True)
    plt.grid(True)
    plt.show()

def poly_best_fit(degree):
    cdf_array = sort_cdf()
    x_val = np.linspace(.001,1,1000)
    best_poly = np.polyfit(x = x_val, y = cdf_array, deg = degree)
    return best_poly
#find_most_negged()
#late_buzzes()
#find_most_skewed()
#sort_pdf()
print(poly_best_fit(3))
plot_cdf(best_poly = poly_best_fit(3))
