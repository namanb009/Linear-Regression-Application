import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tkinter import *
from tkinter import filedialog

def hypo(theta, values, col):
    result = 0
    
    print(theta)
    print(values)

    for i in range(col):
        result=result+ theta[i]*values[i]


    return result

def cost_function(X, y, theta):
    m = y.size
    error = np.dot(X, theta.T) - y
    cost = 1/(2*m) * np.dot(error.T, error)
    
    return cost, error

def gradient_descent(X, y, theta, alpha, iters):
    cost_array = np.zeros(iters)
    m = y.size
    for i in range(iters):
        cost, error = cost_function(X, y, theta)
        theta = theta - (alpha * (1/m) * np.dot(X.T, error))
        cost_array[i] = cost
    return theta, cost_array

def plotChart(iterations, cost_num):
    fig, ax = plt.subplots()
    ax.plot(np.arange(iterations), cost_num, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Error vs Iterations')
    plt.style.use('fivethirtyeight')
    plt.show()

def run(filename, col, values):
    # Import data
    data = pd.read_csv(filename)

    # Extract data into X and y
    X = data.iloc[:,0:col]

    y = data.iloc[:, -1]
    y=y.rename_axis(None)

    # Normalize our features
    X = (X - X.mean()) / X.std()

    # Add a 1 column to the start to allow vectorized gradient descent
    X = np.c_[np.ones(X.shape[0]), X] 

    # Set hyperparameters
    alpha = 1
    iterations = 3000

    # Initialize Theta Values to 0
    theta = np.zeros(X.shape[1])
    initial_cost, _ = cost_function(X, y, theta)

    print('With initial theta values of {0}, cost error is {1}'.format(theta, initial_cost))

    # Run Gradient Descent
    theta, cost_num = gradient_descent(X, y, theta, alpha, iterations)

    # Display cost chart
    plotChart(iterations, cost_num)

    final_cost, _ = cost_function(X, y, theta)

    print('With final theta values of {0}, cost error is {1}'.format(theta, final_cost))
    print(hypo(theta,values,col))

if __name__ == "__main__":
    root = Tk()

    #Root Window Declaration
    root.title("Multivarite Linear Reg. using Gradient Descent")
    root.configure(background="bisque")
    
    ##########################################################################################################################
    #Frame Declaration
    ff = Frame(root, width=300, height=200)
    ff.pack(side="top")

    ##########################################################################################################################
    #Initial data Declaration e1 e2 e3

    l2 = Label(ff, text="Number of parameters: ",font=("Arial", 15)).grid(row=1, column=0, sticky=W, padx=1)
    e2 = Entry(ff, font=("Arial", 12), width=30)
    e2.grid(row=1,column=1,sticky=W,padx=2,pady=5)

    l3 = Label(ff, text="Enter input values seperated by whitespace: ", font=("Arial", 15)).grid(row=2, column=0, sticky=W, padx=1)
    e3 = Entry(ff, font=("Arial", 12), width=30)
    e3.grid(row=2, column=1, sticky=W, padx=2, pady=5)
    
    l4 = Label(ff, text="Number of iterations: ", font=("Arial", 15)).grid(row=3, column=0, sticky=W, padx=1)
    e4 = Entry(ff, font=("Arial", 12), width=30)
    e4.grid(row=3,column=1,sticky=W,padx=2,pady=5)


    L11 = Label(ff, text="Data File   ", font=("Arial", 15)).grid(row=4, column=0, sticky=W, padx=2, pady=5)
    L12 = Label(ff, text="Current Selection", font=("Arial", 15)).grid(row=5, column=0, sticky=W, padx=2, pady=5)
    
    add = []
    ##########################################################################################################################
    def browse(add):
        root.fileName = filedialog.askopenfilename()
        L13 = Label(ff, text=root.fileName, font=("Arial", 12)).grid(row=5, column=1, sticky=W, padx=2, pady=5)

        return root.fileName
        

    def go(add):
        col = int(e2.get())
        
        temp = e3.get().split()
        
        values = []
        for i in range(col):
            values.append(int(temp[i]))

        run(add,col,values)



    ##########################################################################################################################
    B1 = Button(ff, text="Fetch", command=lambda: add.append(browse(add)), height=1, width=33, font=("Arial", 10)).grid(row=4, column=1, sticky=W, padx=2, pady=5)
    B2 = Button(ff, text="Go", command=lambda: go(add[0]), height=1, width=30, font=("Arial", 13)).grid(row=7, column=1, sticky=W, padx=2, pady=10)

    root.mainloop()


    






