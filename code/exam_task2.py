

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_airfoil(x, y, figsize=(12,4)):
    """Plot the airfoil desribed by nodes (x[i], y[i]) """

    plt.figure(figsize=figsize)
    plt.plot(x,y)

    ax = plt.gca()
    ax.set_ylim([-0.1, 0.1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")  
    ax.set_title("Airfoil")

    plt.show()

def plot_cp_distribution(x, cp, title="Cp Distribution", figsize=(8,6)):
    """Plot the Cp distribution of one simulation
    
    Don't worry if you have never seen a Cp distribution. You can just treat it as an dimensionless output vector of size 192"""

    plt.figure(figsize=figsize)
    plt.plot(x, cp, "o-")

    ax = plt.gca()
    ax.set_ylim(ax.get_ylim()[::-1]) # invert the y-axis of the plot
    ax.grid(True)
    ax.set_xlabel("X")
    ax.set_ylabel("Cp")    
    ax.set_title(title)

    plt.show()


def main():
    # load input data
    flow_conditions = pd.read_csv("flow_conditions.csv", index_col=0)
    print("flow_conditions:")
    print(flow_conditions)
    
    # load simulation results
    surface_flow_sim_results = pd.read_csv("surface_flow_sim_results.csv", index_col=0)
    print("surface_flow_sim_results:")    
    print(surface_flow_sim_results)

    # get the coordinates of the airfoil mesh nodes
    # (the geometry does not change between simulation runs, so the first 192 values are enough)
    nodes_x = surface_flow_sim_results["x"][:192].values
    nodes_y = surface_flow_sim_results["y"][:192].values

    plot_airfoil(nodes_x, nodes_y)

    # the only output variable we care about in this assignment is the pressure coefficient Cp
    # we can extract the values and turn the pandas.Series object into a numpy array (2D matrix of size n_sim x n_nodes)
    Cp = surface_flow_sim_results["Pressure_Coefficient"].values
    Cp = Cp.reshape(-1,192)
    print("Cp.shape:", Cp.shape)
    Y = Cp # this is the output matrix from the assignment

    # the same can be done with the input data (flow conditions)
    X = flow_conditions[["Ma", "AoA"]].values
    print("X.shape:", X.shape)

    # plot the Cp distribution of some simulation results
    plot_cp_distribution(nodes_x, Cp[0], title="Cp Distribution i=0 at $M_{\infty}$="+f"{X[0,0]} and AoA={X[0,1]} [deg.]")
    plot_cp_distribution(nodes_x, Cp[1], title="Cp Distribution i=1 at $M_{\infty}$="+f"{X[1,0]} and AoA={X[1,1]} [deg.]")
    plot_cp_distribution(nodes_x, Cp[100], title="Cp Distribution i=100 at $M_{\infty}$="+f"{X[100,0]} and AoA={X[100,1]} [deg.]")

    # now do the analysis...






    # output the results to csv
    Cp_output_data = np.zeros((80,192))   # change this to the true output data generated from 'flow_conditions_test.csv'

    pd.DataFrame(Cp_output_data).to_csv("cp_output_data.csv")




if __name__ == "__main__":
    main()