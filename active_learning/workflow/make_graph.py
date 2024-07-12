from active_learning.model.idnn import IDNN
from active_learning.model.idnn_model import IDNN_Model
from active_learning.workflow.dictionary import Dictionary
import numpy as np
# from CASM_wrapper import compileCASMOutput, loadCASMOutput
from tensorflow import keras
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import load_model
import pandas as pd 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle



# dict = Dictionary('tests/LCO/input.ini')
# model = IDNN_Model(dict)
# # model.load_trained_model(3)


def plot_python(data,title):
    eta = data[:,0:7]
    mu = data[:,7:14]
    free = data[:,14:15]
    eta0_1= eta[:,0]
    eta1_1=eta[:,1]
    g = free.reshape((50,50))-min(free)
    fig2 = plt.figure(figsize=(10, 6))
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.plot_surface(eta0_1.reshape((50,50)), eta1_1.reshape((50,50)), g,
                    cmap='viridis', vmin=0, vmax=np.max(g))
    ax2.set_xlabel(r'$\eta_0$')
    ax2.set_ylabel(r'$\eta_1$')
    ax2.set_zlabel('g')
    ax2.set_ylim(-0.5, 0.5)
    # ax2.set_zlim(0, 1e-5)
    ax2.view_init(21, -135)
    ax2.grid(True)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    plt.gca().invert_yaxis()
    plt.savefig('{}.png'.format(title))


    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.scatter(eta0_1,mu[:,0])
    ax.set_xlabel(r'$\eta_0$')
    ax.set_ylabel(r'$\mu_0$')
    # ax2.set_ylim(-0.5, 0.5)
    # ax2.set_zlim(0, 1e-5)
    # ax2.view_init(21, -135)
    # ax2.grid(True)
    # ax2.tick_params(axis='both', which='major', labelsize=12)
    # plt.gca().invert_yaxis()
    plt.savefig('{}_mu.png'.format(title))

    plt.close('all')


    with open('{}.pxl'.format(title), 'wb') as f:
        pickle.dump(fig2, f)
    plt.clf()


def matlab(title):
    import matlab.engine

    # Start MATLAB engine
    eng = matlab.engine.start_matlab()


    # Call the MATLAB function and pass the value of i
    # eng.eval(f"i={i};", nargout=0) 
    eng.eval(f"var = [{title}  '.txt'];", nargout=0)
    eng.eval("data= load(var);",nargout=0)

    # Plot the figure
    eng.eval("""
    figure
    g = reshape(data(:,end),[50,50])-min(data(:,end));
    surf(reshape(data(:,1),[50,50]),reshape(data(:,2),[50,50]),g,min(g,5e-6))

    xlabel('\eta_0')
    ylabel('\eta_1')
    zlabel('g')
    title('260 K')
    xlim([0.45,0.55])
    zlim([0 1e-5])
    view(-65,16)
    set(gca,'FontSize',16)
    set(gca,'ydir','reverse')
    shading interp 
    grid on 
    hold off
    savefig('graph.fig')
    """, nargout=0)

    # Stop MATLAB engine
    eng.quit()


def predict_and_save(model,eta,T,title):
    pred = model.predict([eta,T])
    free = pred[0]
    mu = pred[1]
    data = np.hstack((eta,mu,free,T))
    np.savetxt('{}.txt'.format(title),data)
    return data
    

def graph(rnd, model,dict):
    [outputFolder,temp,graph,dir_path] = dict.get_individual_keys('Main',['outputfolder','temp','graph','dir_path'])
    model.load_trained_model(rnd)

    # full composition range
    # data = '/Users/jamieholber/Desktop/Software/active-learning/tests/LCO_row/2d_slice_rnd21_0_1.txt'

    # data = '/expanse/lustre/scratch/jholber/temp_project/active-learning/tests/LCO_row/2d_slice_rnd21_0_1.txt'
    data = dir_path + '/graph_points.txt'
    
    eta = np.genfromtxt(data,dtype=np.float32)[:,:7]
    T = np.ones((np.shape(eta)[0],1))*temp
    title1 = outputFolder+'graphs/rnd{}_full'.format(rnd)
    data1= predict_and_save(model,eta,T,title1)


    #near orderings 
    x1 = np.linspace(.45, .55, 50)
    x2 = np.linspace(-.5,.5,50)
    x1_grid, x2_grid = np.meshgrid(x1, x2)
    x1_flat = np.reshape(x1_grid.flatten(),(2500,1))
    x2_flat = np.reshape(x2_grid.flatten(),(2500,1)) 
    # eta = np.hstack((x1_flat,x2_flat,x1_flat*0,x1_flat*0))
    eta = np.hstack((x1_flat,x2_flat,x1_flat*0,x1_flat*0,x1_flat*0,x1_flat*0,x1_flat*0))
    T = np.ones((np.shape(eta)[0],1))*temp
    title2 = outputFolder+'graphs/rnd{}'.format(rnd)
    data2 = predict_and_save(model,eta,T,title2)

    if graph=='both' or graph=='python':
        plot_python(data1,title1)
        plot_python(data2,title2)
    if graph=='both' or graph=='matlab':
        matlab(title1)
        matlab(title2)








