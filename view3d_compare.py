import tkinter.filedialog

import h5py
import numpy as np
from mayavi import mlab

from data_util.body_dataset import body_dataset
from data_util.cloth_dataset import cloth_dataset

RED = (0.7254901960784313, 0.09803921568627451, 0.1411764705882353)
BLUE = (0.2, 0.3, 1)
SCALE_FACTOR = 0.03


def model2xyz(model):
    return model[:, 0], model[:, 1], model[:, 2]


def model2xyzs(model):
    return model[:, 0], model[:, 1], model[:, 2], model[:, -1]


def view_one():
    # while True:
    fns = tkinter.filedialog.askopenfilenames(title='选择文件', filetypes=[('所有文件', '.*'), ('文本文件', '.txt')])
    print(fns)
    if len(fns) == 0: exit(0)
    figure = mlab.figure(bgcolor=(1, 1, 1), size=(800, 700))
    flag = True
    for fn in fns:
        model1 = np.loadtxt(fn, delimiter=' ')
        model1 = model1[:2048]
        # without labels
        if model1.shape[-1] == 2:
            model1 = np.concatenate([model1, np.zeros([model1.shape[0], 1])], axis=-1)
        x, y, z = model2xyz(model1)
        if flag:
            color = BLUE
            flag = False
            mlab.points3d(x, y, z, color=color, figure=figure, scale_factor=SCALE_FACTOR)
        else:
            color = RED
            flag = True
            mlab.points3d(x, y, z, color=color, figure=figure, scale_factor=SCALE_FACTOR)

    # with labels
    # x,y,z,s=model2xyzs(model1)
    # figure = mlab.figure(bgcolor=(1,1,1))
    # mlab.points3d(x,y,z,s,figure=figure,scale_mode="none",scale_factor=0.06)
    # # mlab.savefig(str(fn).split('.')[0]+".jpg",figure=figure)
    # mlab.show()
    mlab.show()


def view_model(model):
    figure = mlab.figure(bgcolor=(1, 1, 1), size=(800, 700))

    model1 = model[:2048]
    x, y, z = model2xyz(model1)
    color = BLUE
    mlab.points3d(x, y, z, color=color, figure=figure, scale_factor=SCALE_FACTOR)

    mlab.show()


def compare_two():
    # while True:
    fns = tkinter.filedialog.askopenfilenames(title='选择文件', filetypes=[('所有文件', '.*'), ('文本文件', '.txt')])
    print(fns)
    if len(fns) == 0: exit(0)
    figure1 = mlab.figure(bgcolor=(1, 1, 1), size=(800, 700))

    figure2 = mlab.figure(bgcolor=(1, 1, 1), size=(800, 700))
    if len(fns) is not 3: return

    model1 = np.loadtxt(fns[0], delimiter=' ')
    model2 = np.loadtxt(fns[1], delimiter=' ')
    model3 = np.loadtxt(fns[2], delimiter=' ')

    x, y, z = model2xyz(model1)
    color1 = BLUE  # blue
    color2 = RED  # red
    mlab.points3d(x, y, z, color=color1, figure=figure1, scale_factor=SCALE_FACTOR)
    mlab.points3d(x, y, z, color=color1, figure=figure2, scale_factor=SCALE_FACTOR)

    x, y, z = model2xyz(model2)
    mlab.points3d(x, y, z, color=color2, figure=figure1, scale_factor=SCALE_FACTOR)

    x, y, z = model2xyz(model3)
    mlab.points3d(x, y, z, color=color2, figure=figure2, scale_factor=SCALE_FACTOR)

    mlab.show()


def view_disp():
    # while True:
    fns = tkinter.filedialog.askopenfilenames(title='选择文件', filetypes=[('所有文件', '.*'), ('文本文件', '.txt')])
    print(fns)
    if len(fns) != 2: exit(-1)
    figure = mlab.figure(bgcolor=(1, 1, 1), size=(800, 700))
    flag = True
    model1 = np.loadtxt(fns[0], delimiter=' ')[:512, :]
    model2 = np.loadtxt(fns[1], delimiter=' ')[:512, :]
    disp = model2 - model1
    x, y, z = model2xyz(model1)
    u, v, w = model2xyz(disp)
    mlab.quiver3d(x, y, z, u, v, w, figure=figure, scale_factor=1)
    mlab.show()

def model_disp(model1,model2):
    figure = mlab.figure(bgcolor=(1, 1, 1), size=(800, 700))
    disp = model2 - model1
    x, y, z = model2xyz(model1)
    u, v, w = model2xyz(disp)
    mlab.quiver3d(x, y, z, u, v, w, figure=figure, scale_factor=1)
    mlab.show()

def compare_two_cpd():
    while True:
        fns = tkinter.filedialog.askopenfilenames(title='选择文件', filetypes=[('所有文件', '.*'), ('文本文件', '.txt')])
        print(fns)
        if len(fns) == 0: exit(0)
        figure1 = mlab.figure(bgcolor=(1, 1, 1), size=(800, 700))
        figure2 = mlab.figure(bgcolor=(1, 1, 1), size=(800, 700))
        if len(fns) is not 3: continue

        model1 = np.loadtxt(fns[1], delimiter=' ')
        model2 = np.loadtxt(fns[0], delimiter=' ')
        model3 = np.loadtxt(fns[2], delimiter=' ')

        x, y, z = model2xyz(model1)
        color1 = BLUE
        color2 = RED
        mlab.points3d(x, y, z, color=color1, figure=figure1, scale_factor=SCALE_FACTOR)
        mlab.points3d(x, y, z, color=color1, figure=figure2, scale_factor=SCALE_FACTOR)

        x, y, z = model2xyz(model2)
        mlab.points3d(x, y, z, color=color2, figure=figure1, scale_factor=SCALE_FACTOR)

        x, y, z = model2xyz(model3)
        mlab.points3d(x, y, z, color=color2, figure=figure2, scale_factor=SCALE_FACTOR)

        mlab.show()


def compare_two_mat():
    import scipy.io as scio
    fns = tkinter.filedialog.askopenfilenames(title='选择文件', filetypes=[('matlab文件', '.mat'), ('所有文件', '.*'), ])
    print(fns)
    if len(fns) == 0: exit(0)
    figure1 = mlab.figure(bgcolor=(1, 1, 1), size=(800, 700))
    figure2 = mlab.figure(bgcolor=(1, 1, 1), size=(800, 700))
    if len(fns) is not 3: return
    loadmat = scio.loadmat(fns[0])
    model1 = np.array(loadmat['pc1'])
    loadmat = scio.loadmat(fns[1])
    model2 = np.array(loadmat['pc2'])
    print(model1.shape)
    mat2 = scio.loadmat(fns[2])
    model3 = np.array(mat2['V'])

    x, y, z = model2xyz(model1)
    color1 = BLUE
    color2 = RED
    mlab.points3d(x, y, z, color=color1, figure=figure1, scale_factor=SCALE_FACTOR)
    mlab.points3d(x, y, z, color=color1, figure=figure2, scale_factor=SCALE_FACTOR)

    x, y, z = model2xyz(model2)
    mlab.points3d(x, y, z, color=color2, figure=figure1, scale_factor=SCALE_FACTOR)
    x, y, z = model2xyz(model3)
    mlab.points3d(x, y, z, color=color2, figure=figure2, scale_factor=SCALE_FACTOR)

    mlab.show()


if __name__ == '__main__':
    file = h5py.File("data/trainset.h5")
    data = file["xyz2"]         # [230000,1024,3]
    i=0

    # while True:
    #     # view_model(data[i])
    #     model_disp(data[i],data[i+1])
    #     i+=1
    while True:
        view_one()
