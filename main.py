#!/usr/bin/python3
import numpy as np
from sympy import Symbol
import sympy as sp

from link import Link, Robot
from se3 import SE3, set_simplify, rot_x, rot_y, rot_z, trans, inv

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider


def main():
    # NED frame, link1 furthest north
    l1 = Link(1.0, 1.0, np.eye(3), np.zeros((6, 6)),
              np.zeros((6, 6)), np.zeros((6, 6)))
    l2 = Link(1.0, 1.0, np.eye(3), np.zeros((6, 6)),
              np.zeros((6, 6)), np.zeros((6, 6)))  # body
    l3 = Link(1.0, 1.0, np.eye(3), np.zeros((6, 6)),
              np.zeros((6, 6)), np.zeros((6, 6)))

    set_simplify(True)

    xn, yn, zn = sp.symbols('xn yn zn', real=True)
    phi, theta, psi = sp.symbols('phi theta psi', real=True)
    a1, a2, a3, a4 = sp.symbols('a1 a2 a3 a4', real=True)

    dxn, dyn, dzn = sp.symbols('dxn dyn dzn', real=True)
    dphi, dtheta, dpsi = sp.symbols('dphi dtheta dpsi', real=True)
    da1, da2, da3, da4 = sp.symbols('da1 da2 da3 da4', real=True)

    a = SE3()
    Tl2n = trans(xn, yn, zn) @ rot_z(psi) @ rot_y(theta) @ rot_x(phi)
    Tl1l2 = trans(1, 0, 0) @ rot_y(a3) @ rot_z(a4) @ trans(1, 0, 0)
    Tl3l2 = inv(trans(1, 0, 0) @ rot_y(a1) @ rot_z(a2) @ trans(1, 0, 0))

    Tl1n = Tl2n @ Tl1l2
    Tl3n = Tl2n @ Tl3l2

    eely = Robot(links=[l1, l2, l3],
                 transforms=[Tl1n, Tl2n, Tl3n],
                 params=[xn, yn, zn, phi, theta, psi, a1, a2, a3, a4],
                 diff_params=[dxn, dyn, dzn, dphi,
                              dtheta, dpsi, da1, da2, da3, da4]
                 )

    model = eely.get_model(simplify=False)
    print(f"{model.M=}")
    print(f"{model.C=}")
    print(f"{model.D=}")
    print(f"{model.g=}")

    exit()
    """
    """

    t = Robot(links=[l1],
              transforms=[Tl2n],
              params=[xn, yn, zn, phi, theta, psi],
              diff_params=[dxn, dyn, dzn, dphi, dtheta, dpsi],
              )

    print("started model computation")
    model = t.get_model()
    print(f"{model.M=}")
    print(f"{model.C=}")
    print(f"{model.D=}")
    print(f"{model.g=}")
    exit()

    print(model)

    set_simplify(False)

    v = np.array([0, 0, 0])
    subs = {xn: 0, yn: 0, zn: 0, phi: 0, theta: 0,
            psi: 0, a1: sp.pi/3, a2: 0, a3: sp.pi/3, a4: 0}
    p1 = np.array(Tl1n.apply(v).evalf(subs=subs)).astype(np.float64)
    p2 = np.array(Tl2n.apply(v).evalf(subs=subs)).astype(np.float64)
    p3 = np.array(Tl3n.apply(v).evalf(subs=subs)).astype(np.float64)

    # Create a new 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    fig.subplots_adjust(bottom=0.25)

    # Set axis labels to NED
    ax.set_xlabel('east')
    ax.set_ylabel('north')
    ax.set_zlabel('negative down')

    # Invert the z-axis to make it point downwards
    # ax.set_zlim(10, 0)  # Set limits to make positive values point downward

    # Example data to plot (replace with your actual data)

    # Plot data in NED frame
    #ax.plot(p1[0], p1[1], p1[2], marker='o', color='r')
    #ax.plot(p2[0], p2[1], p2[2], marker='o', color='g')
    #ax.plot(p3[0], p3[1], p3[2], marker='o', color='b')
    ned_to_xyz = np.array([[0, 1, 0],
                           [1, 0, 0],
                           [0, 0, -1]])
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-1, 1])

    p1_line, = ax.plot(p1[0], p1[1], p1[2], marker='o', color='r')
    p1_arrows = ax.quiver(np.zeros(3), np.zeros(3), np.zeros(
        3), np.zeros(3), np.zeros(3), np.zeros(3), color='r')
    p2_line, = ax.plot(p2[0], p2[1], p2[2], marker='o', color='g')
    p3_line, = ax.plot(p3[0], p3[1], p3[2], marker='o', color='b')
    # ax.set_ylim(ax.get_xlim()[::-1])

    # Define the slider update function
    def update(val):
        # Update the substitution dictionary with the slider values
        subs[phi] = slider_phi.val
        subs[theta] = slider_theta.val
        subs[psi] = slider_psi.val

        vo = np.array([0, 0, 0])
        v1 = np.array([1, 0, 0])
        v2 = np.array([0, 1, 0])
        v3 = np.array([0, 0, 1])
        # Recompute the points based on the new phi, theta, psi values
        p1 = np.array(Tl1n.apply(vo).evalf(subs=subs)).astype(np.float64)
        p11 = np.array(Tl1n.apply(v1).evalf(subs=subs)).astype(np.float64)
        p12 = np.array(Tl1n.apply(v2).evalf(subs=subs)).astype(np.float64)
        p13 = np.array(Tl1n.apply(v3).evalf(subs=subs)).astype(np.float64)

        p2 = np.array(Tl2n.apply(vo).evalf(subs=subs)).astype(np.float64)
        p3 = np.array(Tl3n.apply(vo).evalf(subs=subs)).astype(np.float64)

        p1 = ned_to_xyz @ p1
        p2 = ned_to_xyz @ p2
        p3 = ned_to_xyz @ p3

        # Update the plot data
        p1_line.set_data(p1[0], p1[1])
        p1_line.set_3d_properties(p1[2])
        p2_line.set_data(p2[0], p2[1])
        p2_line.set_3d_properties(p2[2])
        p3_line.set_data(p3[0], p3[1])
        p3_line.set_3d_properties(p3[2])

        # Redraw the figure
        fig.canvas.draw_idle()

    # Create sliders for roll (phi), pitch (theta), and yaw (psi)
    ax_phi = plt.axes([0.1, 0.01, 0.65, 0.03],
                      facecolor='lightgoldenrodyellow')
    ax_theta = plt.axes([0.1, 0.06, 0.65, 0.03],
                        facecolor='lightgoldenrodyellow')
    ax_psi = plt.axes([0.1, 0.11, 0.65, 0.03],
                      facecolor='lightgoldenrodyellow')
    #ax_phi = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    #ax_theta = fig.add_axes([0.1, 0.06, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    #ax_psi = fig.add_axes([0.1, 0.11, 0.65, 0.03], facecolor='lightgoldenrodyellow')

    slider_phi = Slider(ax_phi, 'Roll (phi)', -np.pi, np.pi, valinit=0)
    slider_theta = Slider(ax_theta, 'Pitch (theta)', -
                          np.pi/2, np.pi/2, valinit=0)
    slider_psi = Slider(ax_psi, 'Yaw (psi)', -np.pi, np.pi, valinit=0)

    # Register the update function to the sliders
    slider_phi.on_changed(update)
    slider_theta.on_changed(update)
    slider_psi.on_changed(update)
    update(None)

    # Show the plot
    plt.show()
    # Show the plot
    # plt.show()

    """
    r = Robot([l1, l2, l3], [])

    M, C, D, g = r.mathematical_model()

    M * ddq + C(q, dq) + D(q, dq) + g(q) = M * r + C(q, dq) + D(q, dq) + g(q)

    => ddq = r

    r.to_sdf() # gazebo
    """


if __name__ == "__main__":
    main()
