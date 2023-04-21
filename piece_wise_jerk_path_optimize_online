import math
import numpy
import osqp
from scipy import sparse
from matplotlib import pyplot as plt
from piecewise_jerk_optimize import PieceJerkOptimize
from matplotlib.widgets import Slider

class PieceJerkPathOptimize(PieceJerkOptimize):
    def __init__(self, num_of_point):
        super().__init__(num_of_point)
        self.ref_path_s = []
        self.solution_theta = []
        self.solution_kappa = []
        self.solution_dkappa = []

    def SetReferencePathS(self, ref_path_s):
        self.ref_path_s = ref_path_s

    def CalculateQ(self):
        row = []
        col = []
        data = []

        # l cost
        for i in range(self.num_of_point):
            row.append(i)
            col.append(i)
            data.append(2.0 * self.w_x)

        # dl cost
        for i in range(self.num_of_point):
            row.append(self.num_of_point + i)
            col.append(self.num_of_point + i)
            data.append(2.0 * self.w_dx)

        # ddl cost
        for i in range(self.num_of_point):
            row.append(2 * self.num_of_point + i)
            col.append(2 * self.num_of_point + i)
            data.append(2.0 * self.w_ddx)

        # dddl cost
        for i in range(self.num_of_point - 1):
            row.append(2 * self.num_of_point + i)
            col.append(2 * self.num_of_point + i)
            data.append(data[2 * self.num_of_point + i] + self.w_dddx / self.step_square[i])
            row.append(2 * self.num_of_point + i + 1)
            col.append(2 * self.num_of_point + i + 1)
            data.append(data[2 * self.num_of_point + i + 1] + self.w_dddx / self.step_square[i])
            row.append(2 * self.num_of_point + i + 1)
            col.append(2 * self.num_of_point + i)
            data.append(-2.0 * self.w_dddx / self.step_square[i])

        row.append(2 * self.num_of_point + self.num_of_point - 1)
        col.append(2 * self.num_of_point + self.num_of_point - 1)
        data.append(data[2 * self.num_of_point + self.num_of_point - 1] /
                    + self.w_dddx / self.step_square[self.num_of_point - 2])

        # l_ref cost
        for i in range(self.num_of_point):
            data[i] += 2.0 * self.w_ref_x

        Q = sparse.csc_matrix((data, (row, col)), shape=(
            self.num_of_variable, self.num_of_variable))

        return Q
    def VizResult(self):
        # Create figure and subplots
        fig, axes = plt.subplots(4, 1)
        plt.subplots_adjust(bottom=0.25)

        # Plot the initial data
        axes[0].plot(self.ref_path_s, self.x_upper_bound, 'r', marker="x")
        axes[0].plot(self.ref_path_s, self.x_lower_bound, 'r', marker="x")
        axes[0].plot(self.ref_path_s, self.ref_x, 'g', marker="x")
        axes[0].plot(self.ref_path_s, self.solution_x, 'b')
        axes[0].grid()
        axes[0].legend(["upper_bound", "lower_bound", "ref_path_l", "solution_l"])
        axes[0].set_title("PieceWise Jerk Path Optimization Solution")

        axes[1].plot(self.ref_path_s, self.solution_dx, 'b')
        axes[1].plot(self.ref_path_s, self.dx_upper_bound, 'r')
        axes[1].plot(self.ref_path_s, self.dx_lower_bound, 'r')
        axes[1].grid()
        axes[1].legend(["solution_dl", "dl_bound"])

        axes[2].plot(self.ref_path_s, self.solution_ddx, 'b')
        axes[2].plot(self.ref_path_s, self.ddx_upper_bound, 'r')
        axes[2].plot(self.ref_path_s, self.ddx_lower_bound, 'r')
        axes[2].legend(["solution_ddl", "ddl_bound"])
        axes[2].grid()

        axes[3].plot(self.ref_path_s, self.solution_dddx, 'b')
        axes[3].plot(self.ref_path_s, self.dddx_upper_bound, 'r')
        axes[3].plot(self.ref_path_s, self.dddx_lower_bound, 'r')
        axes[3].legend(["solution_dddl", "dddl_bound"])
        axes[3].grid()

        t1 = plt.text(1.2, 0.4, f"iteration: {self.solution_nums_of_iteration}", transform=plt.gca().transAxes,fontsize=20)
        t2 = plt.text(1.2, 1.4, f"status: {self.solution_status}", transform=plt.gca().transAxes,fontsize=20)

        self.texts.append(t1)
        self.texts.append(t2)

        # Define the update function
        def update(val):
            # Get the slider values
            w_l = slider_w_l.val
            w_dl = slider_w_dl.val
            w_ddl = slider_w_ddl.val
            w_dddl = slider_w_dddl.val
            w_ref_l = slider_w_ref_l.val

            # Recalculate the solution based on the new weights
            self.solution_x = []
            self.solution_dx = []
            self.solution_ddx = []
            self.solution_dddx = []

            self.texts[0].remove()
            self.texts[1].remove()

            self.SetWeight(w_l, w_dl, w_ddl, w_dddl, w_ref_l)
            self.Optimize()
            # Clear the previous plots
            for ax in axes:
                ax.cla()


            # Plot the new data
            axes[0].plot(self.ref_path_s, self.x_upper_bound, 'r', marker="x")
            axes[0].plot(self.ref_path_s, self.x_lower_bound, 'r', marker="x")
            axes[0].plot(self.ref_path_s, self.ref_x, 'g', marker="x")
            axes[0].plot(self.ref_path_s, self.solution_x, 'b')
            axes[0].grid()
            axes[0].legend(["upper_bound", "lower_bound", "ref_path_l", "solution_l"])
            axes[0].set_title("PieceWise Jerk Path Optimization Solution")

            axes[1].plot(self.ref_path_s, self.solution_dx, 'b')
            axes[1].plot(self.ref_path_s, self.dx_upper_bound, 'r')
            axes[1].plot(self.ref_path_s, self.dx_lower_bound, 'r')
            axes[1].grid()
            axes[1].legend(["solution_dl", "dl_bound"])

            axes[2].plot(self.ref_path_s, self.solution_ddx, 'b')
            axes[2].plot(self.ref_path_s, self.ddx_upper_bound, 'r')
            axes[2].plot(self.ref_path_s, self.ddx_lower_bound, 'r')
            axes[2].legend(["solution_ddl", "ddl_bound"])
            axes[2].grid()

            axes[3].plot(self.ref_path_s, self.solution_dddx, 'b')
            axes[3].plot(self.ref_path_s, self.dddx_upper_bound, 'r')
            axes[3].plot(self.ref_path_s, self.dddx_lower_bound, 'r')
            axes[3].legend(["solution_dddl", "dddl_bound"])
            axes[3].grid()

            t1 = plt.text(1.1, 3.4, f"iteration: {self.solution_nums_of_iteration}", transform=plt.gca().transAxes,
                          fontsize=20)
            t2 = plt.text(1.1, 5.4, f"status: {self.solution_status}", transform=plt.gca().transAxes, fontsize=20)

            self.texts.append(t1)
            self.texts.append(t2)

            # Redraw the figure
            fig.canvas.draw_idle()



        # Create the sliders
        slider_w_l = Slider(plt.axes([0.15, 0.20, 0.5, 0.03]), 'w_l', 1, 10, valinit=w_l)
        slider_w_dl = Slider(plt.axes([0.15, 0.15, 0.5, 0.03]), 'w_dl', 1, 100, valinit=w_dl)
        slider_w_ddl = Slider(plt.axes([0.15, 0.10, 0.5, 0.03]), 'w_ddl', 1, 1000, valinit=w_ddl)
        slider_w_dddl = Slider(plt.axes([0.15, 0.05, 0.5, 0.03]), 'w_dddl', 1, 10000,valinit=w_dddl)
        slider_w_ref_l = Slider(plt.axes([0.15, 0.00, 0.5, 0.03]), 'w_refl', 1, 100, valinit=w_ref_l)



        # Connect the sliders to the update function
        slider_w_l.on_changed(update)
        slider_w_dl.on_changed(update)
        slider_w_ddl.on_changed(update)
        slider_w_dddl.on_changed(update)
        slider_w_ref_l.on_changed(update)

        # Show the figure
        plt.show()


if __name__ == "__main__":
    # weight parameter
    w_l = 5
    w_dl = 50
    w_ddl = 500
    w_dddl = 5000
    w_ref_l = 50

    # car paramter
    wheel_base = 2.8
    max_delta = numpy.deg2rad(29.375)
    max_delta_rate = numpy.deg2rad(31.25)

    # Reference path
    step = 0.5
    ref_path_length = 10.0
    ref_path_s = numpy.arange(0, ref_path_length, step)
    if (ref_path_s[-1] != ref_path_length):
        ref_path_s = numpy.append(ref_path_s, ref_path_length)

    ref_path_boundary_upper_l = numpy.array([])
    ref_path_boundary_lower_l = numpy.array([])
    ref_path_kappa = []
    for i in range(len(ref_path_s)):
        ref_path_boundary_upper_l = numpy.append(ref_path_boundary_upper_l, 0.0 + 2.0)
        ref_path_boundary_lower_l = numpy.append(ref_path_boundary_lower_l, 0.0 - 2.0)
        ref_path_kappa.append(0.0)

    ref_path_boundary_upper_l[1] = 3.0
    ref_path_boundary_lower_l[1] = -1.0
    ref_path_boundary_upper_l[5] = 3.0
    ref_path_boundary_lower_l[5] = 0.0
    ref_path_boundary_upper_l[6] = 3.0
    ref_path_boundary_lower_l[6] = 0.0
    ref_path_boundary_upper_l[7] = 3.0
    ref_path_boundary_lower_l[7] = 0.0
    ref_path_boundary_upper_l[8] = 3.0
    ref_path_boundary_lower_l[8] = 0.0
    ref_path_boundary_upper_l[14] = 3.0
    ref_path_boundary_lower_l[14] = -1.0
    ref_path_boundary_upper_l[15] = 3.0
    ref_path_boundary_lower_l[15] = -1.0

    ref_path_size = len(ref_path_s)
    ref_path_s_step = [ref_path_s[i + 1] - ref_path_s[i] for i in range(ref_path_size - 1)]
    ref_path_l = [0.5 * (ref_path_boundary_lower_l[i] + ref_path_boundary_upper_l[i]) for i in range(ref_path_size)]

    init_state = [ref_path_l[0], 0.3, 0.1]
    end_state = [ref_path_l[-1], 0.0, 0.0]

    # bound
    l_lower_bound = ref_path_boundary_lower_l
    l_upper_bound = ref_path_boundary_upper_l

    # dl = (1 - kappa * l) * tan(delta_theta)
    dl_bound = math.tan(numpy.deg2rad(30))
    dl_lower_bound = [-dl_bound for i in range(ref_path_size)]
    dl_upper_bound = [dl_bound for i in range(ref_path_size)]

    # ddl = tan(max_delta)/wheel_base - k_r
    ddl_bound = (math.tan(max_delta) / wheel_base - 0.0)
    ddl_lower_bound = [-ddl_bound for i in range(ref_path_size)]
    ddl_upper_bound = [ddl_bound for i in range(ref_path_size)]

    # dddl
    dddl_bound = max_delta_rate / wheel_base / 2.0
    dddl_lower_bound = [-dddl_bound for i in range(ref_path_size)]
    dddl_upper_bound = [dddl_bound for i in range(ref_path_size)]

    path_optimize = PieceJerkPathOptimize(len(ref_path_s))
    path_optimize.SetWeight(w_l, w_dl, w_ddl, w_dddl, w_ref_l)
    path_optimize.SetReferencePathS(ref_path_s)
    path_optimize.SetReferenceX(ref_path_l)
    path_optimize.SetInitState(init_state)
    path_optimize.SetEndState(end_state)
    path_optimize.SetStep(ref_path_s_step)
    path_optimize.SetXBound(l_upper_bound, l_lower_bound)
    path_optimize.SetDXBound(dl_upper_bound, dl_lower_bound)
    path_optimize.SetDDXBound(ddl_upper_bound, ddl_lower_bound)
    path_optimize.SetDDDXBound(dddl_upper_bound, dddl_lower_bound)
    path_optimize.Optimize()
    path_optimize.VizResult()
