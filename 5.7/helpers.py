import time
import inspect


def timeit(function):
    def timed(*args, **kwargs):
        ts = time.perf_counter_ns()
        result = function(*args, **kwargs)
        te = time.perf_counter_ns()
        print(f"{function.__name__}: {(te - ts) / 10**6} ms")
        return result

    return timed


def power(x):
    superscipt_dict = {
        "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶",
        "7": "⁷", "8": "⁸", "9": "⁹"}
    power_string = ""
    for s in str(x):
        power_string += superscipt_dict[s]
    return power_string


def chi_squared(f, xs, ys, errors):
    return sum([((y - f(x)) / error)**2 for x, y, error in zip(xs, ys, errors)])


class ParamDict(dict):
    def __repr__(self):
        total_str = ""
        for name, param in self.items():
            total_str += f"{name} = {str(param)}\n"
        return total_str[:-1]


def add_arrow(line, position=None, direction='right', size=15, color=None):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()
    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate('',
                       xytext=(xdata[start_ind], ydata[start_ind]),
                       xy=(xdata[end_ind], ydata[end_ind]),
                       arrowprops=dict(arrowstyle="->", color=color),
                       size=size
                       )

# Once again, this add_arrow functions helps out
# https://stackoverflow.com/questions/34017866/arrow-on-a-line-plot-with-matplotlib


def add_arrow(line, position=None, direction='right', size=15, color=None):


"""
add an arrow to a line.

line:       Line2D object
position:   x-position of the arrow. If None, mean of xdata is taken
direction:  'left' or 'right'
size:       size of the arrow in fontsize points
color:      if None, line color is taken.
"""
if color is None:
    color = line.get_color()

xdata = line.get_xdata()
ydata = line.get_ydata()

if position is None:
    position = xdata.mean()
# find closest index
start_ind = np.argmin(np.absolute(xdata - position))
if direction == 'right':
    end_ind = start_ind + 1
else:
    end_ind = start_ind - 1

line.axes.annotate('',
                   xytext=(xdata[start_ind], ydata[start_ind]),
                   xy=(xdata[end_ind], ydata[end_ind]),
                   arrowprops=dict(arrowstyle="->", color=color),
                   size=size
                   )


def better_curve_fit(f, xdata, ydata, p0=None, sigma=None, **kwargs):
    """
    A modified version of curve_fit which returns more useful information.

    """
    global popt
    popt, pcov = optimize.curve_fit(
        f, xdata, ydata, p0=p0, sigma=sigma, **kwargs)
    argspec = inspect.getfullargspec(f)
    params = ParamDict({param: floats.floatE(val, error)
                        for param, val, error in zip(argspec[0][1:], popt, np.sqrt(np.diag(pcov)))})
    return params, popt
