import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon


def add_gradient_patch(polygon, start, end, color1=None, color2=None, stops=None, ax=None, count=100, debug=False, clip_on=True, **kwargs):
    """
    Adds a gradient patch to a matplotlib plot, filling a polygon with a linear gradient.

    Parameters:
        polygon (array-like): Coordinates of the polygon vertices as a list of (x, y) pairs.
        start (tuple): Starting point (x, y) of the gradient.
        end (tuple): Ending point (x, y) of the gradient.
        color1 (str or tuple, optional): Starting color of the gradient. Defaults to None.
        color2 (str or tuple, optional): Ending color of the gradient. Defaults to None.
        stops (list of tuples, optional): Gradient stops as (color, position) pairs. Defaults to None.
        ax (matplotlib.axes.Axes, optional): Axes to draw the gradient on. Defaults to the current axes.
        count (int, optional): Number of gradient steps. Defaults to 100.
        debug (bool, optional): If True, displays debug information. Defaults to False.
        clip_on (bool, optional): Whether to clip the gradient to the polygon. Defaults to True.
        **kwargs: Additional keyword arguments passed to the Polygon patch.

    Returns:
        None
    """
    import matplotlib as mpl
    import matplotlib.transforms as mtransforms

    # get the axes
    if ax is None:
        ax = plt.gca()

    # a rotation transform with a rotation center
    def transform(points, angle_rad, origin):
        origin = np.asarray(origin)
        rot = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                        [np.sin(angle_rad), np.cos(angle_rad)]])
        return (points - origin) @ rot + origin

    # convert points and start&end to array
    polygon = np.asarray(polygon, dtype=float)
    startend = np.asarray([start, end], dtype=float)

    # the angle of the gradient
    start_end_diff = np.diff(startend, axis=0)[0]
    scale = np.linalg.norm(start_end_diff)
    angle_rad = np.arctan2(start_end_diff[1], start_end_diff[0])

    # rotate points and start&end
    points_rotate = transform(polygon, angle_rad, startend[0])
    startend_rot = transform(startend, angle_rad, startend[0])
    scale_y = np.max(points_rotate[:, 1]) - np.min(points_rotate[:, 1])

    # the offset the gradient image has to be shifted perpendicular to the gradient to cover the while polygon
    diff_rot = np.array([0, -startend_rot[0, 1] + np.min(points_rotate, axis=0)[1]])
    diff = transform(diff_rot, -angle_rad, [0, 0])
    # the parallel distance of the bounding box of the polygon to the start of the gradient (to fill with color1)
    diff_x1 = np.array([-startend_rot[0, 0] + np.min(points_rotate, axis=0)[0], 0])
    # the parallel distance of the bounding box of the polygon to the end of the gradient (to fill with color2)
    diff_x2 = np.array([-startend_rot[1, 0] + np.max(points_rotate, axis=0)[0], 0])

    # shift the gradient points to the edge of the bounding box
    startend += diff
    startend_rot += diff_rot

    # how much to add before the start of the gradient
    if diff_x1[0] < 0:
        offset_start = int(np.ceil(-diff_x1[0]/np.linalg.norm(start_end_diff)*count))
    else:
        offset_start = 0
    # how much to add after the start of the gradient
    if diff_x2[0] > 0:
        offset_end = int(np.ceil(diff_x2[0] / np.linalg.norm(start_end_diff) * count))
    else:
        offset_end = 0

    image = np.zeros((1, offset_start+count+offset_end, 4))
    if stops is None:
        stops = [(color1, 0), (color2, 1)]

    fraction = np.hstack((np.zeros(offset_start), np.linspace(0, 1, count, dtype=float), np.ones(offset_end)))
    for i in range(0, len(stops)):
        color = np.array(mpl.colors.to_rgba(stops[i][0]))[None, None, :]

        def start_end(start, end, invert=False):
            f = (fraction - start) / (end - start)
            if invert:
                f = 1 - f
            f[fraction < start] = 0
            if end < 1:
                f[fraction >= end] = 0
            return f
        if i > 0:
            image += start_end(stops[i-1][1], stops[i][1], False)[None, :, None] * color
        if i < len(stops)-1:
            image += start_end(stops[i][1], stops[i+1][1], True)[None, :, None] * color
    # show the image with interpolation
    im = ax.imshow(image, extent=[0, 1, 0, 1], interpolation="bilinear", aspect='auto')

    # transformed image to cover the whole polygon
    offset = startend[0]-start_end_diff*offset_start/count
    im.set_transform(mtransforms.Affine2D().scale(scale*(1+offset_start/count+offset_end/count), scale_y)
                     + mtransforms.Affine2D().rotate_deg(np.rad2deg(angle_rad))
                     + mtransforms.Affine2D().translate(*offset)
                     + ax.transData)

    # optionally show the start and end of the gradient
    if debug:
        startend -= diff
        plt.plot(startend[:, 0], startend[:, 1], "o--k", mfc="none")

    # generate the polygon and clip the image to it
    patch = Polygon(polygon, transform=ax.transData, facecolor="none", clip_on=clip_on, **kwargs)
    ax.add_patch(patch)
    im.set_clip_path(patch)
    if clip_on is False:
        im.set_clip_box(ax.figure.bbox)


def add_gradient(points, start, end, color1=None, color2=[1, 1, 1, 0], ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    points2 = []
    def trans_point(p):
        if isinstance(p, tuple) and getattr(p[1], "transform"):
            p = p[1].transform(p[0])
            p = ax.transData.inverted().transform(p)
            return p
        return p
    for p in points:
        points2.append(trans_point(p))
    start = trans_point(start)
    end = trans_point(end)

    add_gradient_patch(points2, start, end, color1, color2, ax=ax, clip_on=False, **kwargs)#, debug=True, edgecolor="k")


if __name__ == "__main__":
    np.random.seed(42)
    # add axes
    ax1 = plt.axes()
    ax1.spines[["right", "top"]].set_visible(False)
    plt.xlabel("time")
    plt.ylabel("amplitude")

    # set limits
    plt.xlim(-8, 8)
    plt.ylim(-1.4, 1.2)

    # add gradient triangle to indicate noise
    add_gradient_patch([[-6, -1.3], [6, -1.3], [6, -1.1]], [-6, -1.3], [6, -1.2], stops=[([70 / 255., 170 / 255., 70 / 255.], 0), ("orange", 0.7), ([1, 0, 0], 1)], edgecolor="k")
    plt.text(6.1, -1.2, "noise", ha="left", va="center")

    # add gradient to show start
    add_gradient_patch([[-7.5, -0.3], [-2*np.pi, 0], [-6.5, -0.4]], [-7, -0.35], [-7.0, 0],
                       color1="C0", color2="C1")
    plt.text(-7.5, -0.5, "start", ha="left", va="center")

    # crate and plot "data"
    x = np.linspace(-2*np.pi, 2 * np.pi, 360*2)
    y = np.sin(x)+np.random.normal(0, np.linspace(0, 0.1, 360*2))
    plt.plot(x, y)

    # create inset
    ax2 = plt.axes([0.75, 0.75, 0.2, 0.2])
    plt.xlim(-0.11, 0.11)
    plt.ylim(-0.11, 0.11)
    ax2.spines[["right", "top"]].set_visible(False)

    # plot "data"
    plt.plot(x, y)

    add_gradient([
        ([-0.1, -0.1], ax1.transData),
        ([1, 0], ax2.transAxes),
        ([0, 1], ax2.transAxes),
        ([0.1, 0.1], ax1.transData),
    ], ([0, 0], ax1.transData),
        ([0, 0.5], ax2.transAxes), [1, 1, 1, 0.3], [0.6, 0.6, 0.6], ax=ax1)

    plt.show()
