"""Draw the low-precession mechanism schematic (paper Figure 4).

Run from the project root: python Figure_orbital_mechanism.py
Read only the local Natural Earth land outlines. All ocean/ice features and
arrows are conceptual; no statistical analysis or climate model is rerun.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, DrawingArea
from matplotlib.patches import Circle, FancyArrowPatch, PathPatch, Polygon, Rectangle
from matplotlib.path import Path as MplPath
import numpy as np

from paper_figure_export import copy_pdf_to_paper
from toolbox.figure_style import add_panel_label

PROJECT = Path(__file__).resolve().parent
LAND = PROJECT / "data/raw/maps/ne_110m_land.geojson"
OUTPUT = PROJECT / "figures/Figure_orbital_mechanism/Figure_orbital_mechanism"
# Gray land, blue ocean and cyan ice follow the visual idea of Zhang et al.
# (2017), Figure 3. The low-precession arrows have separate source support.
COLOR = dict(ink="#152C45", quiet="#566778", land="#B8BBBF", coast="#9AA6AF",
             ocean="#B3E1F4", water="#B3E1F4", deep="#75BBD8", blue="#5066B8",
             red="#D84858", warm="#F4D293", gold="#EDA927", ice="#FFFFFF",
             red_ink="#A32639", blue_ink="#304890", export="#008769")
MAP_CENTER = (-35, 30)  # Central longitude and latitude of the orthographic globe.
FONT_SCALE = 1.25
SUN_RADIUS_PT = 9

# ICE SHEET EDGES: all pairs are (longitude, latitude), in degrees; west is negative.
# Walk around each ice-sheet edge in order. The final point closes automatically.
# These are illustrative controls based on Zhang (2017) Fig. 3a.

# ICE_SHEETS = {
#     "North America": {
#         "edge": [(-124,63), (-115,70), (-102,74), (-89,79), (-75,79),
#                  (-64,73), (-62,65), (-57,58), (-61,51), (-72,45),
#                  (-84,41), (-96,43), (-107,49), (-113,56)],
#         "center": (-94,58),
#     },
#     "Greenland": {
#         "edge": [(-73,78), (-64,82), (-48,83), (-30,82), (-19,79),
#                  (-19,75), (-23,71), (-29,68), (-36,65), (-43,60),
#                  (-49,64), (-52,69), (-55,74), (-64,76)],
#         "center": (-43,72),
#     },
#     "Eurasia": {
#         "edge": [(1,57), (2,65), (10,72), (18,76), (18,80),
#                  (36,82), (55,80), (67,75), (68,70), (57,66),
#                  (45,63), (35,56), (22,52), (11,53)],
#         "center": (24,65),
#     },
# }


ICE_SHEETS = {
    "North America": {
        "edge": [(-126.1743, 74.2406),
(-104.8819, 75.3966),
(-78.0283, 74.0733),
(-69.2346, 71.2180),
(-63.5468, 68.4096),
(-61.6059, 66.3565),
(-63.2192, 63.0159),
(-62.7283, 60.6886),
(-60.8681, 57.9747),
(-58.1223, 55.5978),
(-54.8015, 53.2603),
(-53.8273, 50.6882),
(-52.8498, 47.1498),
(-56.7818, 46.9869),
(-61.2787, 44.9884),
(-66.7957, 43.7700),
(-71.3891, 41.8468),
(-74.0631, 40.1467),
(-77.2497, 40.7823),
(-81.5802, 41.4154),
(-85.8850, 41.9569),
(-90.4715, 42.5343),
(-94.7997, 43.0629),
(-100.2308, 43.5823),
(-103.3618, 44.5781),
(-108.4506, 44.8200),
(-112.7358, 44.4464),
(-117.2501, 44.3538),
(-122.3226, 43.9686),
(-124.7101, 43.9349),
(-126.1863, 47.9544),
(-128.7859, 50.4018),
(-129.9731, 50.9873),
(-131.0159, 52.8222),
(-135.2427, 55.7788),
(-142.0552, 58.4708),
(-150.8332, 59.0347),
(-162.9343, 55.9697),
(-166.5464, 57.2233),
(-168.1157, 58.3338),
(-167.5961, 61.0568),
(-168.4189, 65.4292),
(-165.9775, 68.9543),
(-156.2456, 71.7896),
(-136.7970, 73.9983),
],
        "center": (-96.4642, 58.3795),
    },
    "Greenland": {
        "edge": [(-124.7481, 74.8420),
(-119.7956, 78.2216),
(-81.4729, 83.5044),
(-38.0511, 83.9572),
(-8.8005, 82.2619),
(-11.1556, 79.1859),
(-16.6007, 74.6966),
(-20.6342, 71.1043),
(-29.2665, 68.0142),
(-36.0239, 65.5439),
(-38.8683, 63.6003),
(-40.6177, 61.1992),
(-41.6996, 59.1586),
(-48.1847, 60.9778),
(-54.1487, 64.1332),
(-55.4130, 66.7042),
(-55.3968, 70.2611),
(-61.6875, 75.3741),
(-70.7814, 76.9383),
(-75.9139, 75.1341),
(-71.6748, 73.5810),
(-65.5330, 70.0610),
(-60.7552, 67.2534),
(-61.6845, 63.2534),
(-62.8749, 61.2475),
(-93.1154, 69.5584),
(-124.0416, 71.6234)],
        "center": (-43,72),
    },
    "Eurasia": {
        "edge": [(106.1615, 77.5437),
(99.0170, 80.7635),
(77.2501, 82.1359),
(55.1369, 81.3092),
(44.5811, 80.8744),
(19.9286, 80.8059),
(12.2900, 79.0997),
(14.4377, 76.7222),
(18.3116, 74.0983),
(20.2507, 71.7055),
(16.3013, 70.5858),
(13.4627, 68.1159),
(5.7080, 64.9986),
(-0.0828, 62.8288),
(-4.3379, 60.1975),
(-8.0468, 57.4266),
(-9.9758, 54.1274),
(-10.6722, 52.1987),
(-7.3350, 50.6574),
(-1.4969, 50.7114),
(3.1198, 52.2096),
(10.3293, 53.7932),
(17.8625, 54.9738),
(26.5700, 58.2011),
(36.2946, 61.9785),
(45.2944, 63.2183),
(62.7472, 65.6642),
(75.0298, 65.6185),
(81.4744, 68.0156),
(88.9487, 70.4015),
(98.9254, 73.9155)],
        "center": (24,65),
    },
    "Iceland": {
        "edge": [(-18.9302, 67.2170), (-24.8500, 67.1461),
                 (-24.8897, 65.4457), (-20.6521, 63.4242),
                 (-13.3487, 64.1031), (-14.9872, 67.3979)],
    },
}






# Independent, simple polygons for the second color layer, in (lon, lat).
# The two lighter layers shrink this polygon toward its own center, not the
# outer ice margin. These are schematic shading, not elevation contours.
# Iceland is deliberately omitted: its small footprint has only one color.
ICE_DOME_POLYGONS = {
    "North America": {
        "edge": [(-108, 64), (-98, 69), (-83, 69), (-72, 64),
                 (-67, 56), (-73, 48), (-87, 47), (-99, 52), (-107, 58)],
        "center": (-86, 58),
    },
    "Greenland": {
        "edge": [(-53, 77), (-42, 80), (-26, 78), (-26, 73),
                 (-34, 68), (-42, 63), (-49, 66), (-51, 72)],
        "center": (-41, 72),
    },
    "Eurasia": {
        "edge": [(8, 61), (17, 68), (29, 72), (40, 78),
                 (61, 78), (78, 73), (65, 69), (43, 67), (27, 61), (17, 58)],
        "center": (37, 69),
    },
}

# Sea ice fills the OCEAN north of this southern margin, including the Arctic.
# Keep longitude increasing from -180 to 180; the first/last latitudes match.
# Add/move pairs here to reshape the Atlantic tongue or the Arctic margin.
SEA_ICE_SOUTHERN_EDGE = [
    (-180,72), (-140,72), (-110,68), (-85,59), (-65,50), (-50,47),
    (-35,48), (-20,51), (-6.8,50), (10,61), (30,69), (60,74),
    (100,75), (140,74), (180,72),
]

# 0 draws the exact control polygon; 1 rounds corners without overshooting them.
ICE_SMOOTHING = 1
# True adds numbered vertices and a latitude/longitude grid, saves a separate
# ice_controls_preview, and DOES NOT overwrite the paper figure.
SHOW_ICE_CONTROLS = False


def text(ax, x, y, label, size=8, color=None, ha="center", **kwargs):
    # Darker versions of the accent colors keep small labels legible on water.
    if color == COLOR["red"]:
        color = COLOR["red_ink"]
    elif color == COLOR["blue"]:
        color = COLOR["blue_ink"]
    return ax.text(x, y, label, fontsize=size*FONT_SCALE, color=color or COLOR["ink"],
                   ha=ha, va="center", linespacing=1.25, zorder=8, **kwargs)


def arrow(ax, start, end, color, width=1.6, curve=0, dashed=False, **kwargs):
    patch = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10,
                            connectionstyle=f"arc3,rad={curve}", linewidth=width,
                            color=color, linestyle=(0, (3, 2)) if dashed else "-",
                            zorder=6, **kwargs)
    ax.add_patch(patch)
    return patch


def sun(ax, x, y, radius=5):
    # DrawingArea uses physical points: the sun stays round on any map/section.
    drawing = DrawingArea(4*radius, 4*radius, 0, 0)
    center = 2*radius
    drawing.add_artist(Circle((center, center), radius,
                              facecolor=COLOR["gold"], edgecolor="none"))
    for angle in np.linspace(0, 2*np.pi, 10, endpoint=False):
        r = radius * np.array([1.35, 1.85])
        drawing.add_artist(Line2D(center + r*np.cos(angle), center + r*np.sin(angle),
                                  color=COLOR["gold"], lw=0.9))
    ax.add_artist(AnnotationBbox(drawing, (x, y), frameon=False,
                                 box_alignment=(0.5, 0.5), zorder=7))


def project(lon, lat):
    """Orthographic coordinates on the unit sphere, centered on the Atlantic."""
    lon0, lat0 = np.deg2rad(MAP_CENTER)
    lon, lat = np.deg2rad(lon), np.deg2rad(lat)
    x = np.cos(lat)*np.sin(lon-lon0)
    y = np.cos(lat0)*np.sin(lat) - np.sin(lat0)*np.cos(lat)*np.cos(lon-lon0)
    return x, y


def draw_geography(ax, globe):
    # Inverse projection + a land mask handles the circular horizon without
    # drawing polygons from the far side of Earth. Contours remain vector paths.
    # Coastlines: Natural Earth v5.1.2, 1:110m; modern geography for orientation.
    coord = np.linspace(-1, 1, 900)
    x, y = np.meshgrid(coord, coord)
    visible = x*x + y*y <= 1
    z = np.sqrt(np.maximum(0, 1-x*x-y*y))
    lon0, lat0 = np.deg2rad(MAP_CENTER)
    lat = np.arcsin(y*np.cos(lat0) + z*np.sin(lat0))
    lon = lon0 + np.arctan2(x, z*np.cos(lat0)-y*np.sin(lat0))
    lon = (np.rad2deg(lon)+180) % 360-180
    lat = np.rad2deg(lat)
    points = np.column_stack([lon[visible], lat[visible]])
    is_land = np.zeros(len(points), dtype=bool)
    for feature in json.loads(LAND.read_text())["features"]:
        geometry = feature["geometry"]
        polygons = ([geometry["coordinates"]] if geometry["type"] == "Polygon"
                    else geometry["coordinates"])
        for rings in polygons:
            outline = np.asarray(rings[0])
            inside_bbox = ((points[:, 0] >= outline[:, 0].min()) &
                           (points[:, 0] <= outline[:, 0].max()) &
                           (points[:, 1] >= outline[:, 1].min()) &
                           (points[:, 1] <= outline[:, 1].max()))
            inside = MplPath(outline).contains_points(points[inside_bbox])
            for hole in rings[1:]:
                inside &= ~MplPath(hole).contains_points(points[inside_bbox])
            is_land[inside_bbox] |= inside
    land = np.zeros_like(x)
    land[visible] = is_land
    land[~visible] = np.nan
    # A complete polar cap avoids the artificial northern edge of a small
    # closed lon/lat polygon. Ice is restricted to ocean before contouring.
    edge = np.asarray(SEA_ICE_SOUTHERN_EDGE)
    if not (np.all(np.diff(edge[:,0]) > 0) and edge[0,0] == -180 and edge[-1,0] == 180):
        raise ValueError("Sea-ice margin longitudes must increase from -180 to 180.")
    if edge[0,1] != edge[-1,1]:
        raise ValueError("Sea-ice margin latitudes at -180 and 180 must match.")
    ice = ((lat >= np.interp(lon, edge[:, 0], edge[:, 1])) & (land == 0)).astype(float)
    ice[~visible] = np.nan
    sea_ice = ax.contourf(x, y, ice, levels=[0.5,1.5], colors=[COLOR["ice"]], zorder=1)
    sea_ice.set_clip_path(globe)
    fill = ax.contourf(x, y, land, levels=[0.5, 1.5], colors=[COLOR["land"]], zorder=3)
    coast = ax.contour(x, y, land, levels=[0.5], colors=[COLOR["coast"]], linewidths=0.25, zorder=3)
    fill.set_clip_path(globe)
    coast.set_clip_path(globe)


def map_polygon(ax, lonlat, smooth=False, **kwargs):
    points = np.asarray(lonlat)
    if smooth:
        # Corner cutting stays inside the control polygon's convex hull,
        # unlike cubic interpolation, which can push the margin outward.
        for _ in range(ICE_SMOOTHING):
            following = np.roll(points, -1, axis=0)
            points = np.stack([.75*points+.25*following,
                               .25*points+.75*following], axis=1).reshape(-1,2)
    lon, lat = points.T
    patch = Polygon(np.column_stack(project(lon, lat)), closed=True, **kwargs)
    ax.add_patch(patch)
    return patch


def draw_ice_sheets(ax):
    # Draw every outer footprint first, so a neighboring sheet's base cannot
    # cover a dome. The user-edited edges alone determine the ice-covered area.
    boundaries = {}
    for name, item in ICE_SHEETS.items():
        boundaries[name] = map_polygon(ax, item["edge"], smooth=True,
                                      facecolor="#06B5DE", edgecolor="none", zorder=4)

    for name, dome in ICE_DOME_POLYGONS.items():
        outline, center = np.asarray(dome["edge"]), np.asarray(dome["center"])
        for scale, color in zip((1, .68, .36), ("#37CEE9", "#79E2EF", "#BFF5F6")):
            layer = map_polygon(ax, center + scale*(outline-center), smooth=True,
                                facecolor=color, edgecolor="none", zorder=4.1)
            layer.set_clip_path(boundaries[name])


def show_ice_controls(ax):
    """Overlay an editing guide; marker numbers index the lists above."""
    lon0, lat0 = np.deg2rad(MAP_CENTER)
    def grid_line(lon, lat):
        x, y = project(lon, lat)
        lam, phi = np.deg2rad(lon), np.deg2rad(lat)
        front = np.sin(lat0)*np.sin(phi) + np.cos(lat0)*np.cos(phi)*np.cos(lam-lon0) >= 0
        ax.plot(np.where(front,x,np.nan), np.where(front,y,np.nan),
                color="0.3", lw=.45, ls=":", zorder=9)
    for lon in (-120,-90,-60,-30,0,30,60):
        grid_line(np.full(181,lon), np.linspace(-90,90,181))
        ax.text(*project(lon,35), f"{lon}°", fontsize=6, zorder=11)
    for lat in (0,30,45,60,75):
        grid_line(np.linspace(-180,180,361), np.full(361,lat))
        ax.text(*project(-25,lat), f"{lat}°N", fontsize=6, zorder=11)
    controls = [(name[0], item["edge"]) for name,item in ICE_SHEETS.items()]
    controls.append(("S",SEA_ICE_SOUTHERN_EDGE))
    for prefix, points in controls:
        points = np.asarray(points)
        if prefix == "S":
            points = points[:-1]  # -180 and 180 represent the same margin point.
        x,y = project(points[:,0],points[:,1])
        ax.scatter(x,y,s=7,c="black",zorder=10)
        for i,(px,py) in enumerate(zip(x,y)):
            ax.annotate(f"{prefix}{i}",(px,py),xytext=(2,2),textcoords="offset points",
                        fontsize=6,color="black",zorder=11)


def draw_atlantic(ax):
    ax.set(xlim=(-1.23, 1.02), ylim=(-1.04, 1.17), aspect="equal")
    ax.axis("off")
    globe = Circle((0, 0), 1, facecolor=COLOR["ocean"], edgecolor=COLOR["ink"], lw=.6)
    ax.add_patch(globe)

    angle = np.linspace(0, 2*np.pi, 100)
    pool = np.column_stack([-58+27*np.cos(angle), 19+13*np.sin(angle)])
    map_polygon(ax, pool, facecolor=COLOR["warm"], edgecolor="none", alpha=.8, zorder=2)
    draw_geography(ax, globe)
    draw_ice_sheets(ax)
    if SHOW_ICE_CONTROLS:
        show_ice_controls(ax)
        return

    # Location labels sit on their regions; no label leaders are needed.
    text(ax, -.39, .67, "Ice sheets", size=7.3)
    text(ax, .11, .50, "Sea ice\nthickness ↓", size=7.1, color=COLOR["red"])
    text(ax, .02, .27, "Stratification ↑\nAMOC ↓ initially", size=7.2)
    text(ax, -.40, -.075, "Warm-pool\ntemperature ↑", size=7.4, color=COLOR["red"])
    text(ax, -.14, -.285, "Surface\nsalinity ↓", size=7.3, color=COLOR["blue"])
    text(ax, .36, -.01, "Fresher water\ncarried north", size=7.1, color=COLOR["blue"])

    # Freshwater transport, not stronger AMOC. The green arrow is atmospheric
    # moisture export, which decreases in the low-precession experiment.
    arrow(ax, project(-42,22), project(-27,41), COLOR["blue"], width=2.8, curve=-.2)
    arrow(ax, project(-82,12), project(-100,6), COLOR["export"], width=1.8)
    text(ax, -.70, -.36, "Moisture\nexport ↓\nto Pacific", size=7.0, color=COLOR["export"])

    # Solar forcing is outside the globe and reaches both latitude bands.
    sun(ax, -1.09, .39, radius=SUN_RADIUS_PT)
    text(ax, -.99, .78, "Low precession\nNH summer\ninsolation ↑", size=7.3)
    arrow(ax, (-.95,.45), (-.07,.50), COLOR["gold"], width=1.55, curve=-.13)
    arrow(ax, (-1.0,.26), (-.52,.06), COLOR["gold"], width=1.55, curve=.1)
    text(ax, 0, 1.10, "Stadial", size=8.4, color="black")


def draw_section(ax):
    ax.set(xlim=(0, 100), ylim=(0, 100))
    ax.axis("off")
    # One conceptual section. Ice-covered and open-water processes are
    # state dependent, not a reconstruction of one simultaneous ocean state.
    ax.add_patch(Rectangle((3, 15), 94, 45, facecolor=COLOR["water"], edgecolor="none"))
    ax.add_patch(Rectangle((3, 15), 94, 14, facecolor=COLOR["deep"], edgecolor="none", alpha=0.35))
    ax.plot([3, 97], [60, 60], color=COLOR["blue"], lw=0.9, zorder=3)

    ax.plot([3, 3, 97, 97], [60, 15, 15, 60], color=COLOR["deep"], lw=0.7)
    # A warm reservoir does not imply a positive orbital temperature anomaly.
    heat = MplPath([(3,19), (3,28), (30,34), (61,21), (97,26),
                    (97,18), (62,14), (30,28), (3,19)],
                   [MplPath.MOVETO, MplPath.LINETO] + [MplPath.CURVE4]*3
                   + [MplPath.LINETO] + [MplPath.CURVE4]*3)
    ax.add_patch(PathPatch(heat, facecolor=COLOR["warm"], edgecolor="none", zorder=2))
    text(ax, 49, 23, "Subsurface heat reservoir", size=7.4)
    # ax.plot([4, 38], [49, 49], color=COLOR["deep"], lw=0.7, zorder=3)
    # text(ax, 23, 54.5, "Cold surface\nlayer", size=7)

    # Summer sea ice: strong seasonality can thin it (Kuniyoshi et al., 2022).
    for left, right, top in [(3,18,64), (20,34,63), (36,45,62.5)]:
        ax.add_patch(Polygon([(left,59.3),(right,59.5),(right-1,top),
                              (left+3,top+0.8),(left,top)],
                             facecolor=COLOR["ice"], edgecolor=COLOR["deep"], lw=0.75, zorder=4))
    sun(ax, 9, 92, radius=SUN_RADIUS_PT)
    arrow(ax, (10,87), (16,67), COLOR["gold"], width=2.0)
    text(ax, 57, 96, "Summer insolation ↑", size=7.4, color=COLOR["red"])
    text(ax, 36, 81, "Summer melt ↑", size=7.5, color=COLOR["red"])
    text(ax, 33, 69.5, "Sea-ice\nthickness ↓", size=7.5)

    # Winter heat loss is shown over open water, not through the ice slab.
    for x in (68, 81, 93):
        arrow(ax, (x,61), (x,77), COLOR["red"], width=1.65, curve=0.12)
    text(ax, 78, 84, "Winter heat\nloss ↑", size=7.5, color=COLOR["red"])
    text(ax, 78, 55.2, "Open water\n(warm state)", size=6.8, color=COLOR["quiet"])
    arrow(ax, (93,49), (93,34), COLOR["blue"], width=1.4, curve=-0.12)
    text(ax, 71, 44, "Faster cooling", size=7.5, color=COLOR["blue"])
    text(ax, 71, 35.5, "Shorter warm\ninterval possible", size=6.8, color=COLOR["blue"])

    # Fohlmeister et al. (2023): a proposed lower temperature threshold,
    # not faster subsurface heat accumulation. Dashed to distinguish this link.
    arrow(ax, (48,30), (48,58.5), COLOR["red"], width=1.5, curve=-0.04, dashed=True)
    text(ax, 22, 49, "Sea ice Retreat\nThreshold ↓", size=7.2, color=COLOR["red"])
    text(ax, 22, 37, "Earlier warming\npossible", size=6.8, color=COLOR["red"])


def plot_mechanism():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    if SHOW_ICE_CONTROLS:
        # A larger, uncluttered map makes individual control points inspectable.
        fig = plt.figure(figsize=(8,8), facecolor="white")
        atlantic = fig.add_axes([.04,.04,.92,.92])
        draw_atlantic(atlantic)
        atlantic.set(xlim=(-1.03,1.03), ylim=(-1.03,1.03))
        return fig
    fig = plt.figure(figsize=(190/25.4, 112/25.4), facecolor="white")
    atlantic = fig.add_axes([0.018, 0.06, 0.59, 0.89])
    section = fig.add_axes([0.625, 0.13, 0.37, 0.72])
    draw_atlantic(atlantic)
    draw_section(section)
    add_panel_label(atlantic, "a", x=0, y=1.005)
    add_panel_label(section, "b", x=0, y=1.145)
    return fig


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_mechanism()
    output = OUTPUT.with_name("ice_controls_preview") if SHOW_ICE_CONTROLS else OUTPUT
    for suffix in (".pdf", ".svg", ".png"):
        fig.savefig(output.with_suffix(suffix), dpi=450, facecolor="white")
    plt.close(fig)
    if not SHOW_ICE_CONTROLS:
        copy_pdf_to_paper(output.with_suffix(".pdf"))
    print(output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
