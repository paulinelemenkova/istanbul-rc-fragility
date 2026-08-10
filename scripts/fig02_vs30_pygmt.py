#!/usr/bin/env python3
"""
fig02_vs30_site_class — Istanbul metropolitan area (PyGMT).

Produces a two-panel figure:
  (a) Vs30 (m/s) over shaded relief, with AFAD station Vs30 measurements overlaid.
  (b) EC8/NEHRP site classes derived from the Vs30 grid.

INPUTS you provide (do NOT fabricate station data):
  VS30_GRID  : USGS global slope-based Vs30 grid, clipped to the region, as .grd/.nc
               e.g.  gmt grdcut global_vs30.grd -R28/30/40.5/41.5 -Gvs30_istanbul.grd
  AFAD_CSV   : whitespace/comma file with columns: lon  lat  vs30   (AFAD measurements)
Requires GMT 6+ and PyGMT (pip install pygmt; GMT must be installed separately).
"""
import os
import pygmt

REGION = [28.0, 30.0, 40.5, 41.5]          # Istanbul metropolitan window
PROJ   = "M15c"
VS30_GRID = "vs30_istanbul.grd"            # <-- your clipped USGS Vs30 grid
AFAD_CSV  = "afad_vs30.csv"                # <-- your AFAD stations (lon,lat,vs30)
OUT = "fig02_vs30_site_class"

# shaded relief from the Vs30 grid's own relief is wrong; use a DEM for shading:
DEM = "istanbul_dem.grd"                   # e.g. gmt grdcut gebco.nc -R28/30/40.5/41.5 -Gistanbul_dem.grd
shade = pygmt.grdgradient(grid=DEM, radiance=[315, 30], normalize="t0.8") if os.path.exists(DEM) else None

fig = pygmt.Figure()
pygmt.config(FONT_TITLE="12p", FONT_ANNOT_PRIMARY="8p", MAP_FRAME_TYPE="plain")

with fig.subplot(nrows=2, ncols=1, figsize=("15c", "19c"), margins="0.4c",
                 title="Site characterisation of the Istanbul metropolitan area"):

    # ---- panel (a): Vs30 + AFAD stations ----
    with fig.set_panel(panel=0):
        pygmt.makecpt(cmap="jet", series=[180, 760, 20], reverse=True, continuous=True)
        fig.grdimage(grid=VS30_GRID, region=REGION, projection=PROJ,
                     shading=shade, frame=["xa0.5f10m", "ya0.25f5m", "WSne",
                                           "+t(a) V@-S30@- and AFAD stations"])
        fig.coast(region=REGION, projection=PROJ, shorelines="0.5p,black",
                  water="173/210/235")
        if os.path.exists(AFAD_CSV):
            fig.plot(data=AFAD_CSV, incols=[0, 1, 2], style="c0.20c",
                     cmap=True, pen="0.4p,black")  # color = Vs30 column
        fig.colorbar(frame='x+l"V@-S30@- (m s@+-1@+)"', position="JMR+o0.6c/0c+w8c")
        fig.basemap(map_scale="jBR+w50k+o0.7c/0.7c+f", region=REGION, projection=PROJ)

    # ---- panel (b): EC8/NEHRP site classes ----
    with fig.set_panel(panel=1):
        # categorical CPT at the class thresholds 180/360/760
        pygmt.makecpt(cmap="215/25/28,253/174/97,43/131/186",
                      series=[180, 360, 760, 900], color_model="+cD,C,B")
        fig.grdimage(grid=VS30_GRID, region=REGION, projection=PROJ, shading=shade,
                     frame=["xa0.5f10m", "ya0.25f5m", "WSne", "+t(b) EC8/NEHRP site classes"])
        fig.coast(region=REGION, projection=PROJ, shorelines="0.5p,black",
                  water="173/210/235")
        fig.colorbar(frame='x+l"Site class"', position="JMR+o0.6c/0c+w8c")

fig.savefig(OUT + ".pdf")
fig.savefig(OUT + ".png", dpi=300)
print("wrote", OUT + ".pdf/.png")
