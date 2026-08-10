#!/bin/sh
# Purpose: shaded-relief topographic + seismicity map of the Sea of Marmara / Istanbul
#          region from the GEBCO 2026 15 arc-sec grid + EarthScope IEB earthquake catalogue.
# Adapted from the author's GMT-01-JM-GEBCO-JT.sh (relief) and GMT-34-MX-seis.sh (seismicity).
# GMT modules: gmtset, grdconvert, makecpt, grdimage, psscale, grdcontour, pscoast,
#              psbasemap, psxy, pstext, pslegend, gmtlogo, psconvert
# Required input files (place in the working directory):
#   gebco_2026_n42_0_s39_0_w25_0_e31_0_geotiff.tif   GEBCO 2026 DEM (GeoTIFF)
#   marmara_seis.txt   IEB catalogue, columns:  lon  lat  depth  mag  size_cm
#   naf_trace.txt      indicative North Anatolian (Main Marmara) Fault, columns: lon lat
# ---------------------------------------------------------------------------
ps=fig01_study_area.ps
R=-R25/31/39/42          # study-area window (Marmara Sea)
J=-JM16c                 # Mercator, 16 cm wide

# Step-1. GMT defaults (fonts / frame), mirroring the author's set-up
gmt set FORMAT_GEO_MAP=dddF \
    MAP_FRAME_PEN=dimgray \
    MAP_FRAME_WIDTH=0.1c \
    MAP_TITLE_OFFSET=1c \
    MAP_ANNOT_OFFSET=0.1c \
    MAP_TICK_PEN_PRIMARY=thinner,dimgray \
    MAP_GRID_PEN_PRIMARY=thin,white \
    MAP_GRID_PEN_SECONDARY=thinner,white \
    FONT_TITLE=14p,Palatino-Roman,black \
    FONT_ANNOT_PRIMARY=8p,Helvetica,black \
    FONT_LABEL=8p,Helvetica,black

# Step-2. Convert the GeoTIFF DEM to a GMT grid (GMT reads it through GDAL)
gmt grdconvert gebco_2026_n42_0_s39_0_w25_0_e31_0_geotiff.tif marmara_relief.nc

# Step-3. Colour palettes:  'geo' for topo/bathy, 'seis' (banded) for magnitude
gmt makecpt -Cgeo  -T-2200/2500          > topo.cpt
gmt makecpt -Cseis -T0/47               > depth.cpt   # shallow=red ... deep=blue

# Step-4. Shaded-relief raster (auto illumination, NW light) -- as in the JT script
gmt grdimage marmara_relief.nc $R $J -Ctopo.cpt -I+a315+ne0.8 -Xc -P -K > $ps

# Step-5. Topographic colour bar
gmt psscale -Dg25/38.4+w16c/0.4c+h+o0/0.4c+ml+e -R -J -Ctopo.cpt \
    --FONT_LABEL=8p,Helvetica,black --FONT_ANNOT_PRIMARY=7p,Helvetica,black \
    -Bxa1000f100+l"Elevation / bathymetry (m)" -By+lm -O -K >> $ps

# Step-6. Coastline + a bathymetric contour
gmt pscoast $R $J -Df -W0.4p,black -I1/0.2p,steelblue -N1/0.3p,gray40 -O -K >> $ps
gmt grdcontour marmara_relief.nc $R $J -C1000 -L-3000/-500 -W0.2p,steelblue4 -O -K >> $ps

# Step-7. Indicative North Anatolian (Main Marmara) Fault trace
gmt psxy naf_trace.txt $R $J -W1.8p,red2,- -O -K >> $ps

# Step-8. Earthquakes: colour by HYPOCENTRE DEPTH (depth.cpt, col 2),
#         symbol size by MAGNITUDE (col 4, cm).  Recipe adapted from GMT-34-MX-seis.sh
#         ( -i column selection / -Sc / -C<cpt> ).  Sort ascending by mag beforehand so
#         large events draw last:  sort -k4 -n marmara_seis.txt > tmp && mv tmp marmara_seis.txt
gmt psxy marmara_seis.txt $R $J -i0,1,2,4 -Sc -Cdepth.cpt -W0.25p,black -t10 -O -K >> $ps

# Step-9. Frame, grid, title
gmt psbasemap $R $J \
    --MAP_FRAME_AXES=WESN \
    -Bpxa1f10mg1 -Bpya30mf2mg1 \
    --MAP_TITLE_OFFSET=0.6c \
    -B+t"Study area: Sea of Marmara and Istanbul region (GEBCO 2026; IEB seismicity 1998-2025)" \
    -O -K >> $ps

# Step-10. Scale bar + directional rose
gmt psbasemap $R $J \
    --FONT=9p,Palatino-Roman,black --MAP_TITLE_OFFSET=0.2c \
    -Lx2c/1.2c+c40.5+w100k+f+l"Mercator projection. Scale (km)" \
    -Tdx14.7c/2.0c+w0.3i+f2+l+o0.15i -O -K >> $ps

# Step-11. Place / sea labels  (white halo, as in the author's maps)
gmt pstext $R $J -N -O -K -F+f11p,Times-Italic,navy+jLM >> $ps << EOF
28.0 40.7667 Sea of Marmara
EOF
gmt pstext $R $J -N -O -K -F+f11p,Times-Italic,white+jCB >> $ps << EOF
29.6 41.7 BLACK SEA
EOF
gmt pstext $R $J -N -O -K -F+f11p,Times-Italic,navy+jCB >> $ps << EOF
25.7 39.45 AEGEAN SEA
EOF
gmt pstext $R $J -N -O -K -F+f9p,Helvetica-Bold,black+jLB -Gwhite@45 >> $ps << EOF
29.0  41.05 Istanbul
29.06 40.12 Bursa
27.40 41.06 Tekirdag
EOF
gmt pstext $R $J -N -O -K -F+f9p,Helvetica-Bold,darkred+jLM -Gwhite@30 >> $ps << EOF
29.0 41.3 1999 Izmit M7.6
EOF
gmt pstext $R $J -N -O -K -F+f8p,Times-Italic,darkred+jCB -Gwhite@55 >> $ps << EOF
27.9 40.95 North Anatolian Fault (Main Marmara Fault)
EOF
gmt pstext $R $J -N -O -K -F+f8p,Times-Italic,navy+jLM+a52 -Gwhite@40 >> $ps << EOF
26.3333 40.25 Dardanelles
EOF

# Step-12a. Depth colour bar for the earthquakes (bottom-left)
gmt psscale -Dx0c/-2.2c+w7c/0.4c+h+ml+e -R -J -Cdepth.cpt \
    --FONT_LABEL=8p,Helvetica,black --FONT_ANNOT_PRIMARY=7p,Helvetica,black \
    -Bxaf+l"Hypocentre depth (km)" -O -K >> $ps

# Step-12b. Magnitude SIZE legend (neutral circles) + fault key (bottom-right)
gmt pslegend $R $J -Dx9.5c/-1.4c+w6.5c+o0/0.1c -F+pthin+gwhite --FONT=8p,black -O -K << FIN >> $ps
H 8 Helvetica Symbol size @~\265@~ magnitude (M) @%0%\267@%% colour = depth
N 4
S 0.25c c 0.18c gray70 0.25p,black 0.6c M 3
S 0.25c c 0.30c gray70 0.25p,black 0.6c M 5
S 0.25c c 0.47c gray70 0.25p,black 0.6c M 7
S 0.25c - 0.5c  -      1.8p,red2,-  0.6c N. Anatolian Fault
FIN

# Step-13. GMT logo + data/source subtitle
gmt logo -Dx0c/-3.6c+o0.1i/0.1i+w2c -O -K >> $ps
gmt pstext -R0/10/0/15 -JX16c/1c -Y-3.9c -N -O \
    -F+f9p,Palatino-Roman,black+jLB >> $ps << EOF
0.0 0.0 DEM: GEBCO 2026 15 arc-sec grid. Earthquakes: EarthScope IEB (IRIS/USGS). Source: authors
EOF

# Step-14. Rasterise with GhostScript (720 dpi JPEG, cropped)
gmt psconvert fig01_study_area.ps -A0.3c -E720 -Tj -Z
# For a vector PDF instead:  gmt psconvert fig01_study_area.ps -A0.3c -Tf
