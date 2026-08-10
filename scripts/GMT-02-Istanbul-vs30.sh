#!/usr/bin/env bash
# GMT-02-Istanbul-vs30.sh
# fig02: Istanbul metropolitan area site characterisation.
#   (a) Vs30 (m/s) over shaded relief + AFAD station measurements
#   (b) EC8/NEHRP site classes derived from the Vs30 grid
# Requires GMT 6+ (modern mode). Provide your own input grids/CSV (see below).
# ----------------------------------------------------------------------------
set -e
R=28/30/40.5/41.5            # Istanbul metropolitan window
J=M15c

# ---- inputs you supply (do NOT fabricate station data) ---------------------
VS30=vs30_istanbul.grd       # USGS slope-based Vs30 grid, clipped to $R:
                             #   gmt grdcut global_vs30.grd -R$R -G$VS30
DEM=istanbul_dem.grd         # DEM for shading (e.g. GEBCO), clipped to $R:
                             #   gmt grdcut gebco.nc -R$R -G$DEM
AFAD=afad_vs30.txt           # AFAD stations, columns: lon lat vs30

gmt grdgradient $DEM -A315 -Nt0.8 -Gshade.grd

gmt begin GMT-02-Istanbul-vs30 pdf,png
  gmt set FONT_TITLE 12p FONT_ANNOT_PRIMARY 8p MAP_FRAME_TYPE plain

  gmt subplot begin 2x1 -Fs15c -M0.5c \
      -T"Site characterisation of the Istanbul metropolitan area"

    # ---- (a) Vs30 + AFAD stations ----
    gmt subplot set 0
      gmt makecpt -Cjet -I -T180/760/20 -Z
      gmt grdimage $VS30 -R$R -J$J -Ishade.grd -B+t"(a) Vs30 and AFAD stations" \
          -Bxa0.5f10m -Bya0.25f5m -BWSne
      gmt coast -R$R -J$J -W0.5p,black -Sskyblue
      [ -f "$AFAD" ] && gmt plot $AFAD -i0,1,2 -Sc0.20c -C -W0.4p,black
      gmt colorbar -DJMR+o0.6c/0c+w8c -Bx+l"Vs30 (m s@+-1@+)"
      gmt basemap -LjBR+w50k+o0.7c/0.7c+f

    # ---- (b) EC8/NEHRP site classes ----
    gmt subplot set 1
      # categorical CPT at class thresholds 180/360/760
      gmt makecpt -C215/25/28,253/174/97,43/131/186 -T180,360,760,900 -F+cD,C,B
      gmt grdimage $VS30 -R$R -J$J -Ishade.grd -B+t"(b) EC8/NEHRP site classes" \
          -Bxa0.5f10m -Bya0.25f5m -BWSne
      gmt coast -R$R -J$J -W0.5p,black -Sskyblue
      gmt colorbar -DJMR+o0.6c/0c+w8c -Bx+l"Site class"

  gmt subplot end
gmt end show

# Site-class thresholds (Vs30, m/s): EC8 C/NEHRP D 180-360 | EC8 B/NEHRP C 360-760
#                                    | EC8 A-B/NEHRP B >=760.
# Vs30 from the USGS global slope-based model (Wald & Allen, 2007). Source: authors.
