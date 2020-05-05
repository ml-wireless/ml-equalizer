set term png size 1200,500
set output 'tap2-ber-split.png'
set multiplot layout 1,2

set key at 1, 0.6 horizontal 
set xlabel 'Split'
set ylabel 'BER'
set xrange [0.1:1]
# set yrange [0.001:1]
set grid
set xtic 0.2
set logscale y 10

set title '2 tap, order=5, training SNR=10'
plot 'split-ber.dat' u 1:2 w l t 'SNR=3', \
    '' u 1:3 w l t 'SNR=6', \
    '' u 1:4 w l t 'SNR=9', \
    '' u 1:5 w l t 'SNR=12', \
    '' u 1:6 w l lt 7 t 'SNR=15', \
    '' u 1:7 w l t 'SNR=18', \

set title '2 tap, order=31, training SNR=10'
plot 'split-ber.dat' u 1:8 w l t 'SNR=3', \
    '' u 1:9 w l t 'SNR=6', \
    '' u 1:10 w l t 'SNR=9', \
    '' u 1:11 w l t 'SNR=12', \
    '' u 1:12 w l lt 7 t 'SNR=15', \
    '' u 1:13 w l t 'SNR=18', \