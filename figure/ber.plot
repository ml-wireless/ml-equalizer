set term png
set output 'tap2-ber.png'

set title '2 tap BER'
set xlabel 'SNR'
set ylabel 'BER'
set xrange [-3.2:30.2]
# set yrange [0.001:1]
set grid
set xtic 6
set logscale y 10

plot 'classic-ber.dat' u 1:2 w l t 'ZFE', \
     '' u 1:3 w l t 'MMSE', \
     '' u 1:4 w l t 'LMS', \
     'cnn-ber.dat' u 1:2 w l t 'CNN', \
     #'hybrid-ber.dat' u 1:2 w l t 'Hybrid', \
     
