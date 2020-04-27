set term png size 600,500
set output 'tap2-ber-cnn.png'

set key at 30, 0.3
set xlabel 'SNR'
set ylabel 'BER'
set xrange [-3.2:30.2]
# set yrange [0.001:1]
set grid
set xtic 6
set logscale y 10

set title 'order=5'
plot 'classic-ber.dat' u 1:4 w l t 'Zero Forcing', \
     '' u 1:5 w l t 'MMSE', \
     'lms-ber.dat' u 1:2 w l t 'LMS', \
     'cnn-ber.dat' u 1:2 w l t 'CNN w/ LMS data ', \
     'cnn-ber.dat' u 1:4 w l t 'CNN w/ true inverse', \
