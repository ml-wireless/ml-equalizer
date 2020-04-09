set term png size 1200,450
set output 'tap2-ber.png'
set multiplot layout 1,2

set title '2 tap, order=31'
set key at 30, 0.3
set xlabel 'SNR'
set ylabel 'BER'
set xrange [-3.2:30.2]
# set yrange [0.001:1]
set grid
set xtic 6
set logscale y 10

plot 'classic-ber.dat' u 1:2 w l t 'ZFE', \
     '' u 1:3 w l t 'MMSE', \
     'lms-ber.dat' u 1:3 w l t 'LMS', \
     'cnn-ber.dat' u 1:3 w l t 'CNN', \

set title '2 tap, order=5'
plot 'classic-ber.dat' u 1:4 w l t 'ZFE', \
     '' u 1:5 w l t 'MMSE', \
     'lms-ber.dat' u 1:2 w l t 'LMS', \
     'cnn-ber.dat' u 1:2 w l t 'CNN', \
