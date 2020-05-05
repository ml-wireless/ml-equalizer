set term png size 1200,500
set output 'tap2-loss.png'
set multiplot layout 1,2

set xlabel 'epoch'
set ylabel 'loss'
set grid
set yrange [0.0005:0.1]
set logscale y 10

set title '2 tap, order=5, training SNR=10'
plot 'loss-5.dat' u 1 w l t 'training', \
     '' u 2 w l t 'testing', \

set title '2 tap, order=31, training SNR=10'
plot 'loss-31.dat' u 1 w l t 'training', \
     '' u 2 w l t 'testing', \
