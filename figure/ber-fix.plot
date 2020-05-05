set term png size 1200,1000
set output 'tap2-ber-fix.png'
set multiplot layout 2,2

set key at 30, 0.3
set xlabel 'SNR'
set ylabel 'BER'
set xrange [-3.2:30.2]
set yrange [0.001:1]
set grid
set xtic 6
set logscale y 10

set title 'order=5, tap=[0.99 0.11], training SNR=10'
plot 'fix-ber.dat' u 1:2 w l t 'ZFE', \
     '' u 1:3 w l t 'MMSE', \
     '' u 1:4 w l t 'LMS', \
     '' u 1:5 w l t 'Hybrid', \

set title 'order=5 tap=[0.73 -0.68], training SNR=10'
plot 'fix-ber.dat' u 1:10 w l t 'ZFE', \
     '' u 1:11 w l t 'MMSE', \
     '' u 1:12 w l t 'LMS', \
     '' u 1:13 w l t 'Hybrid', \

set title 'order=31 tap=[0.99 0.11], training SNR=10'
plot 'fix-ber-31.dat' u 1:2 w l t 'ZFE', \
     '' u 1:3 w l t 'MMSE', \
     '' u 1:4 w l t 'LMS', \
     '' u 1:5 w l t 'Hybrid', \

set title 'order=31 tap=[0.73 -0.68], training SNR=10'
plot 'fix-ber-31.dat' u 1:10 w l t 'ZFE', \
     '' u 1:11 w l t 'MMSE', \
     '' u 1:12 w l t 'LMS', \
     '' u 1:13 w l t 'Hybrid', \
