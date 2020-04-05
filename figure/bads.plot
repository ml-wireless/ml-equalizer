set term png
set output 'tap2-bads.png'
set size square

set title '2 tap bad channels'
set xlabel 'a0'
set ylabel 'a1'
set xrange [-1.3:1.3]
set yrange [-1.3:1.3]
set grid

plot 'cnn-bads.dat' u 1:2 w p t 'CNN',\
