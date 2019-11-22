set term png
set output 'tap2-ber.png'

set title '2 tap BER'
set xlabel 'log SNR'
set ylabel 'log Error'
set xrange [-0.2:4]
set yrange [-8.2:0.2]
set xtic 0.5
set ytic 1

set style line 1 lc rgb 'orange' pt 7
set style line 2 lc rgb 'blue' pt 7

plot 'tap2-ber.dat' u (log10($1)):(log10($2)) w p ls 1 t 'RNN',\
     'zf-est-ber.dat' u (log10($1)):(log10($2)) w p ls 2 t 'Zero Forcing'
