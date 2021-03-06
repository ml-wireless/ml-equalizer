set term png
set output 'tap2-ber.png'

set title '2 tap BER'
set xlabel 'log SNR'
set ylabel 'log Error'
set xrange [-0.2:4]
set yrange [-5.2:0.2]
set xtic 0.5
set ytic 1

set style line 1 lc rgb 'orange' pt 7
set style line 2 lc rgb 'blue' pt 7
set style line 3 lc rgb 'green' pt 7
set style line 4 lc rgb 'magenta' pt 7
set style line 5 lc rgb 'red' pt 7

plot 'tap2-ber.dat' u (log10($1)):(log10($2)) w p ls 1 t 'Neural Est + RNN',\
     'zf-mmse-ber.dat' u (log10($1)):(log10($2)) w p ls 2 t 'Zero Forcing',\
     'zf-mmse-ber.dat' u (log10($1)):(log10($3)) w p ls 3 t 'MMSE',\
     'lms-ber.dat' u (log10($1)):(log10($2)) w p ls 4 t 'LMS',\
     'cnn.dat' u (log10($1)):(log10($2)) w p ls 5 t 'CNN'
     
