set term pdf
set out 'build_speedup.pdf'

set title 'Tree Build Speedup versus FLANN'
set grid

set datafile separator ","
# set logscale y

set key top left

set xlabel 'Number of cores'
set ylabel 'Speedup'

plot '../build10k.txt' u 1:8 with lp lw 2 lt 1 pt 1 title '10k vectors', '' u 1:8:9 with yerrorbars lt 1 notitle, \
     '../build1M.txt' u 1:8 with lp lw 2 lt 3 pt 3 title '1M vectors', '' u 1:8:9 with yerrorbars lt 3 notitle
