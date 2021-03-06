set term pdf
set out 'search_compare.pdf'

set title 'Search Comparison (using 5 randomized)'
set grid

set datafile separator ","
set logscale y

set xlabel 'Number of cores'
set ylabel 'Time [s]'

plot '../search10k.txt' u 1:2 with lp lw 2 lt 1 pt 1 title 'Our Code (10k vectors)', '' u 1:2:4 with yerrorbars lt 1 notitle, \
     '../search10k.txt' u 1:5 with lp lw 2 lt 2 pt 2 title 'FLANN (10k vectors)', '' u 1:5:7 with yerrorbars lt 2 notitle, \
     '../search1M.txt' u 1:2 with lp lw 2 lt 3 pt 3 title 'Our Code (1M vectors)', '' u 1:2:4 with yerrorbars lt 3 notitle, \
     '../search1M.txt' u 1:5 with lp lw 2 lt 4 pt 4 title 'FLANN (1M vectors)', '' u 1:5:7 with yerrorbars lt 4 notitle, \
     14.562899999999999 with l lw 2 lt 8 title 'XeonPhi (240 Threads, warm cache)'
