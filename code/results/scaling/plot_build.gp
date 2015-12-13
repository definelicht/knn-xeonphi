set term pdf
set out 'build_compare.pdf'

set title 'Tree Build Comparison (5 randomized)'
set grid

set logscale y

set xlabel 'Number of cores'
set ylabel 'Time [s]'

plot './merge_build10k.dat' u 1:2:4 w lp lw 2 pt 1 yerrorbars title 'Our Code (10k vectors)', \
     './merge_build10k.dat' u 1:5:7 w lp lw 2 pt 2 yerrorbars title 'FLANN (10k vectors)', \
     './merge_build1M.dat'  u 1:2:4 w lp lw 2 pt 3 yerrorbars title 'Our Code (1M vectors)', \
     './merge_build1M.dat'  u 1:5:7 w lp lw 2 pt 4 yerrorbars title 'FLANN (1M vectors)'
