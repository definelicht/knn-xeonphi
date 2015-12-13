set term png enhanced large
set out 'build_compare.png'

set title 'Tree Build Comparison (5 randomized)'
set grid

set logscale y

set xlabel 'Number of cores'
set ylabel 'Time [s]'

plot './merge_build10k.dat' u 1:2 w lp lw 2 pt 1 title 'Our Code (10k vectors)', \
     './merge_build10k.dat' u 1:3 w lp lw 2 pt 2 title 'FLANN (10k vectors)', \
     './merge_build1M.dat'  u 1:2 w lp lw 2 pt 3 title 'Our Code (1M vectors)', \
     './merge_build1M.dat'  u 1:3 w lp lw 2 pt 4 title 'FLANN (1M vectors)'
