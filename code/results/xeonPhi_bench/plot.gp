set term pdf
set out 'phi_bandwidth.pdf'

set logscale x 2
set format x '2^{%L}'

set xrange [2**6:2**28]
set yrange [0:4.5]

# set x2tics ( \
# "6.1e-5"   2**6  ,\
# "1.2e-4"   2**7  ,\
# "2.4e-4"   2**8  ,\
# "4.9e-4"   2**9  ,\
# "9.8e-4"   2**10 ,\
# "2.0e-3"   2**11 ,\
# "3.9e-3"   2**12 ,\
# "7.8e-3"   2**13 ,\
# "1.6e-2"   2**14 ,\
# "3.1e-2"   2**15 ,\
# "6.3e-2"   2**16 ,\
# "0.125"    2**17 ,\
# "0.25"     2**18 ,\
# "0.5"      2**19 ,\
# "1"        2**20 ,\
# "2"        2**21 ,\
# "4"        2**22 ,\
# "8"        2**23 ,\
# "16"       2**24 ,\
# "32"       2**25 ,\
# "64"       2**26 ,\
# "128"      2**27 ,\
# "256"      2**28)

set grid
set key bottom right

set xlabel 'Transfer size [Byte]'
set ylabel 'Bandwidth [GB/s]'

plot 'h2d.dat' u 2:6 w lp lw 2 pt 1 title 'Host to Device', \
     'd2h.dat' u 2:6 w lp lw 2 pt 2 title 'Device to Host', \
     4.0 axes x2y1 w l lw 2 title 'Theoretical Maximum'
