# Data Audit Report

**Dataset:** data/raw/Car details v3.csv
**Shape:** 8128 rows x 13 columns

## Column Info

| Column | Dtype | Non-Null | Missing | Missing % | Unique |
|--------|-------|----------|---------|-----------|--------|
| name | str | 8128 | 0 | 0.0% | 2058 |
| year | int64 | 8128 | 0 | 0.0% | 29 |
| selling_price | int64 | 8128 | 0 | 0.0% | 677 |
| km_driven | int64 | 8128 | 0 | 0.0% | 921 |
| fuel | str | 8128 | 0 | 0.0% | 4 |
| seller_type | str | 8128 | 0 | 0.0% | 3 |
| transmission | str | 8128 | 0 | 0.0% | 2 |
| owner | str | 8128 | 0 | 0.0% | 5 |
| mileage | str | 7907 | 221 | 2.7% | 393 |
| engine | str | 7907 | 221 | 2.7% | 121 |
| max_power | str | 7913 | 215 | 2.6% | 322 |
| torque | str | 7906 | 222 | 2.7% | 441 |
| seats | float64 | 7907 | 221 | 2.7% | 9 |

## Duplicates

1202 duplicate rows found (14.8%).

## Numeric Summary

```
              year  selling_price     km_driven        seats
count  8128.000000   8.128000e+03  8.128000e+03  7907.000000
mean   2013.804011   6.382718e+05  6.981951e+04     5.416719
std       4.044249   8.062534e+05  5.655055e+04     0.959588
min    1983.000000   2.999900e+04  1.000000e+00     2.000000
25%    2011.000000   2.549990e+05  3.500000e+04     5.000000
50%    2015.000000   4.500000e+05  6.000000e+04     5.000000
75%    2017.000000   6.750000e+05  9.800000e+04     5.000000
max    2020.000000   1.000000e+07  2.360457e+06    14.000000
```

## Target Variable (selling_price)

- Mean: Rs.638,272
- Median: Rs.450,000
- Min: Rs.29,999
- Max: Rs.10,000,000
- Std: Rs.806,253

## Categorical Columns

### name
```
name
Maruti Swift Dzire VDI                    129
Maruti Alto 800 LXI                        82
Maruti Alto LXi                            71
BMW X4 M Sport X xDrive20d                 62
Maruti Swift VDI                           61
Maruti Swift VDI BSIV                      59
Maruti Wagon R LXI                         53
Maruti Alto K10 VXI                        50
Hyundai EON Era Plus                       48
Maruti Wagon R VXI BS IV                   45
Maruti Ertiga VDI                          45
Maruti Alto LX                             44
Toyota Innova 2.5 VX (Diesel) 7 Seater     44
Maruti Ritz VDi                            42
Maruti Swift Dzire VXI                     38
```

### fuel
```
fuel
Diesel    4402
Petrol    3631
CNG         57
LPG         38
```

### seller_type
```
seller_type
Individual          6766
Dealer              1126
Trustmark Dealer     236
```

### transmission
```
transmission
Manual       7078
Automatic    1050
```

### owner
```
owner
First Owner             5289
Second Owner            2105
Third Owner              555
Fourth & Above Owner     174
Test Drive Car             5
```

### mileage
```
mileage
18.9 kmpl     225
19.7 kmpl     173
18.6 kmpl     164
21.1 kmpl     157
17.0 kmpl     133
15.96 kmpl    115
16.1 kmpl     111
22.0 kmpl     111
17.8 kmpl     108
12.99 kmpl    105
18.2 kmpl      95
16.8 kmpl      95
16.78 kmpl     95
12.8 kmpl      94
28.4 kmpl      93
```

### engine
```
engine
1248 CC    1017
1197 CC     832
998 CC      453
796 CC      444
2179 CC     389
1498 CC     375
1396 CC     321
1199 CC     271
2494 CC     222
2523 CC     195
1461 CC     185
1198 CC     182
1497 CC     181
1582 CC     156
1493 CC     142
```

### max_power
```
max_power
74 bhp       377
81.80 bhp    220
88.5 bhp     204
67 bhp       165
46.3 bhp     162
67.1 bhp     151
88.7 bhp     148
67.04 bhp    145
47.3 bhp     141
62.1 bhp     141
70 bhp       141
88.73 bhp    132
81.86 bhp    131
120 bhp      131
140 bhp      128
```

### torque
```
torque
190Nm@ 2000rpm           530
200Nm@ 1750rpm           445
90Nm@ 3500rpm            405
113Nm@ 4200rpm           223
114Nm@ 4000rpm           171
62Nm@ 3000rpm            164
69Nm@ 3500rpm            141
160Nm@ 2000rpm           138
330Nm@ 1600-2800rpm      122
74.5Nm@ 4000rpm          121
59Nm@ 2500rpm            112
110Nm@ 4800rpm           112
219.7Nm@ 1500-2750rpm    101
195Nm@ 1400-2200rpm      101
114.7Nm@ 4000rpm          97
```
