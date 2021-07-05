from os import path

DATA_PATH = path.join('..', 'data')
RESULTS_PATH = path.join('..', 'results')
PLOTS_PATH = path.join('..', 'plots')

# smartwatches ids
devices = [
    '14df',
    '38b8',
    'a650',
    'd884'
]

# beacons ids
ibks105_D0 = [
    'F0:C1:64:9B:71:8A', 'DF:47:ED:03:9C:24', 'D3:30:0D:56:C7:70', 'C7:F8:84:6B:03:A4', 'DA:4A:C9:76:10:39',
    'C7:9F:30:FE:E6:ED', 'C5:53:84:B9:11:46', 'EC:8A:32:11:DB:F3', 'F8:5F:B8:BE:8C:76', 'E5:B6:B3:71:CF:B1'
]

# NO -> 3B:43:A1:9E:5E:9A
ibksplus_D0 = [
    'E1:A3:C1:C7:3F:1A', 'EE:67:B9:E8:A8:87', '3B:43:A1:9E:5E:9A', 'FD:E8:09:40:4E:6A', 'EE:C8:47:1D:2F:9B',
    'F5:65:FB:0C:D8:10', 'E3:88:8A:F5:83:2C', 'C8:30:3E:55:3E:27', 'E1:48:EA:70:98:5E', 'F8:EC:F7:70:78:B9'
]


UJI_D0 = []
ALL_D0 = ['FA:01:A7:EB:C4:5B']
for i in range(len(ibks105_D0)):
    ALL_D0.append(ibks105_D0[i])
    UJI_D0.append(ibks105_D0[i])
    if i < len(ibksplus_D0):
        ALL_D0.append(ibksplus_D0[i])
        UJI_D0.append(ibksplus_D0[i])

beacon_separation = 300.0
