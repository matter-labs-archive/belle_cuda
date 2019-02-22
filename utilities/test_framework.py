p = 21888242871839275222246405745257275088696311157297823662689037894645226208583
r = 21888242871839275222246405745257275088548364400416034343698204186575808495617

base_field = GF(p)
curve = EllipticCurve(base_field, [0, 3]);
G = curve(1, 2, 1)

R = base_field(2 ^ 256)

def to_mont_form(x):
    return x * R

def from_mont_form(x):
    return x / R

def parse_affine_point(line1, line2):
    x = int(line1.split('=')[1], 0x10)
    y = int(line2.split('=')[1], 0x10)
    
    return curve(x, y, R)

def parse_bignum(line):
    return int(line, 0x10)

def parse_ec_point(line1, line2, line3):
    x = int(line1.split('=')[1], 0x10)
    y = int(line2.split('=')[1], 0x10)
    z = int(line3.split('=')[1], 0x10)
    
    return curve(x, y, z)

A_arr = []
B_arr = []
C_arr = []
bench_len = 10
FILE_LOCATION = "/home/k/TestCuda3/benches.txt"

file = open(FILE_LOCATION, "r")

print file.readline()

for _ in xrange(bench_len):
    x = file.readline()[:-1]
    y = file.readline()[:-1]
    file.readline()
    
    A_arr.append(parse_affine_point(x, y))
    
print file.readline()
print file.readline()

for _ in xrange(bench_len):
    num = file.readline()[:-1]
    B_arr.append(parse_bignum(num))
    
print file.readline()
print file.readline()

for _ in xrange(bench_len):
    x = file.readline()[:-1]
    y = file.readline()[:-1]
    z = file.readline()[:-1]
    file.readline()
    
    C_arr.append(parse_ec_point(x, y, z))
    
    
for i in xrange(bench_len):
    print (B_arr[i] * A_arr[i] == C_arr[i])
    

