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

def parse_bignum(line, base = 0x10):
    return int(line, base)

def parse_ec_point(line1, line2, line3):
    x = int(line1.split('=')[1], 0x10)
    y = int(line2.split('=')[1], 0x10)
    z = int(line3.split('=')[1], 0x10)
    
    return curve(x, y, z)


def extractKBits(num,k,p):   
    num = num >> p
    num = num & (2^k - 1)
    return num


pt_arr = []

FILE_LOCATION = "/home/k/TestCuda3/benches.txt"

file = open(FILE_LOCATION, "r")

bench_len = parse_bignum(file.readline().split("=")[1][:-1], 10)
print bench_len


num_of_results = 2

for _ in xrange(num_of_results):
    file.readline()
    file.readline()
    
    x = file.readline()[:-1]
    y = file.readline()[:-1]
    z = file.readline()[:-1]
    file.readline()    
    C = parse_ec_point(x, y, z)
    pt_arr.append(C)

print pt_arr[0] == pt_arr[1]
print len(set(pt_arr)) == 1

time1 = 85375141521
time2 = 12400215642

print float(time1 / time2)