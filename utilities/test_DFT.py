#FFT

p = 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001
field = GF(p)
R = field(0xe0a77c19a07df2f666ea36f7879462e36fc76959f60cd29ac96341c4ffffffb)
root_of_unity = field(0x1860ef942963f9e756452ac01eb203d8a22bf3742445ffd6636e735580d13d9c) / R
FILE_LOCATION = "/home/k/TestCuda3/benches.txt"

unity_order = root_of_unity.multiplicative_order()

#all elements of arr are given in Montgomery form

def from_mont_form(arr):
    return  [x / R for x in arr]


def to_mont_form(arr):
    return [x * R for x in arr]


def DFT(arr):
    n = len(arr)
    omega = root_of_unity ^ (unity_order / n)
    res = []
    
    temp_arr = from_mont_form(arr)
    
    
    for i in xrange(n):
        temp = field(0)
        for j, elem in enumerate(temp_arr):
            temp += elem * omega ^ (i * j)
        res.append(temp)
    
    return to_mont_form(res)
    #return [hex(int(x)) for x in to_mont_form(res)]


def parse_bignum(line, base = 0x10):
    return int(line, base)


def read_arr(bench_len, file):
    arr = []
    
    file.readline()
    file.readline()

    for _ in xrange(bench_len):
        line = file.readline()
        num = parse_bignum(line)
        arr.append(field(num))
        
    return arr

    

def sample_from_file():
    file = open(FILE_LOCATION, "r")

    bench_len = parse_bignum(file.readline().split("=")[1][:-1], 10)
    print bench_len
    
    A = read_arr(bench_len, file)
    B = read_arr(bench_len, file)
    C = read_arr(bench_len, file)
    
    D = DFT(A)
    for i in xrange(bench_len):
        if D[i] != C[i]:
            print "arrs are different"
            
    print "finish"
    

    
    
sample_from_file()
    
