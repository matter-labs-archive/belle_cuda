p = 21888242871839275222246405745257275088548364400416034343698204186575808495617
max_k = 28

field = GF(p)

gen = field.multiplicative_generator()

R = field(2^256)

a = (p - 1) / (2 ^ max_k)

root_of_unity = gen ^ a

def splitter(x):
    x =  hex(int(x * R))[2:-1]
    str_len = len(x)
    if str_len % 8 != 0:
        x = "0" * (8 - (str_len % 8)) + x
        
    res =  ["0x" + x[i:i+8] for i in range(0, len(x), 8)]
    return res[::-1]
    
def printer(x):
    res = "{ "
    for j in xrange(8):
        res += x[j]
        if j != 7:
                res += ", "
    res += " };"
    return res

x = root_of_unity

for i in xrange(max_k):
    ww = splitter(x)
    print printer(ww)
    x *= x