P = [ 0xd87cfd47, 0x3c208c16, 0x6871ca8d, 0x97816a91, 0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72 ]
N = 0xe4866389

word_len = 2^32



def CIOS(A, B):
    S = [0, 0, 0, 0, 0, 0, 0, 0]
    
    for j in xrange(2):
        for i in xrange(8):
            S[i] += A[i] * B[j]
        
        q = (S[0] * N) % word_len
        for i in xrange(8):
            S[i] += q * P[i]
        
        for i in xrange(7):
            S[i] = (S[i] >> 32) + (S[i+1] % word_len)
            
        S[7] = (S[7] >> 32)       
        
    temp = 0
    res = []
    for i in xrange(8):
        S[i] = S[i] + temp
        res.append(hex(S[i] % word_len))
        temp = (S[i] >> 32)
           
    return res


def splitter(num):
    res = []
    for i in xrange(8):
        res.append(num % word_len)
        num = num >> 32
        
    return res