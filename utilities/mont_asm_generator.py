def generate_mont_asm_Listing(asm_len):
    printed_asm = ""
    for i in xrange(asm_len):
        printed_asm += 'mul.lo.u32   m, a{:d}, q;\\n\\t\"\n'.format(i)
        first = True
        for j in xrange(asm_len):
            if first:
                printed_asm += "mad."
                first = False
            else:
                printed_asm += "madc."
            printed_asm += 'lo.cc.u32  a{:d}, m, n{:d}, a{:d};\\n\\t\"\n'.format(i+j, j, i+j)
        j = i + asm_len
        while (j < 2 * asm_len):
            if (j < 2 * asm_len - 1):
                printed_asm += 'addc.cc.u32  a{:d}, a{:d}, 0;\\n\\t\"\n'.format(j, j)
            else:
                printed_asm += 'add.cc.u32  a{:d}, a{:d}, 0;\\n\\t\"\n'.format(j, j)
            j = j + 1
        first = True
        for j in xrange(asm_len):
            if first:
                printed_asm += "mad."
                first = False
            else:
                printed_asm += "madc."
            printed_asm += 'hi.cc.u32  a{:d}, m, n{:d}, a{:d};\\n\\t\"\n'.format(i+j+1, j, i+j+1)
        j = i + asm_len + 1
        while (j < 2 * asm_len):
            if (j < 2 * asm_len - 1):
                printed_asm += 'addc.cc.u32  a{:d}, a{:d}, 0;\\n\\t\"\n'.format(j, j)
            else:
                printed_asm += 'add.cc.u32  a{:d}, a{:d}, 0;\\n\\t\"\n'.format(j, j)
            j = j + 1
    return printed_asm
    
print generate_mont_asm_Listing(8)