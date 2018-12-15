def is_in_range(x, shift, array_len):
    flag = (x >= shift and x < array_len + shift)
    return flag

def gen_sublists(idx, array_len):
    sublists = []
    if (idx > 0):
        for j in xrange(min(idx, array_len)):
            a = idx - 1 - j + 2 * array_len
            b = j + 3 * array_len
            if is_in_range(a, 2 * array_len, array_len) and is_in_range(b, 3 * array_len, array_len):
                sublists.append((a, b, True))
    for j in xrange(min(idx, array_len) + 1):
        a = idx - j + 2 * array_len
        b = j + 3 * array_len
        if is_in_range(a, 2 * array_len, array_len) and is_in_range(b, 3 * array_len, array_len):
            sublists.append((a, b, False))
    return sublists

from collections import namedtuple

#c = a*b + d
AsmInsn = namedtuple("AsmInsn", "gen_carry use_carry op_type is_high a b c d")
op_mul = 0
op_mad = 1
op_add = 2
op_unknown = 3

def gen_table(array_len):
    table = {}
    table_len = 0
    for i in xrange(array_len * 2):
        arr = gen_sublists(i, array_len)
        table_len += len(arr)
        table[i] = arr
    return table, table_len


def gen_asm(array_len):
    
    carry_arr = [False] * (2 * array_len)
    AsmListing = []
    
    table, table_len = gen_table(array_len)
    lowest_index = 0
    cur_index = 0
    use_carry = False
    while(table_len > 0 or use_carry):
        if cur_index >= 2 * array_len:
            use_carry = False
            
            if table_len == 0:
                break
            #try to find next suitable index
            while not table[lowest_index]:
                lowest_index = lowest_index + 1
            cur_index = lowest_index        
        
        elif carry_arr[cur_index]:
            gen_carry = True
            
            if table[cur_index]:           
                (a, b, is_high) = table[cur_index][0]
                table[cur_index].pop(0)
                table_len = table_len - 1
                op_type = op_mad
            else:
                (a, b, is_high) = (cur_index, cur_index, False)
                op_type = op_add
            
            insn = AsmInsn(gen_carry, use_carry, op_type, is_high, a, b, cur_index, cur_index)
            AsmListing.append(insn)
            
            use_carry = True
            cur_index = cur_index + 1
            
        else:
            with_addition = use_carry
            gen_carry = False
            
            (a, b, is_high) = table[cur_index][0]
            table[cur_index].pop(0)
            table_len = table_len - 1
            
            insn = AsmInsn(gen_carry, use_carry, with_addition, is_high, a, b, cur_index, -1)
            AsmListing.append(insn)
            
            carry_arr[cur_index] = True
            use_carry = False
            
            if table_len == 0:
                break
            #try to find next suitable index
            while not table[lowest_index]:
                lowest_index = lowest_index + 1
            cur_index = lowest_index
        
    return AsmListing


def generate_printable_asm(AsmListing):
    printed_asm = ""
    #print len(AsmListing)
    for elem in AsmListing:
        high_low = "hi." if elem.is_high else "lo."
        if (elem.op_type == op_mul):         
            printed_asm += "\"mul." + high_low + "u32";
            printed_asm += '   %{:d}, %{:d}, %{:d};\\n\\t\"\n'.format(elem.c, elem.a, elem.b)
        elif (elem.op_type == op_mad):
            printed_asm += "\"mad"
            if (elem.use_carry):
                printed_asm += "c"
            printed_asm += "." + high_low
            if (elem.gen_carry):
                printed_asm += "cc."
            printed_asm += "u32" + '   %{:d}, %{:d}, %{:d}, '.format(elem.c, elem.a, elem.b)
            ending = "0;\\n\\t\"\n" if elem.d == -1 else '%{:d};\\n\\t\"\n'.format(elem.d)
            printed_asm += ending
        elif (elem.op_type == op_add):
            printed_asm += "\"add"
            if (elem.use_carry):
                printed_asm += "c"
            printed_asm += "."
            if (elem.gen_carry):
                printed_asm += "cc."
            printed_asm += "u32" + '   %{:d}, %{:d}, 0;\\n\\t\"\n'.format(elem.c, elem.a)
        else:
            raise ValueError('Incorrect operand type.')
     
    return printed_asm

    

AsmListing = gen_asm(4)
print generate_printable_asm(AsmListing)